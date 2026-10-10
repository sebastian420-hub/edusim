// Hodgkin–Huxley cables: one invocation advances one whole fibre by `steps` steps of Δt. Per step: the gates
// relax at the old voltage (Rush–Larsen), then the voltages are solved implicitly along the fibre with the
// Thomas algorithm (stable for any Δt), as stepCable() in cable.ts, which the GPU tests compare against.
//
// Units: mV, ms, cm; conductances mS, capacitances µF, currents µA.

struct Params {
  fibres: u32,
  steps: u32,
  histEvery: u32, // a kymograph row of fibre 0 every this many steps (0 = off)
  histRows: u32,
  stepIndex: u32, // global index of the first step of this dispatch
  sigSteps: u32, // per-fibre extracellular signal is written for this many steps (0 = off)
  histOffset: u32, // where the kymograph starts in `out`
  sigOffset: u32, // where the per-fibre signals start in `out`
  time: f32, // ms at the start of the dispatch
  dt: f32,
  phi: f32, // Q10 temperature factor of the gates
  stimDuration: f32,
  startA: f32, // stimulus A and B start times (ms; negative = never)
  startB: f32,
  electrode: f32, // recording electrode position (cm) and distance from the fibres (cm)
  height: f32,
  probeA: u32, // fibre-0 compartments whose voltage is recorded every step (two traces)
  probeB: u32,
  probeOffset: u32, // where the traces start in `out` (pairs, a ring of probeLen)
  probeLen: u32,
}

struct Fibre {
  offset: u32,
  count: u32,
  pad0: u32,
  pad1: u32,
}

// Static per compartment: capacitance (µF), axial conductance to the next compartment (mS), position (cm),
// stimulus A current (µA); maximal Na, K and leak conductances (mS), stimulus B current (µA).
struct Comp {
  a: vec4f,
  b: vec4f,
}

// Dynamic per compartment: (V, m, h, n) and (c′, d′ of the solve, first arrival time, crossings).
struct Dyn {
  s: vec4f,
  r: vec4f,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> fibres: array<Fibre>;
@group(0) @binding(2) var<storage, read> comps: array<Comp>;
@group(0) @binding(3) var<storage, read_write> dyn: array<Dyn>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const E_NA: f32 = 50.0;
const E_K: f32 = -77.0;
const E_L: f32 = -54.387;
const THRESHOLD: f32 = -20.0;

fn relax(x: f32, a: f32, b: f32) -> f32 {
  let inf = a / (a + b);
  return inf + (x - inf) * exp(-params.dt * (a + b));
}

fn stimulus(c: Comp, t: f32) -> f32 {
  var i = 0.0;
  if (params.startA >= 0.0 && t + 1e-6 >= params.startA && t + 1e-6 < params.startA + params.stimDuration) { i += c.a.w; }
  if (params.startB >= 0.0 && t + 1e-6 >= params.startB && t + 1e-6 < params.startB + params.stimDuration) { i += c.b.w; }
  return i;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let f = gid.x;
  if (f >= params.fibres) { return; }
  let first = fibres[f].offset;
  let n = fibres[f].count;
  let last = first + n - 1u;
  let dt = params.dt;
  let phi = params.phi;

  for (var k = 0u; k < params.steps; k++) {
    let t = params.time + f32(k) * dt;
    // Forward sweep: gates, then assemble and eliminate row i.
    var prevC = 0.0;
    var prevD = 0.0;
    for (var i = first; i <= last; i++) {
      let c = comps[i];
      var s = dyn[i].s;
      let V = s.x;
      let vs = V + 65.0;
      var am = 0.1 * (25.0 - vs) / (exp((25.0 - vs) / 10.0) - 1.0);
      if (abs(vs - 25.0) < 0.001) { am = 1.0; }
      let bm = 4.0 * exp(-vs / 18.0);
      let ah = 0.07 * exp(-vs / 20.0);
      let bh = 1.0 / (exp((30.0 - vs) / 10.0) + 1.0);
      var an = 0.01 * (10.0 - vs) / (exp((10.0 - vs) / 10.0) - 1.0);
      if (abs(vs - 10.0) < 0.001) { an = 0.1; }
      let bn = 0.125 * exp(-vs / 80.0);
      s.y = relax(s.y, am * phi, bm * phi);
      s.z = relax(s.z, ah * phi, bh * phi);
      s.w = relax(s.w, an * phi, bn * phi);
      let gNa = c.b.x * s.y * s.y * s.y * s.z;
      let gK = c.b.y * s.w * s.w * s.w * s.w;
      let gL = c.b.z;
      let C = c.a.x / dt;
      var lower = 0.0;
      if (i > first) { lower = comps[i - 1u].a.y; }
      var upper = c.a.y;
      if (i == last) { upper = 0.0; }
      let diag = C + gNa + gK + gL + lower + upper;
      let rhs = C * V + gNa * E_NA + gK * E_K + gL * E_L + stimulus(c, t);
      let denom = diag + lower * prevC;
      prevC = -upper / denom;
      prevD = (rhs + lower * prevD) / denom;
      dyn[i].s = s;
      dyn[i].r = vec4f(prevC, prevD, dyn[i].r.z, dyn[i].r.w);
    }
    // Back substitution, recording the first threshold crossing of every compartment.
    var next = 0.0;
    for (var j = 0u; j < n; j++) {
      let i = last - j;
      var r = dyn[i].r;
      var V = r.y;
      if (j > 0u) { V = r.y - r.x * next; }
      next = V;
      let old = dyn[i].s.x;
      if (old < THRESHOLD && V >= THRESHOLD) {
        if (r.w == 0.0) { r.z = t + dt * ((THRESHOLD - old) / (V - old)); }
        r.w += 1.0;
      }
      dyn[i].s.x = V;
      dyn[i].r = r;
    }
    // Kymograph and voltage traces of fibre 0.
    let step = params.stepIndex + k + 1u;
    if (f == 0u && params.probeLen > 0u) {
      let slot = params.probeOffset + 2u * (step % params.probeLen);
      out[slot] = dyn[first + params.probeA].s.x;
      out[slot + 1u] = dyn[first + params.probeB].s.x;
    }
    if (f == 0u && params.histEvery > 0u && step % params.histEvery == 0u) {
      let row = (step / params.histEvery) % params.histRows;
      for (var i = first; i <= last; i++) { out[params.histOffset + row * n + (i - first)] = dyn[i].s.x; }
    }
    // Extracellular signal at the electrode: Σ Iₘ / distance, Iₘ = net axial inflow + stimulus (Kirchhoff).
    if (k < params.sigSteps) {
      var sum = 0.0;
      for (var i = first; i <= last; i++) {
        let c = comps[i];
        let V = dyn[i].s.x;
        var im = stimulus(c, t);
        if (i > first) { im += comps[i - 1u].a.y * (dyn[i - 1u].s.x - V); }
        if (i < last) { im += c.a.y * (dyn[i + 1u].s.x - V); }
        let dx = c.a.z - params.electrode;
        sum += im / sqrt(dx * dx + params.height * params.height);
      }
      out[params.sigOffset + k * params.fibres + f] = sum;
    }
  }
}
