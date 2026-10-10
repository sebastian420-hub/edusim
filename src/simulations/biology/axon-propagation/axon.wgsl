// The Axon view: on top, the axon as a tube coloured by its voltage now (myelin drawn as a pale sheath around
// the internodes); below, the kymograph: voltage against position (across) and time since the sweep began (down).
// A travelling spike is a bright line whose slope is the conduction speed.

struct View {
  resolution: vec2f,
  length: f32, // cm
  sweepMs: f32,
  n: u32, // compartments of the fibre
  histEvery: u32,
  histRows: u32,
  rowsDone: u32, // kymograph rows recorded so far this sweep
  dt: f32,
  myelinated: u32,
  tubeTop: f32, // in uv units: tube band, then the kymograph
  tubeBottom: f32,
  kymoTop: f32,
  kymoBottom: f32,
  margin: f32, // horizontal margin, uv
  pad: f32,
}

struct Comp {
  a: vec4f,
  b: vec4f,
}

struct Dyn {
  s: vec4f,
  r: vec4f,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> comps: array<Comp>;
@group(0) @binding(2) var<storage, read> dyn: array<Dyn>;
@group(0) @binding(3) var<storage, read> out: array<f32>;

/** Voltage colour: deep blue at rest, through magenta and orange, to pale yellow at the spike's peak. */
fn voltageColor(v: f32) -> vec3f {
  let t = clamp((v + 80.0) / 120.0, 0.0, 1.0);
  let c0 = vec3f(0.03, 0.05, 0.16);
  let c1 = vec3f(0.12, 0.22, 0.55);
  let c2 = vec3f(0.75, 0.2, 0.55);
  let c3 = vec3f(0.98, 0.55, 0.2);
  let c4 = vec3f(1.0, 0.95, 0.7);
  if (t < 0.15) { return mix(c0, c1, t / 0.15); }
  if (t < 0.45) { return mix(c1, c2, (t - 0.15) / 0.3); }
  if (t < 0.75) { return mix(c2, c3, (t - 0.45) / 0.3); }
  return mix(c3, c4, (t - 0.75) / 0.25);
}

/** The compartment whose span contains position x (cm): binary search over the compartment centres. */
fn compartmentAt(x: f32) -> u32 {
  var lo = 0u;
  var hi = view.n - 1u;
  while (lo < hi) {
    let mid = (lo + hi + 1u) / 2u;
    // Boundary between mid − 1 and mid: halfway between their centres.
    let edge = 0.5 * (comps[mid - 1u].a.z + comps[mid].a.z);
    if (x >= edge) { lo = mid; } else { hi = mid - 1u; }
  }
  return lo;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let bg = vec3f(0.016, 0.02, 0.03);
  let u = (uv.x - view.margin) / (1.0 - 2.0 * view.margin);
  if (u < 0.0 || u > 1.0) { return vec4f(bg, 1.0); }
  let x = u * view.length;
  let i = compartmentAt(x);
  var internode = view.myelinated == 1u && comps[i].b.y == 0.0;
  if (internode) {
    // Nodes are ~1 µm, far thinner than a pixel: show one as a gap wherever it falls within ±1.5 px.
    let halfPixel = 1.5 * view.length / ((1.0 - 2.0 * view.margin) * view.resolution.x);
    let lo = compartmentAt(x - halfPixel);
    let hi = compartmentAt(x + halfPixel);
    for (var k = lo; k <= hi; k++) {
      if (comps[k].b.y > 0.0) { internode = false; }
    }
  }

  // Tube.
  if (uv.y >= view.tubeTop && uv.y <= view.tubeBottom) {
    let mid = 0.5 * (view.tubeTop + view.tubeBottom);
    let half = 0.5 * (view.tubeBottom - view.tubeTop);
    let r = abs(uv.y - mid) / half; // 0 at the axis, 1 at the edge of the band
    var core = 0.55;
    if (view.myelinated == 1u) { core = 0.38; }
    if (r <= core) {
      let shade = 0.75 + 0.25 * sqrt(max(0.0, 1.0 - (r / core) * (r / core)));
      return vec4f(voltageColor(dyn[i].s.x) * shade, 1.0);
    }
    if (internode && r <= 0.95) {
      // Myelin: a pale sheath (the treated stretch is marked by the overlay).
      return vec4f(vec3f(0.72, 0.76, 0.84) * (1.0 - 0.35 * (r - core) / (0.95 - core)), 1.0);
    }
    return vec4f(bg, 1.0);
  }

  // Kymograph.
  if (uv.y >= view.kymoTop && uv.y <= view.kymoBottom) {
    let t = (uv.y - view.kymoTop) / (view.kymoBottom - view.kymoTop) * view.sweepMs;
    let rowTime = f32(view.histEvery) * view.dt;
    let row = u32(max(0.0, t / rowTime));
    if (row >= view.rowsDone || row >= view.histRows) { return vec4f(bg * 1.6, 1.0); }
    return vec4f(voltageColor(out[row * view.n + i]), 1.0);
  }
  return vec4f(bg, 1.0);
}
