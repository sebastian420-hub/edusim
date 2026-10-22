// Evaluates the network on many samples at once (the loss curves, accuracy, the digits' guesses and the step
// probe): one invocation per (probe point, sample), the whole forward pass in private memory.
// Weights are taken at w + (s − 1)·(−η·v): s = 1 is the network now, s = 0 before the last step (stepProfile()).
// Keep in sync with forward() / head() in net.ts.
//
// out[k] = (loss, 1 if correct, the guessed class, its probability).

const MAX_WIDTH: u32 = 128u;

struct Eval {
  L: u32,
  first: u32,     // first sample
  count: u32,     // samples per probe point
  S: u32,         // probe points
  inputKind: u32,
  nInputs: u32,
  labelOff: u32,
  act: u32,
  lr: f32,
  P: u32,
  pad0: u32,
  pad1: u32,
  layers: array<vec4u, 8>,
  s: array<vec4f, 5>,
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read> grads: array<f32>;
@group(0) @binding(2) var<storage, read> data: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4f>;
@group(0) @binding(4) var<uniform> ev: Eval;

fn inputAt(s: u32, i: u32) -> f32 {
  if (ev.inputKind == 0u) {
    return bitcast<f32>(data[s * ev.nInputs + i]);
  }
  let word = data[s * (ev.nInputs / 4u) + i / 4u];
  return f32((word >> (8u * (i % 4u))) & 255u) / 255.0;
}

fn activate(z: f32) -> f32 {
  switch ev.act {
    case 0u: { return tanh(clamp(z, -15.0, 15.0)); }
    case 1u: { return max(z, 0.0); }
    case 2u: { return 1.0 / (1.0 + exp(-z)); }
    default: { return z; }
  }
}

fn softplus(z: f32) -> f32 {
  return max(z, 0.0) + log(1.0 + exp(-abs(z)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let id = gid.x + gid.y * nwg.x * 64u;
  if (id >= ev.count * ev.S) { return; }
  let probe = id / ev.count;
  let sample = ev.first + id % ev.count;
  let c = -(ev.s[probe / 4u][probe % 4u] - 1.0) * ev.lr; // weight = w + c·v
  var a: array<f32, MAX_WIDTH>;
  var b: array<f32, MAX_WIDTH>;
  for (var l = 0u; l < ev.L; l++) {
    let ly = ev.layers[l];
    let last = l + 1u == ev.L;
    for (var j = 0u; j < ly.y; j++) {
      let bj = ly.w + j;
      var z = params[bj] + c * grads[ev.P + bj];
      let row = ly.z + j * ly.x;
      for (var i = 0u; i < ly.x; i++) {
        var x: f32;
        if (l == 0u) { x = inputAt(sample, i); } else { x = a[i]; }
        z += (params[row + i] + c * grads[ev.P + row + i]) * x;
      }
      b[j] = select(activate(z), z, last);
    }
    for (var j = 0u; j < ly.y; j++) { a[j] = b[j]; }
  }
  let n = ev.layers[ev.L - 1u].y;
  let label = data[ev.labelOff + sample];
  if (n == 1u) {
    let z = a[0];
    let p = 1.0 / (1.0 + exp(-z));
    let guess = select(0u, 1u, z > 0.0);
    out[id] = vec4f(softplus(z) - f32(label) * z, select(0.0, 1.0, guess == label), f32(guess), select(1.0 - p, p, guess == 1u));
    return;
  }
  var zmax = a[0];
  var best = 0u;
  for (var k = 1u; k < n; k++) {
    if (a[k] > zmax) { zmax = a[k]; best = k; }
  }
  var sum = 0.0;
  for (var k = 0u; k < n; k++) { sum += exp(a[k] - zmax); }
  out[id] = vec4f(log(sum) + zmax - a[label], select(0.0, 1.0, best == label), f32(best), 1.0 / sum);
}
