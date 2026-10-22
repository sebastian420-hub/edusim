// Training kernels of "How AI Learns": one mini-batch step of stochastic gradient descent on a fully connected
// network, as a fixed sequence of dispatches (all sums in a fixed order, no atomics, so the GPU is deterministic):
//   forward (each hidden layer) → head (output layer, loss gradient) → backward (each hidden layer, last first)
//   → gradients (every parameter) → update (every parameter).
// Keep in sync with forward() / head() / gradient() / update() in net.ts.
//
// params: weights W[j][i] (row j = neuron) then biases, layer after layer (layout() in net.ts).
// work:   per layer, the batch's activations (B × nOut), then per layer the batch's errors dL/dz (B × nOut),
//         already divided by B so the gradients are the batch mean.
// grads:  [0, P) the gradient of the last batch, [P, 2P) the momentum velocity v.
// data:   inputKind 0: f32 features (bitcast), sample-major; 1: bytes packed four per u32 (digits).
//         Then one label (u32) per sample from labelOff.
// batch:  the batch's sample indices.

struct Net {
  L: u32,          // layers with weights (hidden + output)
  B: u32,          // batch size
  layer: u32,      // the layer this dispatch works on (0-based)
  inputKind: u32,
  act: u32,        // 0 tanh, 1 ReLU, 2 sigmoid, 3 linear (hidden layers)
  P: u32,          // number of parameters
  labelOff: u32,
  nInputs: u32,
  lr: f32,
  momentum: f32,
  l2: f32,
  pad: f32,
  layers: array<vec4u, 8>, // (nIn, nOut, weight offset, bias offset)
  offs: array<vec4u, 8>,   // (activation offset, error offset) in work
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read> batch: array<u32>;
@group(0) @binding(3) var<storage, read_write> work: array<f32>;
@group(0) @binding(4) var<storage, read_write> grads: array<f32>;
@group(0) @binding(5) var<uniform> net: Net;

fn flatId(gid: vec3u, nwg: vec3u) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

fn inputAt(s: u32, i: u32) -> f32 {
  if (net.inputKind == 0u) {
    return bitcast<f32>(data[s * net.nInputs + i]);
  }
  let word = data[s * (net.nInputs / 4u) + i / 4u];
  return f32((word >> (8u * (i % 4u))) & 255u) / 255.0;
}

fn activate(z: f32) -> f32 {
  switch net.act {
    case 0u: { return tanh(clamp(z, -15.0, 15.0)); }
    case 1u: { return max(z, 0.0); }
    case 2u: { return 1.0 / (1.0 + exp(-z)); }
    default: { return z; }
  }
}

/// d(activation)/dz from the activation's output y.
fn slope(y: f32) -> f32 {
  switch net.act {
    case 0u: { return 1.0 - y * y; }
    case 1u: { return select(0.0, 1.0, y > 0.0); }
    case 2u: { return y * (1.0 - y); }
    default: { return 1.0; }
  }
}

/// Hidden layer `net.layer`: one invocation per (sample, neuron).
@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let l = net.layer;
  let ly = net.layers[l];
  let id = flatId(gid, nwg);
  if (id >= net.B * ly.y) { return; }
  let b = id / ly.y;
  let j = id % ly.y;
  var z = params[ly.w + j];
  let row = ly.z + j * ly.x;
  if (l == 0u) {
    let s = batch[b];
    for (var i = 0u; i < ly.x; i++) { z += params[row + i] * inputAt(s, i); }
  } else {
    let prev = net.offs[l - 1u].x + b * ly.x;
    for (var i = 0u; i < ly.x; i++) { z += params[row + i] * work[prev + i]; }
  }
  work[net.offs[l].x + b * ly.y + j] = activate(z);
}

/// Output layer: one invocation per sample. Logits → probabilities (sigmoid or softmax) → dL/dz = (p − y) / B.
@compute @workgroup_size(64)
fn head(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let b = flatId(gid, nwg);
  if (b >= net.B) { return; }
  let l = net.L - 1u;
  let ly = net.layers[l];
  let s = batch[b];
  let label = data[net.labelOff + s];
  let out = net.offs[l].x + b * ly.y;
  let err = net.offs[l].y + b * ly.y;
  var zmax = -3.4e38;
  for (var j = 0u; j < ly.y; j++) {
    var z = params[ly.w + j];
    let row = ly.z + j * ly.x;
    if (l == 0u) {
      for (var i = 0u; i < ly.x; i++) { z += params[row + i] * inputAt(s, i); }
    } else {
      let prev = net.offs[l - 1u].x + b * ly.x;
      for (var i = 0u; i < ly.x; i++) { z += params[row + i] * work[prev + i]; }
    }
    work[out + j] = z;
    zmax = max(zmax, z);
  }
  let scale = 1.0 / f32(net.B);
  if (ly.y == 1u) {
    let p = 1.0 / (1.0 + exp(-work[out]));
    work[out] = p;
    work[err] = (p - f32(label)) * scale;
    return;
  }
  var sum = 0.0;
  for (var j = 0u; j < ly.y; j++) {
    let e = exp(work[out + j] - zmax);
    work[out + j] = e;
    sum += e;
  }
  for (var j = 0u; j < ly.y; j++) {
    let p = work[out + j] / sum;
    work[out + j] = p;
    work[err + j] = (p - select(0.0, 1.0, j == label)) * scale;
  }
}

/// Hidden layer `net.layer`: its error from the layer above, one invocation per (sample, neuron).
@compute @workgroup_size(64)
fn backward(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let l = net.layer;
  let ly = net.layers[l];
  let up = net.layers[l + 1u];
  let id = flatId(gid, nwg);
  if (id >= net.B * ly.y) { return; }
  let b = id / ly.y;
  let j = id % ly.y;
  let above = net.offs[l + 1u].y + b * up.y;
  var sum = 0.0;
  for (var k = 0u; k < up.y; k++) { sum += params[up.z + k * up.x + j] * work[above + k]; }
  work[net.offs[l].y + b * ly.y + j] = sum * slope(work[net.offs[l].x + b * ly.y + j]);
}

fn layerOf(p: u32) -> u32 {
  var l = 0u;
  for (; l + 1u < net.L; l++) {
    if (p < net.layers[l].w + net.layers[l].y) { break; }
  }
  return l;
}

/// Every parameter: its gradient summed over the batch, in sample order. One invocation per parameter.
@compute @workgroup_size(64)
fn gradients(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let p = flatId(gid, nwg);
  if (p >= net.P) { return; }
  let l = layerOf(p);
  let ly = net.layers[l];
  var sum = 0.0;
  if (p >= ly.w) {
    let j = p - ly.w;
    for (var b = 0u; b < net.B; b++) { sum += work[net.offs[l].y + b * ly.y + j]; }
  } else {
    let r = p - ly.z;
    let j = r / ly.x;
    let i = r % ly.x;
    for (var b = 0u; b < net.B; b++) {
      var x: f32;
      if (l == 0u) { x = inputAt(batch[b], i); } else { x = work[net.offs[l - 1u].x + b * ly.x + i]; }
      sum += work[net.offs[l].y + b * ly.y + j] * x;
    }
  }
  grads[p] = sum;
}

/// SGD with momentum and L2 decay on the weights: v ← μv + g + λw, w ← w − ηv.
@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let p = flatId(gid, nwg);
  if (p >= net.P) { return; }
  let ly = net.layers[layerOf(p)];
  var g = grads[p];
  if (p < ly.w) { g += net.l2 * params[p]; }
  let v = net.momentum * grads[net.P + p] + g;
  grads[net.P + p] = v;
  params[p] -= net.lr * v;
}
