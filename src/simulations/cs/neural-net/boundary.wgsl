// The decision map of "How AI Learns": the network evaluated on a grid over the plane, x and y from −1 to 1
// (row 0 at the top, y = +1). Slot 0 is the output (the probability of class 1); slot k ≥ 1 is hidden neuron k − 1
// (counted through the layers), drawn small in the network diagram. One invocation per (slot, cell).
// Keep in sync with probabilityAt() / features() in net.ts.

const MAX_WIDTH: u32 = 32u;
const PI: f32 = 3.14159265358979;

struct Map {
  L: u32,
  act: u32,
  mask: u32,      // input features, one bit each (FEATURES in net.ts)
  size: u32,      // output map: size × size cells
  thumb: u32,     // neuron maps: thumb × thumb cells each
  neurons: u32,   // hidden neurons drawn (slots 1 … neurons)
  pad0: u32,
  pad1: u32,
  layers: array<vec4u, 8>,
}

@group(0) @binding(0) var<storage, read> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> map: array<f32>;
@group(0) @binding(2) var<uniform> m: Map;

fn feature(f: u32, x: f32, y: f32) -> f32 {
  switch f {
    case 0u: { return x; }
    case 1u: { return y; }
    case 2u: { return x * x; }
    case 3u: { return y * y; }
    case 4u: { return x * y; }
    case 5u: { return sin(PI * x); }
    default: { return sin(PI * y); }
  }
}

fn activate(z: f32) -> f32 {
  switch m.act {
    case 0u: { return tanh(clamp(z, -15.0, 15.0)); }
    case 1u: { return max(z, 0.0); }
    case 2u: { return 1.0 / (1.0 + exp(-z)); }
    default: { return z; }
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let id = gid.x + gid.y * nwg.x * 64u;
  let big = m.size * m.size;
  let small = m.thumb * m.thumb;
  var slot = 0u;
  var cell = id;
  var side = m.size;
  if (id >= big) {
    slot = 1u + (id - big) / small;
    cell = (id - big) % small;
    side = m.thumb;
    if (slot > m.neurons) { return; }
  }
  let x = (f32(cell % side) + 0.5) / f32(side) * 2.0 - 1.0;
  let y = 1.0 - (f32(cell / side) + 0.5) / f32(side) * 2.0;

  var a: array<f32, MAX_WIDTH>;
  var b: array<f32, MAX_WIDTH>;
  var n = 0u;
  for (var f = 0u; f < 7u; f++) {
    if (((m.mask >> f) & 1u) == 1u) { a[n] = feature(f, x, y); n++; }
  }
  // Which neuron this slot shows: hidden neurons are numbered through the layers.
  var pick = slot - 1u;
  for (var l = 0u; l < m.L; l++) {
    let ly = m.layers[l];
    let last = l + 1u == m.L;
    for (var j = 0u; j < ly.y; j++) {
      var z = params[ly.w + j];
      let row = ly.z + j * ly.x;
      for (var i = 0u; i < ly.x; i++) { z += params[row + i] * a[i]; }
      b[j] = select(activate(z), z, last);
    }
    if (slot > 0u && !last) {
      if (pick < ly.y) {
        map[big + (slot - 1u) * small + cell] = b[pick];
        return;
      }
      pick -= ly.y;
    }
    for (var j = 0u; j < ly.y; j++) { a[j] = b[j]; }
  }
  if (slot == 0u) { map[cell] = 1.0 / (1.0 + exp(-a[0])); }
}
