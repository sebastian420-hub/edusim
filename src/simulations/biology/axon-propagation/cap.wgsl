// Sums the per-fibre extracellular signals of one dispatch into the compound action potential ring:
// cap[(head + s) % len] = Σ_fibres signal[s][fibre]. One invocation per step.

struct CapParams {
  fibres: u32,
  steps: u32,
  sigOffset: u32,
  capOffset: u32,
  head: u32,
  len: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<uniform> params: CapParams;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let s = gid.x;
  if (s >= params.steps) { return; }
  var sum = 0.0;
  for (var f = 0u; f < params.fibres; f++) { sum += out[params.sigOffset + s * params.fibres + f]; }
  out[params.capOffset + (params.head + s) % params.len] = sum;
}
