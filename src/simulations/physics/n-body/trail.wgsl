// Appends the current position of the first `count` bodies to their trail ring buffers (slot `head`).

struct TrailParams {
  count: u32,
  len: u32,
  head: u32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> pos: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> trail: array<vec4f>;
@group(0) @binding(2) var<uniform> params: TrailParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < params.count) {
    trail[i * params.len + params.head] = vec4f(pos[i].xyz, 1.0);
  }
}
