// Appends the lower bob's position of pendulum 0 to the trail ring buffer (slot `head`).

struct TrailParams {
  head: u32,
  pad: u32,
  l1: f32,
  l2: f32,
}

@group(0) @binding(0) var<storage, read> state: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> trail: array<vec4f>;
@group(0) @binding(2) var<uniform> params: TrailParams;

@compute @workgroup_size(1)
fn main() {
  let s = state[0];
  let b1 = vec2f(params.l1 * sin(s.x), -params.l1 * cos(s.x));
  let b2 = b1 + vec2f(params.l2 * sin(s.y), -params.l2 * cos(s.y));
  trail[params.head] = vec4f(b2, 0.0, 1.0);
}
