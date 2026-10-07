// Draws the lower bob's trail as a line strip through its ring buffer, fading with age (oldest → newest drawn in
// two runs: [head, len) then [0, head)).

struct TrailView {
  center: vec2f,
  resolution: vec2f,
  pxPerUnit: f32,
  pad0: f32,
  len: u32,
  head: u32,
  filled: u32,
  pad1: u32,
  color: vec4f,
}

@group(0) @binding(0) var<uniform> view: TrailView;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) tint: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @location(0) point: vec4f) -> VOut {
  let len = view.len;
  let age = (view.head + len - 1u - vi) % len;
  var out: VOut;
  out.clip = vec4f(2.0 * (point.xy - view.center) * view.pxPerUnit / view.resolution, 0.0, 1.0);
  let fade = select(0.0, 1.0 - f32(age) / f32(max(view.filled, 1u)), age < view.filled && point.w > 0.0);
  out.tint = vec4f(view.color.rgb, fade * view.color.a);
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  return in.tint;
}
