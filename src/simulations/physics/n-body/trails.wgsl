// Draws one body's orbit trail as a line strip through its ring buffer of recorded positions, fading with
// age. The ring is drawn in two runs ([head, len) then [0, head)); the instance index is the body, which
// selects its colour.

struct TrailView {
  center: vec2f,
  resolution: vec2f,
  pxPerUnit: f32,
  pad0: f32,
  len: u32,
  head: u32,
  filled: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<uniform> view: TrailView;

struct VIn {
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) body: u32,
  /** (x, y, z, valid) */
  @location(0) point: vec4f,
  /** (r, g, b, radius) of the body */
  @location(1) look: vec4f,
}

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) tint: vec4f,
}

@vertex
fn vs_main(in: VIn) -> VOut {
  let len = view.len;
  let k = in.vi - in.body * len;
  // age 0 = the newest point (head − 1).
  let age = (view.head + len - 1u - k) % len;
  var out: VOut;
  let screen = (in.point.xy - view.center) * view.pxPerUnit;
  out.clip = vec4f(2.0 * screen / view.resolution, 0.0, 1.0);
  let fade = select(0.0, 1.0 - f32(age) / f32(max(view.filled, 1u)), age < view.filled && in.point.w > 0.0);
  // Alpha (not additive) blending: an orbit traced many times stays its colour instead of saturating to white.
  out.tint = vec4f(in.look.rgb, fade * 0.75);
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  return in.tint;
}
