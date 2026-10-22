// The data points on the plot: one disc per point (an instance vertex attribute: x, y ∈ [−1, 1] with y up, the
// label, and 1 for a test point). Training points are filled with their class colour and a light outline; test
// points are drawn as rings, so you can see what the network was never shown.

struct Points {
  resolution: vec2f,
  radius: f32,
  showTest: f32,
  plot: vec4f, // x, y, w, h (device pixels, y down)
}

@group(0) @binding(0) var<uniform> points: Points;

struct VIn {
  @builtin(vertex_index) vi: u32,
  @location(0) point: vec4f,
}

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) local: vec2f,
  @location(1) @interpolate(flat) kind: vec2f,
}

@vertex
fn vs_main(in: VIn) -> VOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let c = corners[in.vi];
  let center = points.plot.xy + vec2f((in.point.x + 1.0) * 0.5, (1.0 - in.point.y) * 0.5) * points.plot.zw;
  let hidden = in.point.w > 0.5 && points.showTest < 0.5;
  let px = center + c * (points.radius + 1.0) * select(1.0, 0.0, hidden);
  var out: VOut;
  out.clip = vec4f(px.x / points.resolution.x * 2.0 - 1.0, 1.0 - px.y / points.resolution.y * 2.0, 0.0, 1.0);
  out.local = c * (points.radius + 1.0);
  out.kind = in.point.zw;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let r = length(in.local);
  let R = points.radius;
  let color = select(vec3f(0.902, 0.624, 0.0), vec3f(0.337, 0.706, 0.914), in.kind.x > 0.5);
  let disc = clamp(R + 0.5 - r, 0.0, 1.0);
  if (in.kind.y > 0.5) {
    // Test point: a ring.
    let ring = disc * clamp(r - (R - 1.8) + 0.5, 0.0, 1.0);
    return vec4f(color * ring, ring);
  }
  let inner = clamp(R - 0.7 - r, 0.0, 1.0);
  let rgb = mix(vec3f(0.95), color, inner);
  return vec4f(rgb * disc, disc);
}
