// Draws double pendulums: two rods and two bobs per pendulum (and the pivot), one instance per pendulum. The state arrives as an
// instance vertex attribute (copied from the simulation's storage buffer: vertex shaders may not read storage
// buffers on every device), and the bob positions are computed here from the angles. World y points up, the pivot
// is at the origin.

struct ArmView {
  center: vec2f,
  resolution: vec2f,
  pxPerUnit: f32,
  l1: f32,
  l2: f32,
  rodPx: f32,
  bobPx: f32,
  alpha: f32,
  /** 0 = one colour (`color`), 1 = a rainbow by pendulum index (to see a crowd fan out) */
  colorMode: u32,
  count: u32,
  color: vec4f,
}

@group(0) @binding(0) var<uniform> view: ArmView;

struct VIn {
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
  @location(0) state: vec4f,
}

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) local: vec2f,
  @location(1) tint: vec4f,
  @location(2) disc: f32,
}

fn toClip(px: vec2f) -> vec4f {
  return vec4f(2.0 * px / view.resolution, 0.0, 1.0);
}

fn hue(t: f32) -> vec3f {
  return 0.5 + 0.5 * cos(6.28318 * (t + vec3f(0.0, 0.33, 0.67)));
}

@vertex
fn vs_main(in: VIn) -> VOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let part = in.vi / 6u; // 0 rod 1, 1 rod 2, 2 bob 1, 3 bob 2, 4 the pivot (single pendulums only)
  let c = corners[in.vi % 6u];
  let b1 = vec2f(view.l1 * sin(in.state.x), -view.l1 * cos(in.state.x));
  let b2 = b1 + vec2f(view.l2 * sin(in.state.y), -view.l2 * cos(in.state.y));
  let s0 = (vec2f(0.0) - view.center) * view.pxPerUnit;
  let s1 = (b1 - view.center) * view.pxPerUnit;
  let s2 = (b2 - view.center) * view.pxPerUnit;
  var out: VOut;
  var tint = view.color;
  if (view.colorMode == 1u) {
    tint = vec4f(hue(f32(in.ii) / f32(max(view.count, 1u))), 1.0);
  }
  out.tint = vec4f(tint.rgb, tint.a * view.alpha);
  if (part < 2u) {
    let a = select(s1, s0, part == 0u);
    let b = select(s2, s1, part == 0u);
    var dir = b - a;
    let len = length(dir);
    dir = select(vec2f(1.0, 0.0), dir / len, len > 1e-6);
    let normal = vec2f(-dir.y, dir.x);
    let along = select(a, b, c.x > 0.0);
    out.clip = toClip(along + normal * c.y * 0.5 * view.rodPx);
    out.local = vec2f(0.0);
    out.disc = 0.0;
  } else if (part == 4u) {
    out.tint = vec4f(0.45, 0.5, 0.6, view.alpha);
    out.clip = toClip(s0 + c * view.bobPx * 0.45);
    out.local = c;
    out.disc = 1.0;
  } else {
    let centre = select(s2, s1, part == 2u);
    let r = view.bobPx * select(1.0, 0.8, part == 2u);
    out.clip = toClip(centre + c * r);
    out.local = c;
    out.disc = 1.0;
  }
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  var a = in.tint.a;
  if (in.disc > 0.5) {
    a *= 1.0 - smoothstep(0.8, 1.0, length(in.local));
  }
  return vec4f(in.tint.rgb * a, a);
}
