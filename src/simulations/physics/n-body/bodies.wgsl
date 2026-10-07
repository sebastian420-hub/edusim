// Draws every body as a glowing disc: one instanced quad per body. Positions arrive as an instance vertex
// buffer copied from the simulation's storage buffer on the GPU (vertex shaders may not read storage buffers
// on every device). World y points up; `view` maps world units to pixels.

struct View {
  center: vec2f,
  resolution: vec2f,
  pxPerUnit: f32,
  glow: f32,
  pad0: f32,
  pad1: f32,
}

@group(0) @binding(0) var<uniform> view: View;

struct VIn {
  @builtin(vertex_index) vi: u32,
  /** (x, y, z, mass) */
  @location(0) body: vec4f,
  /** (r, g, b, radius in pixels) */
  @location(1) look: vec4f,
}

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) local: vec2f,
  @location(1) tint: vec3f,
}

const EXTENT: f32 = 3.0; // quad half-size in body radii: room for the glow

@vertex
fn vs_main(in: VIn) -> VOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0), vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let corner = corners[in.vi];
  let screen = (in.body.xy - view.center) * view.pxPerUnit + corner * EXTENT * in.look.w;
  var out: VOut;
  out.clip = vec4f(2.0 * screen / view.resolution, 0.0, 1.0);
  out.local = corner * EXTENT;
  out.tint = in.look.rgb;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  let d = length(in.local);
  let core = 1.0 - smoothstep(0.75, 1.0, d);
  let halo = view.glow * exp(-1.6 * d * d);
  let intensity = max(core, halo);
  return vec4f(in.tint * intensity, intensity);
}
