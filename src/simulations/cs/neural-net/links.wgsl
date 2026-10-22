// The network diagram's weights: one thin quad per weight from input node i (column l) to node j (column l + 1),
// blue for positive, orange for negative, thicker and brighter for larger |w|. The weights arrive as an instance
// vertex attribute (copied from the parameters each frame, layer by layer, without the biases); the endpoints are
// computed here from the instance index and the column sizes. Node positions follow nodeCenter() in scene.ts.

struct Links {
  resolution: vec2f,
  spacing: f32,
  node: f32,
  diagram: vec4f, // x, y, w, h
  columns: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  sizes: array<vec4u, 2>,
}

@group(0) @binding(0) var<uniform> links: Links;

struct VIn {
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
  @location(0) weight: f32,
}

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) tint: vec4f,
  @location(1) across: f32,
  @location(2) half: f32,
}

fn sizeOf(c: u32) -> u32 {
  return links.sizes[c / 4u][c % 4u];
}

fn center(c: u32, k: u32) -> vec2f {
  let n = f32(sizeOf(c));
  return vec2f(links.diagram.x + (f32(c) + 0.5) * links.diagram.z / f32(links.columns), links.diagram.y + links.diagram.w * 0.5 + (f32(k) - (n - 1.0) * 0.5) * links.spacing);
}

@vertex
fn vs_main(in: VIn) -> VOut {
  // Which weight: layer l, neuron j, input i (row-major per layer, as in the parameters).
  var rest = in.ii;
  var l = 0u;
  for (; l + 1u < links.columns; l++) {
    let count = sizeOf(l) * sizeOf(l + 1u);
    if (rest < count) { break; }
    rest -= count;
  }
  let nIn = sizeOf(l);
  let j = rest / nIn;
  let i = rest % nIn;
  let a = center(l, i) + vec2f(links.node * 0.5, 0.0);
  let b = center(l + 1u, j) - vec2f(links.node * 0.5, 0.0);
  let along = normalize(b - a + vec2f(1e-6, 0.0));
  let normal = vec2f(-along.y, along.x);
  let strength = min(abs(in.weight), 3.0) / 3.0;
  let half = 0.5 + 2.2 * strength;
  var corners = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(0.0, 1.0), vec2f(0.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
  let corner = corners[in.vi];
  let px = mix(a, b, corner.x) + normal * corner.y * (half + 1.0);
  var out: VOut;
  out.clip = vec4f(px.x / links.resolution.x * 2.0 - 1.0, 1.0 - px.y / links.resolution.y * 2.0, 0.0, 1.0);
  let color = select(vec3f(0.902, 0.624, 0.0), vec3f(0.337, 0.706, 0.914), in.weight > 0.0);
  out.tint = vec4f(color, 0.15 + 0.75 * strength);
  out.across = corner.y * (half + 1.0);
  out.half = half;
  return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
  // Soft edges: full inside the half-width, fading out over one pixel.
  let fade = clamp(in.half + 0.5 - abs(in.across), 0.0, 1.0);
  return vec4f(in.tint.rgb * in.tint.a * fade, in.tint.a * fade);
}
