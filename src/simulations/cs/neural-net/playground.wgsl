// Playground background: the decision map in the plot (the probability of class 1, from boundary.wgsl's map,
// interpolated), and the network diagram's nodes (inputs: their feature over the plane; hidden: each neuron's
// small map; output: the decision map again). Node positions follow nodeCenter() in scene.ts.
// Class 0 is orange, class 1 blue (Okabe–Ito).

const PI: f32 = 3.14159265358979;

struct Scene {
  resolution: vec2f,
  spacing: f32,
  node: f32,
  plot: vec4f,     // x, y, w, h (device pixels, y down)
  diagram: vec4f,  // x, y, w, h: .z is the width, .w the height
  columns: u32,
  mapSize: u32,
  thumb: u32,
  act: u32,        // 0 tanh, 1 ReLU, 2 sigmoid, 3 linear
  sizes: array<vec4u, 2>,    // nodes per column (inputs, hidden…, output)
  features: array<vec4u, 2>, // feature id of each input node
  hasMap: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> map: array<f32>;

const BG = vec3f(0.035, 0.043, 0.06);
const CLASS0 = vec3f(0.902, 0.624, 0.0);
const CLASS1 = vec3f(0.337, 0.706, 0.914);

fn sizeOf(c: u32) -> u32 {
  return scene.sizes[c / 4u][c % 4u];
}

/// Bilinear sample of an n × n block of the map starting at `base`, uv ∈ [0, 1]² (top-left origin).
fn sampleMap(base: u32, n: u32, uv: vec2f) -> f32 {
  let p = clamp(uv * f32(n) - 0.5, vec2f(0.0), vec2f(f32(n) - 1.0));
  let i = vec2u(floor(p));
  let j = min(i + 1u, vec2u(n - 1u));
  let f = p - floor(p);
  let a = mix(map[base + i.y * n + i.x], map[base + i.y * n + j.x], f.x);
  let b = mix(map[base + j.y * n + i.x], map[base + j.y * n + j.x], f.x);
  return mix(a, b, f.y);
}

/// A probability as a colour: orange (0) through the background (½) to blue (1).
fn probabilityColor(p: f32) -> vec3f {
  let c = select(CLASS0, CLASS1, p > 0.5);
  return mix(BG * 1.6, c, 0.15 + 0.6 * abs(2.0 * p - 1.0));
}

/// A neuron's or feature's value as a colour: orange below its midpoint, blue above.
fn valueColor(v: f32) -> vec3f {
  let t = tanh(2.0 * v);
  return mix(vec3f(0.12, 0.14, 0.18), select(CLASS0, CLASS1, t > 0.0), abs(t));
}

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

fn midpoint() -> f32 {
  return select(0.0, 0.5, scene.act == 2u);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let px = uv * scene.resolution;
  // The plot.
  let pu = (px - scene.plot.xy) / scene.plot.zw;
  if (all(pu >= vec2f(0.0)) && all(pu <= vec2f(1.0))) {
    if (scene.hasMap == 0u) { return vec4f(BG * 1.6, 1.0); }
    var col = probabilityColor(sampleMap(0u, scene.mapSize, pu));
    // Faint axes through the origin.
    let g = abs(pu - 0.5) * scene.plot.zw;
    if (min(g.x, g.y) < 0.6) { col = mix(col, vec3f(1.0), 0.12); }
    return vec4f(col, 1.0);
  }
  // The diagram's nodes.
  let colW = scene.diagram.z / f32(scene.columns);
  let c = u32(clamp(floor((px.x - scene.diagram.x) / colW), 0.0, f32(scene.columns - 1u)));
  let n = sizeOf(c);
  let cx = scene.diagram.x + (f32(c) + 0.5) * colW;
  let k = round((px.y - scene.diagram.y - scene.diagram.w * 0.5) / scene.spacing + (f32(n) - 1.0) * 0.5);
  if (k >= 0.0 && k < f32(n)) {
    let cy = scene.diagram.y + scene.diagram.w * 0.5 + (k - (f32(n) - 1.0) * 0.5) * scene.spacing;
    let d = (px - vec2f(cx, cy)) / scene.node + 0.5;
    if (all(d >= vec2f(0.0)) && all(d <= vec2f(1.0))) {
      let edge = min(min(d.x, 1.0 - d.x), min(d.y, 1.0 - d.y)) * scene.node;
      if (edge < 1.0) { return vec4f(0.55, 0.6, 0.68, 1.0); }
      let x = d.x * 2.0 - 1.0;
      let y = 1.0 - d.y * 2.0;
      if (c == 0u) {
        let ki = u32(k);
        return vec4f(valueColor(feature(scene.features[ki / 4u][ki % 4u], x, y)), 1.0);
      }
      if (scene.hasMap == 0u) { return vec4f(0.12, 0.14, 0.18, 1.0); }
      if (c == scene.columns - 1u) {
        return vec4f(probabilityColor(sampleMap(0u, scene.mapSize, d)), 1.0);
      }
      // Hidden neuron: its index counted through the hidden layers.
      var index = u32(k);
      for (var h = 1u; h < c; h++) { index += sizeOf(h); }
      let base = scene.mapSize * scene.mapSize + index * scene.thumb * scene.thumb;
      return vec4f(valueColor(sampleMap(base, scene.thumb, d) - midpoint()), 1.0);
    }
  }
  return vec4f(BG, 1.0);
}
