// The Digits view: a strip of "templates" (the weights each output class — or the first hidden neurons — give to
// every pixel: blue = this ink votes for it, orange = against) above a grid of test digits, each with the
// network's guess in a corner (green right, red wrong; on a mistake the true label is shown below it).

const GLYPHS = array<u32, 10>(31599u, 29850u, 29671u, 31207u, 18925u, 31183u, 31695u, 18727u, 31727u, 31215u);
const BG = vec3f(0.035, 0.043, 0.06);
const CLASS0 = vec3f(0.902, 0.624, 0.0);
const CLASS1 = vec3f(0.337, 0.706, 0.914);
const RIGHT = vec3f(0.0, 0.62, 0.45);
const WRONG = vec3f(0.835, 0.369, 0.0);

struct Digits {
  resolution: vec2f,
  tile: f32,
  templateSize: f32,
  grid: vec4f,       // x, y, w, h (device pixels, y down)
  strip: vec4f,      // templates: x, y, w, h
  cols: u32,
  tiles: u32,        // tiles filled
  templates: u32,    // templates drawn
  rowStride: u32,    // parameters between template rows (784: one weight row per template)
  firstWeight: u32,  // offset of template 0's first weight
  wordsPerSample: u32,
  hasParams: u32,
  pad0: u32,
  scale: f32,        // weights are shown relative to this
  pad1: f32,
  pad2: f32,
  pad3: f32,
}

@group(0) @binding(0) var<uniform> dg: Digits;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read> tiles: array<vec4f>;  // sample, guess, correct, true label
@group(0) @binding(3) var<storage, read> params: array<f32>;

fn ink(sample: u32, x: u32, y: u32) -> f32 {
  let i = y * 28u + x;
  let word = data[sample * dg.wordsPerSample + i / 4u];
  return f32((word >> (8u * (i % 4u))) & 255u) / 255.0;
}

/// A 3 × 5 glyph of `digit` in the box [0, 1]², true where it has ink.
fn glyph(digit: u32, p: vec2f) -> bool {
  if (any(p < vec2f(0.0)) || any(p >= vec2f(1.0))) { return false; }
  let c = u32(p.x * 3.0);
  let r = u32(p.y * 5.0);
  return ((GLYPHS[min(digit, 9u)] >> (r * 3u + c)) & 1u) == 1u;
}

/// A badge in the box b (x, y, side) showing `digit` on `color`; returns the colour, or a negative alpha outside.
fn badge(px: vec2f, b: vec3f, digit: u32, color: vec3f) -> vec4f {
  let q = (px - b.xy) / b.z;
  if (any(q < vec2f(0.0)) || any(q >= vec2f(1.0))) { return vec4f(0.0, 0.0, 0.0, -1.0); }
  // Glyph in the middle 3:5 box, with a margin.
  let g = (q - vec2f(0.25, 0.12)) / vec2f(0.5, 0.76);
  return vec4f(select(color, vec3f(1.0), glyph(digit, g)), 1.0);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let px = uv * dg.resolution;

  // Templates.
  let s = (px - dg.strip.xy);
  if (dg.hasParams == 1u && s.y >= 0.0 && s.y < dg.templateSize && s.x >= 0.0) {
    let k = u32(s.x / (dg.templateSize * 1.08));
    let inside = (s.x - f32(k) * dg.templateSize * 1.08) / dg.templateSize;
    if (k < dg.templates && inside < 1.0) {
      let x = u32(inside * 28.0);
      let y = u32(s.y / dg.templateSize * 28.0);
      let w = params[dg.firstWeight + k * dg.rowStride + y * 28u + x] / dg.scale;
      let t = clamp(w, -1.0, 1.0);
      return vec4f(mix(vec3f(0.1, 0.11, 0.14), select(CLASS0, CLASS1, t > 0.0), sqrt(abs(t))), 1.0);
    }
  }

  // Tiles.
  let g = px - dg.grid.xy;
  if (all(g >= vec2f(0.0)) && g.x < f32(dg.cols) * dg.tile && g.y < dg.grid.w) {
    let cell = vec2u(g / dg.tile);
    let index = cell.y * dg.cols + cell.x;
    if (index < dg.tiles) {
      let t = tiles[index];
      let local = (g - vec2f(cell) * dg.tile) / dg.tile; // 0 … 1 within the tile
      let margin = 0.06;
      let d = (local - margin) / (1.0 - 2.0 * margin);
      if (all(d >= vec2f(0.0)) && all(d < vec2f(1.0))) {
        let correct = t.z > 0.5;
        let side = dg.tile * 0.3;
        let corner = dg.grid.xy + vec2f(cell) * dg.tile + vec2f(dg.tile * (1.0 - margin) - side, dg.tile * margin);
        let top = badge(px, vec3f(corner, side), u32(t.y), select(WRONG, RIGHT, correct));
        if (top.a >= 0.0 && dg.hasParams == 1u) { return vec4f(top.rgb, 1.0); }
        if (!correct && dg.hasParams == 1u) {
          let below = badge(px, vec3f(corner + vec2f(0.0, side * 1.1), side), u32(t.w), vec3f(0.35, 0.38, 0.44));
          if (below.a >= 0.0) { return vec4f(below.rgb, 1.0); }
        }
        let v = ink(u32(t.x), u32(d.x * 28.0), u32(d.y * 28.0));
        let paper = select(vec3f(0.09, 0.1, 0.13), vec3f(0.2, 0.08, 0.05), !correct && dg.hasParams == 1u);
        return vec4f(mix(paper, vec3f(0.95), v), 1.0);
      }
    }
  }
  return vec4f(BG, 1.0);
}
