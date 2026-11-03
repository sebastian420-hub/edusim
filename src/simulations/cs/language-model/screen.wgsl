// The canvas of "How AI Writes": panels of text cells (glyphs from an atlas rendered at exactly the cell size, so
// they are drawn 1:1), each cell with a colour and a tinted background (surprise, attention, guesses), and heat maps
// (attention matrices). Laid out on the CPU (screen.ts); this pass just draws what the buffers say.
//
// Cell word: bits 0–7 glyph (255 = none), 8–11 text colour, 12–15 tint kind, 16–31 tint strength (0 … 65535).

struct Panel {
  rect: vec4f,  // x, y, w, h in device pixels (y down)
  grid: vec4u,  // columns, rows, kind (0 text, 1 heat map, 2 box: columns = 1 highlights it), first cell / heat value
  cell: vec4f,  // cell width, height (px), gap between heat cells (px), unused
}

struct Screen {
  resolution: vec2f,
  count: u32,
  glyphs: u32,
  glyph: vec2u, // atlas glyph size in pixels (= text cell size)
  pad0: u32,
  pad1: u32,
  panels: array<Panel, 24>,
}

@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var<storage, read> cells: array<u32>;
@group(0) @binding(2) var<storage, read> atlas: array<u32>;
@group(0) @binding(3) var<storage, read> heat: array<f32>;

const BG = vec3f(0.035, 0.043, 0.06);
const BOX = vec3f(0.07, 0.085, 0.115);
const BOX_HI = vec3f(0.09, 0.15, 0.22);

fn textColour(k: u32) -> vec3f {
  switch k {
    case 0u: { return vec3f(0.886, 0.910, 0.941); } // text
    case 1u: { return vec3f(0.392, 0.455, 0.545); } // dim
    case 2u: { return vec3f(1.0, 1.0, 1.0); }       // bright
    case 3u: { return vec3f(0.490, 0.827, 0.988); } // sky
    case 4u: { return vec3f(0.988, 0.827, 0.302); } // amber
    case 5u: { return vec3f(0.992, 0.643, 0.686); } // rose
    case 6u: { return vec3f(0.431, 0.906, 0.718); } // emerald
    default: { return vec3f(0.580, 0.639, 0.722); } // label
  }
}

/// Background tint of a cell: 1 surprise (warm), 2 attention (sky), 3 cursor, 4 guesses (amber → rose), 5 right.
fn tintColour(kind: u32, s: f32) -> vec4f {
  switch kind {
    case 1u: {
      let warm = mix(vec3f(0.55, 0.25, 0.05), vec3f(0.95, 0.35, 0.2), s);
      return vec4f(warm, 0.75 * s);
    }
    case 2u: { return vec4f(0.22, 0.6, 0.95, 0.9 * sqrt(s)); }
    case 3u: { return vec4f(0.49, 0.83, 0.99, 0.35 + 0.35 * s); }
    case 4u: { return vec4f(mix(vec3f(0.85, 0.6, 0.1), vec3f(0.9, 0.25, 0.3), s), 0.25 + 0.55 * s); }
    case 5u: { return vec4f(0.2, 0.7, 0.5, 0.35 * s); }
    default: { return vec4f(0.0); }
  }
}

fn glyphAlpha(g: u32, x: u32, y: u32) -> f32 {
  let i = (g * screen.glyph.y + y) * screen.glyph.x + x;
  return f32((atlas[i / 4u] >> (8u * (i % 4u))) & 255u) / 255.0;
}

/// Heat colour: dark → sky → white.
fn heatColour(v: f32) -> vec3f {
  let s = clamp(v, 0.0, 1.0);
  let a = mix(vec3f(0.06, 0.08, 0.12), vec3f(0.13, 0.45, 0.8), smoothstep(0.0, 0.5, s));
  return mix(a, vec3f(0.85, 0.95, 1.0), smoothstep(0.5, 1.0, s));
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let px = uv * screen.resolution;
  var colour = BG;
  for (var k = 0u; k < screen.count; k++) {
    let p = screen.panels[k];
    let local = px - p.rect.xy;
    if (local.x < 0.0 || local.y < 0.0 || local.x >= p.rect.z || local.y >= p.rect.w) { continue; }
    if (p.grid.z == 2u) {
      colour = select(BOX, BOX_HI, p.grid.x == 1u);
      continue;
    }
    let col = u32(local.x / p.cell.x);
    let row = u32(local.y / p.cell.y);
    if (col >= p.grid.x || row >= p.grid.y) { continue; }
    if (p.grid.z == 1u) {
      let inCell = local - vec2f(f32(col) * p.cell.x, f32(row) * p.cell.y);
      if (inCell.x < p.cell.x - p.cell.z && inCell.y < p.cell.y - p.cell.z) {
        colour = heatColour(heat[p.grid.w + row * p.grid.x + col]);
      }
      continue;
    }
    let word = cells[p.grid.w + row * p.grid.x + col];
    let tint = tintColour((word >> 12u) & 15u, f32(word >> 16u) / 65535.0);
    colour = mix(colour, tint.rgb, tint.a);
    let g = word & 255u;
    if (g < screen.glyphs) {
      let x = u32(local.x - f32(col) * p.cell.x);
      let y = u32(local.y - f32(row) * p.cell.y);
      if (x < screen.glyph.x && y < screen.glyph.y) {
        colour = mix(colour, textColour((word >> 8u) & 15u), glyphAlpha(g, x, y));
      }
    }
  }
  return vec4f(colour, 1.0);
}
