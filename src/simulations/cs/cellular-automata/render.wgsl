@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<uniform> params: RenderParams;

struct RenderParams {
  gridWidth: u32,
  gridHeight: u32,
  resolution: vec2f, // canvas size in device pixels
  center: vec2f,     // grid coordinate shown at the middle of the canvas
  cellPx: f32,       // device pixels per cell (same on both axes, so cells are square)
  theme: u32,
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  // Canvas pixel -> grid coordinate (uv.y points down, matching grid rows).
  let px = uv * params.resolution;
  let g = params.center + (px - 0.5 * params.resolution) / params.cellPx;

  let outside = vec3f(0.02, 0.02, 0.025);
  if (g.x < 0.0 || g.y < 0.0 || g.x >= f32(params.gridWidth) || g.y >= f32(params.gridHeight)) {
    return vec4f(outside, 1.0);
  }

  let x = u32(g.x);
  let y = u32(g.y);
  var color = vec3f(0.05, 0.05, 0.06);

  // Grid lines, one device pixel wide, only once cells are big enough to read.
  if (params.cellPx >= 8.0) {
    let f = fract(g);
    let edge = 1.0 / params.cellPx;
    if (f.x < edge || f.y < edge) {
      color = vec3f(0.1, 0.1, 0.15);
    }
  }

  if (cells[y * params.gridWidth + x] == 1u) {
    // Themes: 0 = Classic Green, 1 = Cyberpunk Neon, 2 = Minimal White
    if (params.theme == 0u) {
      color = vec3f(0.0, 0.8, 0.2);
    } else if (params.theme == 1u) {
      color = vec3f(0.0, 1.0, 1.0);
    } else {
      color = vec3f(0.9, 0.9, 0.9);
    }
  }
  return vec4f(color, 1.0);
}
