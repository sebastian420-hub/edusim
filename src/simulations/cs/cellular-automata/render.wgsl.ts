export const shaderSource = `
@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<uniform> params: RenderParams;

struct RenderParams {
  gridWidth: u32,
  gridHeight: u32,
  canvasWidth: f32,
  canvasHeight: f32,
  panX: f32,
  panY: f32,
  zoom: f32,
  theme: u32,
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let u = uv.x;
  let v = uv.y;
  
  // Apply pan and zoom
  let zoom = params.zoom;
  let pan = vec2f(params.panX, params.panY);
  
  // Convert screen coordinates to grid coordinates
  let gridX_f = (u - pan.x) * (f32(params.gridWidth) / zoom);
  let gridY_f = (v - pan.y) * (f32(params.gridHeight) / zoom);
  
  let x = u32(floor(gridX_f));
  let y = u32(floor(gridY_f));
  
  var color = vec3f(0.05, 0.05, 0.06); // Default background

  // Grid lines
  let cellFractX = fract(gridX_f);
  let cellFractY = fract(gridY_f);
  let lineThickness = 0.05;
  var isLine = false;
  
  // Only show grid lines if sufficiently zoomed in
  if (zoom > 10.0) {
    if (cellFractX < lineThickness || cellFractY < lineThickness) {
      isLine = true;
    }
  }

  if (x < params.gridWidth && y < params.gridHeight && gridX_f >= 0.0 && gridY_f >= 0.0) {
    let idx = y * params.gridWidth + x;
    let alive = cells[idx];
    
    if (alive == 1u) {
      // Themes: 0 = Classic Green, 1 = Cyberpunk Neon, 2 = Minimal White
      if (params.theme == 0u) {
        color = vec3f(0.0, 0.8, 0.2); // Green
      } else if (params.theme == 1u) {
        color = vec3f(0.0, 1.0, 1.0); // Cyan/Neon
      } else {
        color = vec3f(0.9, 0.9, 0.9); // White
      }
    } else {
      if (isLine) {
        color = vec3f(0.1, 0.1, 0.15); // Grid line color
      }
    }
  }

  return vec4f(color, 1.0);
}
`;
