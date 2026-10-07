// Fills an n×n grid of pendulums for the fractal, one per pixel, released from rest at the pixel's angles
// (θ₁ across, θ₂ upwards) inside the view window. Keep in sync with pixelAngles() in pendulum.ts.

struct SeedParams {
  n: u32,
  pad: u32,
  cx: f32,
  cy: f32,
  span: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
}

@group(0) @binding(0) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> flip: array<f32>;
@group(0) @binding(2) var<uniform> params: SeedParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = params.n;
  if (gid.x >= n || gid.y >= n) {
    return;
  }
  let fn_ = f32(n);
  let t1 = params.cx + ((f32(gid.x) + 0.5) / fn_ - 0.5) * params.span;
  let t2 = params.cy + (0.5 - (f32(gid.y) + 0.5) / fn_) * params.span;
  let i = gid.y * n + gid.x;
  state[i] = vec4f(t1, t2, 0.0, 0.0);
  flip[i] = -1.0;
}
