// Advances every double pendulum by `steps` steps of RK4 (or explicit Euler): one invocation per pendulum.
// Also records each pendulum's first flip (an arm passing over the top: |θ| > π, angles are not wrapped).
// Keep in sync with derivatives() / step() / hasFlipped() in pendulum.ts.
//
// state[i] = (θ₁, θ₂, ω₁, ω₂);  flip[i] = time of the first flip, or −1.

struct StepParams {
  n: u32,
  steps: u32,
  integrator: u32, // 0 = RK4, 1 = Euler
  pad: u32,
  dt: f32,
  time: f32, // simulation time at the start of this dispatch
  m1: f32,
  m2: f32,
  l1: f32,
  l2: f32,
  g: f32,
  pad2: f32,
}

@group(0) @binding(0) var<storage, read_write> state: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> flip: array<f32>;
@group(0) @binding(2) var<uniform> params: StepParams;

const PI: f32 = 3.14159265358979;

fn deriv(s: vec4f) -> vec4f {
  let m1 = params.m1;
  let m2 = params.m2;
  let l1 = params.l1;
  let l2 = params.l2;
  let g = params.g;
  let d = s.x - s.y;
  let den = 2.0 * m1 + m2 - m2 * cos(2.0 * s.x - 2.0 * s.y);
  let a1 = (-g * (2.0 * m1 + m2) * sin(s.x) - m2 * g * sin(s.x - 2.0 * s.y) - 2.0 * sin(d) * m2 * (s.w * s.w * l2 + s.z * s.z * l1 * cos(d))) / (l1 * den);
  let a2 = (2.0 * sin(d) * (s.z * s.z * l1 * (m1 + m2) + g * (m1 + m2) * cos(s.x) + s.w * s.w * l2 * m2 * cos(d))) / (l2 * den);
  return vec4f(s.z, s.w, a1, a2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  let dt = params.dt;
  var s = state[i];
  var flipped = flip[i];
  for (var k = 0u; k < params.steps; k++) {
    if (params.integrator == 0u) {
      let k1 = deriv(s);
      let k2 = deriv(s + 0.5 * dt * k1);
      let k3 = deriv(s + 0.5 * dt * k2);
      let k4 = deriv(s + dt * k3);
      s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    } else {
      s = s + dt * deriv(s);
    }
    if (flipped < 0.0 && (abs(s.x) > PI || abs(s.y) > PI)) {
      flipped = params.time + f32(k + 1u) * dt;
    }
  }
  state[i] = s;
  flip[i] = flipped;
}
