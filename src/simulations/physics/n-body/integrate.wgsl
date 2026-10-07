// Per-body time integration. Leapfrog (kick–drift–kick) is two dispatches around the force pass:
//   mode 0: v += a·dt/2; x += v·dt      mode 1: v += a·dt/2
// mode 2 is explicit Euler (x += v·dt with the old v; v += a·dt), kept for the "numerics matter" lesson.
// Keep in sync with step() in nbody.ts.

struct StepParams {
  n: u32,
  mode: u32,
  dt: f32,
  pad: f32,
}

@group(0) @binding(0) var<storage, read_write> pos: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> vel: array<vec4f>;
@group(0) @binding(2) var<storage, read> acc: array<vec4f>;
@group(0) @binding(3) var<uniform> params: StepParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) {
    return;
  }
  let dt = params.dt;
  let a = acc[i].xyz;
  var p = pos[i];
  var v = vel[i].xyz;
  if (params.mode == 0u) {
    v += 0.5 * dt * a;
    p = vec4f(p.xyz + dt * v, p.w);
  } else if (params.mode == 1u) {
    v += 0.5 * dt * a;
  } else {
    p = vec4f(p.xyz + dt * v, p.w);
    v += dt * a;
  }
  pos[i] = p;
  vel[i] = vec4f(v, 0.0);
}
