// All-pairs Plummer-softened gravity, tiled through workgroup memory (Nyland, Harris & Prins, GPU Gems 3,
// ch. 31): each workgroup loads 64 bodies at a time into shared memory and every invocation sums the pull of
// that tile on its own body. Also accumulates the potential φ_i (free: the distances are already computed).
// Keep in sync with computeAccelerations() in nbody.ts.
//
// pos[i] = (x, y, z, mass); acc[i] = (ax, ay, az, φ).

struct ForceParams {
  n: u32,
  pad: u32,
  g: f32,
  eps2: f32,
}

@group(0) @binding(0) var<storage, read> pos: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> acc: array<vec4f>;
@group(0) @binding(2) var<uniform> params: ForceParams;

const TILE: u32 = 64u; // keep in sync with WORKGROUP in sim-constants.ts

var<workgroup> tile: array<vec4f, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let i = gid.x;
  let n = params.n;
  var p = vec4f(0.0);
  if (i < n) {
    p = pos[i];
  }
  var a = vec3f(0.0);
  var phi = 0.0;
  let tiles = (n + TILE - 1u) / TILE;
  for (var t = 0u; t < tiles; t++) {
    let j = t * TILE + li;
    if (j < n) {
      tile[li] = pos[j];
    } else {
      tile[li] = vec4f(0.0); // zero mass: contributes nothing
    }
    workgroupBarrier();
    for (var k = 0u; k < TILE; k++) {
      let q = tile[k];
      let d = q.xyz - p.xyz;
      let inv = inverseSqrt(dot(d, d) + params.eps2);
      // Skip the body itself (mass 0 for padding already contributes nothing).
      let m = select(q.w, 0.0, t * TILE + k == i);
      a += (m * inv * inv * inv) * d;
      phi -= m * inv;
    }
    workgroupBarrier();
  }
  if (i < n) {
    acc[i] = vec4f(params.g * a, params.g * phi);
  }
}
