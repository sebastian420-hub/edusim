/** Workgroup size of pendulum.wgsl (64 invocations: valid on every adapter). */
export const WORKGROUP = 64;
export const workgroupsFor = (n: number) => Math.max(1, Math.ceil(n / WORKGROUP));
/** Edge of seed.wgsl's 2-D workgroup. */
export const SEED_TILE = 8;
/** Points kept in the lower bob's trail. */
export const TRAIL_LEN = 2048;
