/** Workgroup size of gravity.wgsl / integrate.wgsl / trail.wgsl (64 invocations: valid on every adapter). */
export const WORKGROUP = 64;
export const workgroupsFor = (n: number) => Math.max(1, Math.ceil(n / WORKGROUP));

/** Points kept per orbit trail. */
export const TRAIL_LEN = 1024;
