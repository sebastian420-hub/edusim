/**
 * Workgroup edge length of compute.wgsl (`@workgroup_size(8, 8)`, 64 invocations). Kept small so the
 * shader is valid on every adapter, including software renderers with a 128-invocation limit.
 */
export const WORKGROUP_SIZE = 8;

export const workgroupsFor = (cells: number): number => Math.ceil(cells / WORKGROUP_SIZE);
