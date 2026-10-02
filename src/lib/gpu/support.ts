/**
 * WebGPU feature detection, in a module with no imports so that code which only needs the check (like the
 * home page) doesn't pull in the GPU library.
 */
export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && !!navigator.gpu;
}
