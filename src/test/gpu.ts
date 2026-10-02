import { init } from "vgpu/node";

export type NodeGpu = Awaited<ReturnType<typeof init>>;

/**
 * Acquires a headless WebGPU device (hardware, or the CPU software renderer from
 * `npx vgpu install-software-renderer`). Returns null when none is available so local runs
 * without a driver skip the GPU tests; CI sets REQUIRE_GPU=1 to make that a failure instead.
 */
export async function tryInitGpu(): Promise<NodeGpu | null> {
  try {
    return await init();
  } catch (e) {
    if (process.env.REQUIRE_GPU) throw e;
    console.warn("No headless WebGPU device available; GPU tests skipped.");
    return null;
  }
}
