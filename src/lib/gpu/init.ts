import { init } from "vgpu";

export type GPUStatus = "ready" | "initializing" | "unsupported" | "error";

export async function initGPU() {
  if (typeof navigator === "undefined" || !(navigator as any).gpu) {
    return { gpu: null, status: "unsupported" as GPUStatus, error: "WebGPU not supported" };
  }
  try {
    const gpu = await init();
    return { gpu, status: "ready" as GPUStatus, error: null };
  } catch (e) {
    return { gpu: null, status: "error" as GPUStatus, error: String(e) };
  }
}
