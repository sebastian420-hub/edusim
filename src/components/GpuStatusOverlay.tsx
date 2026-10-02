import type { GpuStatus } from "@/lib/gpu/runtime";

/** Covers the canvas while the GPU starts, or explains why it can't run. */
export function GpuStatusOverlay({ status }: { status: GpuStatus }) {
  if (status.state === "ready") return null;

  if (status.state === "loading") {
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80" role="status">
        <p className="animate-pulse text-sm text-slate-200">Initializing GPU…</p>
      </div>
    );
  }

  const unsupported = status.state === "unsupported";
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-slate-950/90 p-6" role="alert">
      <div className="max-w-md text-center text-slate-200">
        <p className="mb-2 text-lg font-bold">
          {unsupported ? "WebGPU isn’t available in this browser" : "The GPU simulation failed"}
        </p>
        <p className="text-sm text-slate-400">
          {unsupported
            ? "EduSim runs on WebGPU. Try a recent version of Chrome, Edge or Safari on desktop (Firefox is still catching up), and make sure hardware acceleration is enabled."
            : status.message}
        </p>
      </div>
    </div>
  );
}
