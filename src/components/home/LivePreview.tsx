"use client";
import { useEffect } from "react";
import type { LaunchOptions } from "@/lib/gpu/runtime";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { PREVIEWS } from "@/simulations/previews";

/** Small previews don't need a 2x canvas. */
const PREVIEW_OPTIONS: LaunchOptions = { maxDpr: 1.5 };

/**
 * The real simulation, running small inside a plate. Fades in over the poster once its first frame is
 * ready; reports failure (no WebGPU, GPU error) so the plate falls back to the poster for good.
 */
export default function LivePreview({ id, onFail }: { id: string; onFail: () => void }) {
  const preview = PREVIEWS[id];
  const { canvasRef, status, sim } = useGpuSim(preview.factory, PREVIEW_OPTIONS);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!sim || !canvas) return;
    return preview.setup(sim, canvas) ?? undefined;
  }, [sim, canvasRef, preview]);

  const failed = status.state === "error" || status.state === "unsupported";
  useEffect(() => {
    if (failed) onFail();
  }, [failed, onFail]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      data-testid="plate-live"
      data-state={status.state}
      className={`absolute inset-0 h-full w-full transition-opacity duration-500 ${status.state === "ready" ? "opacity-100" : "opacity-0"}`}
    />
  );
}
