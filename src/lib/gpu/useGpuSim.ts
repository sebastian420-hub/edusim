"use client";
import { useEffect, useRef, useState } from "react";
import { launchSim } from "./runtime";
import type { GpuStatus, LaunchOptions, SimFactory, SimHandle } from "./runtime";

/**
 * Owns the canvas + GPU lifecycle of a simulation.
 *
 * `factory` and `options` must be referentially stable (declare them at module level). Returns the canvas ref to
 * attach, the GPU status, the sim handle once ready, and the current render scale (1 = full resolution) — so `useEffect(() => sim?.setParams(p),
 * [sim, p])` also delivers parameters chosen while the GPU was still starting up.
 */
export function useGpuSim<H extends SimHandle>(factory: SimFactory<H>, options?: LaunchOptions) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<GpuStatus>({ state: "loading" });
  const [sim, setSim] = useState<H | null>(null);
  const [quality, setQuality] = useState(1);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const teardown = launchSim(canvas, factory, setSim, setStatus, setQuality, options);
    return () => {
      teardown();
      setSim(null);
      setStatus({ state: "loading" });
      setQuality(1);
    };
  }, [factory, options]);

  return { canvasRef, status, sim, quality };
}
