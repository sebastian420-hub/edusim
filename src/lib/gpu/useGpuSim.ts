"use client";
import { useEffect, useRef, useState } from "react";
import { launchSim } from "./runtime";
import type { GpuStatus, SimFactory, SimHandle } from "./runtime";

/**
 * Owns the canvas + GPU lifecycle of a simulation.
 *
 * `factory` must be referentially stable (declare it at module level). Returns the canvas ref to
 * attach, the GPU status, and the sim handle once ready — so `useEffect(() => sim?.setParams(p),
 * [sim, p])` also delivers parameters chosen while the GPU was still starting up.
 */
export function useGpuSim<H extends SimHandle>(factory: SimFactory<H>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<GpuStatus>({ state: "loading" });
  const [sim, setSim] = useState<H | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const teardown = launchSim(canvas, factory, setSim, setStatus);
    return () => {
      teardown();
      setSim(null);
      setStatus({ state: "loading" });
    };
  }, [factory]);

  return { canvasRef, status, sim };
}
