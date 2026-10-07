"use client";
import { useEffect, useState } from "react";

/** True when the visitor asked their system for reduced motion. Client-only. */
export function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

/**
 * Play/pause state owned by the page and kept in sync with the simulation (pass it to SimLayout as
 * `playing` / `onPlayPause`). Simulations that would start running start paused instead when the visitor
 * prefers reduced motion.
 */
export function usePlaying(sim: { play(): void; pause(): void } | null, autoplay: boolean) {
  const [playing, setPlaying] = useState(() => autoplay && !prefersReducedMotion());
  useEffect(() => {
    if (playing) sim?.play();
    else sim?.pause();
  }, [sim, playing]);
  return [playing, setPlaying] as const;
}
