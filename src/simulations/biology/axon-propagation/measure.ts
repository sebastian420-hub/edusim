import type { Fibre } from "./cable";
import { compartmentAt } from "./model";

export type Outcome = "conducted" | "blocked" | "no-spike";

export interface AxonReadout {
  /** First arrival at recording electrodes 1 and 2 (ms), if any. */
  t1?: number;
  t2?: number;
  /** Electrode separation, cm. */
  distance: number;
  /** Speed measured between the electrodes, m/s. */
  velocity?: number;
  /** Spikes recorded at each electrode, and at the far end. */
  spikes1: number;
  spikes2: number;
  spikesEnd: number;
  outcome: Outcome;
  /** Peak voltage at each electrode (mV), when known. */
  peak1?: number;
  peak2?: number;
}

/**
 * Readouts of one sweep from per-compartment arrival times and crossing counts (from the GPU, or the twin):
 * what a student measures with two recording electrodes.
 */
export function axonReadout(f: Fibre, arrival: ArrayLike<number>, crossings: ArrayLike<number>, e1: number, e2: number, peaks?: [number, number]): AxonReadout {
  const i1 = compartmentAt(f, Math.min(e1, e2));
  const i2 = compartmentAt(f, Math.max(e1, e2));
  const valid = (t: number) => (Number.isFinite(t) && t >= 0 ? t : undefined);
  const t1 = valid(arrival[i1]);
  const t2 = valid(arrival[i2]);
  const distance = f.x[i2] - f.x[i1];
  const velocity = t1 !== undefined && t2 !== undefined && t2 > t1 ? (distance / (t2 - t1)) * 10 : undefined;
  // Did a spike start? Look just beyond the stimulated stretch (the first 2.5 % of the fibre).
  const end = f.n - 1;
  const started = crossings[compartmentAt(f, 0.06)] > 0 || crossings[end] > 0;
  const outcome: Outcome = !started ? "no-spike" : crossings[end] > 0 ? "conducted" : "blocked";
  return { t1, t2, distance, velocity, spikes1: crossings[i1], spikes2: crossings[i2], spikesEnd: crossings[end], outcome, peak1: peaks?.[0], peak2: peaks?.[1] };
}
