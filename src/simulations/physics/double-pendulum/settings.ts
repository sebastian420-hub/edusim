import { boolField, enumField, numberField } from "@/lib/urlState";
import type { FieldCodec, Schema } from "@/lib/urlState";
import { FULL_WINDOW, rad } from "./pendulum";
import type { FractalWindow, Integrator, PendulumParams } from "./pendulum";

export type PendulumView = "pendulum" | "butterfly" | "fractal";
export const CROWD_COUNTS = [2, 100, 1000, 10000] as const;
export type CrowdCount = (typeof CROWD_COUNTS)[number];
export const FRACTAL_SIZES = [512, 1024] as const;
export type FractalSize = (typeof FRACTAL_SIZES)[number];

/** Narrowest fractal window (radians): f32 angles still resolve its pixels at 1024². */
export const MIN_SPAN = 1e-3;

/** Everything a shared link reproduces. Angles are in degrees (as shown), the fractal window in radians. */
export interface PendulumSettings {
  view: PendulumView;
  /** Release angles from straight down (released from rest). */
  a1: number;
  a2: number;
  /** Lower mass and arm length, relative to the upper ones (m₁ = 1 kg, l₁ = 1 m). */
  m2: number;
  l2: number;
  g: number;
  /** Simulated seconds per second. */
  speed: number;
  integrator: Integrator;
  trail: boolean;
  /** Butterfly: how many pendulums, and how far apart they start (10^nudge rad). */
  count: CrowdCount;
  nudge: number;
  /** Fractal: the window of starting angles shown, and the grid size. */
  fx: number;
  fy: number;
  fspan: number;
  res: FractalSize;
  boundary: boolean;
}

export const PENDULUM_DEFAULTS: PendulumSettings = {
  view: "pendulum",
  a1: 120,
  a2: 150,
  m2: 1,
  l2: 1,
  g: 9.81,
  speed: 1,
  integrator: "rk4",
  trail: true,
  count: 100,
  nudge: -9,
  fx: FULL_WINDOW.cx,
  fy: FULL_WINDOW.cy,
  fspan: FULL_WINDOW.span,
  res: 512,
  boundary: true,
};

/** Seven significant digits: a deep fractal zoom still reproduces from its link. */
const preciseField = (min: number, max: number): FieldCodec<number> => {
  const base = numberField(min, max);
  return { decode: base.decode, encode: (v) => String(Number(v.toPrecision(7))) };
};

const countField: FieldCodec<CrowdCount> = { decode: (raw) => CROWD_COUNTS.find((c) => String(c) === raw), encode: String };
const resField: FieldCodec<FractalSize> = { decode: (raw) => FRACTAL_SIZES.find((c) => String(c) === raw), encode: String };

export const PENDULUM_SCHEMA: Schema<PendulumSettings> = {
  view: enumField(["pendulum", "butterfly", "fractal"] as const),
  a1: numberField(-180, 180),
  a2: numberField(-180, 180),
  m2: numberField(0.1, 5),
  l2: numberField(0.25, 2),
  g: numberField(1, 25),
  speed: numberField(0.1, 2),
  integrator: enumField(["rk4", "euler"] as const),
  trail: boolField,
  count: countField,
  nudge: numberField(-12, -1),
  fx: preciseField(-2 * Math.PI, 2 * Math.PI),
  fy: preciseField(-2 * Math.PI, 2 * Math.PI),
  fspan: preciseField(MIN_SPAN, 2 * Math.PI),
  res: resField,
  boundary: boolField,
};

export const physicsOf = (s: Pick<PendulumSettings, "m2" | "l2" | "g">): PendulumParams => ({ m1: 1, m2: s.m2, l1: 1, l2: s.l2, g: s.g });
export const windowOf = (s: Pick<PendulumSettings, "fx" | "fy" | "fspan">): FractalWindow => ({ cx: s.fx, cy: s.fy, span: s.fspan });
export const startOf = (s: Pick<PendulumSettings, "a1" | "a2">): [number, number] => [rad(s.a1), rad(s.a2)];
/** True for equal masses and lengths, where the normal modes and the 2 cos θ₁ + cos θ₂ boundary hold exactly. */
export const isSymmetric = (s: Pick<PendulumSettings, "m2" | "l2">) => s.m2 === 1 && s.l2 === 1;

/** Zooms the fractal window by `factor` (> 1 = in) keeping the map point (u, v) ∈ [0, 1]² where it is. */
export function zoomWindow(w: FractalWindow, factor: number, u: number, v: number): FractalWindow {
  const span = Math.min(2 * Math.PI, Math.max(MIN_SPAN, w.span / factor));
  const ax = w.cx + (u - 0.5) * w.span;
  const ay = w.cy + (0.5 - v) * w.span;
  return clampWindow({ cx: ax - (u - 0.5) * span, cy: ay - (0.5 - v) * span, span });
}

/** Moves the window by a fraction of its width (du right, dv down on screen). */
export const panWindow = (w: FractalWindow, du: number, dv: number): FractalWindow => clampWindow({ cx: w.cx - du * w.span, cy: w.cy + dv * w.span, span: w.span });

const clampWindow = (w: FractalWindow): FractalWindow => {
  const limit = 2 * Math.PI;
  return { cx: Math.min(limit, Math.max(-limit, w.cx)), cy: Math.min(limit, Math.max(-limit, w.cy)), span: w.span };
};
