/**
 * CPU twin of the double-pendulum shaders, plus everything measured from them: energy, flips, normal modes,
 * periods, the spread between nearby runs and its exponential growth rate (Lyapunov exponent), and the
 * fractal's pixel ↔ starting-angle mapping. The GPU tests compare the shaders against these functions.
 *
 * Model: two point masses on massless rigid rods; angles θ₁, θ₂ measured from straight down (absolute, not
 * wrapped, so a flip shows as |θ| passing π); ω = dθ/dt. State layout matches the GPU: (θ₁, θ₂, ω₁, ω₂).
 */

export interface PendulumParams {
  m1: number;
  m2: number;
  l1: number;
  l2: number;
  g: number;
}

export const DEFAULT_PENDULUM: PendulumParams = { m1: 1, m2: 1, l1: 1, l2: 1, g: 9.81 };

export type State = [number, number, number, number];
export type Integrator = "rk4" | "euler";

/** dState/dt: the standard equations of motion of the double pendulum (Lagrangian, solved for θ̈₁, θ̈₂). */
export function derivatives([t1, t2, w1, w2]: State, { m1, m2, l1, l2, g }: PendulumParams): State {
  const d = t1 - t2;
  const den = 2 * m1 + m2 - m2 * Math.cos(2 * t1 - 2 * t2);
  const a1 = (-g * (2 * m1 + m2) * Math.sin(t1) - m2 * g * Math.sin(t1 - 2 * t2) - 2 * Math.sin(d) * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * Math.cos(d))) / (l1 * den);
  const a2 = (2 * Math.sin(d) * (w1 * w1 * l1 * (m1 + m2) + g * (m1 + m2) * Math.cos(t1) + w2 * w2 * l2 * m2 * Math.cos(d))) / (l2 * den);
  return [w1, w2, a1, a2];
}

const axpy = (s: State, k: State, h: number): State => [s[0] + h * k[0], s[1] + h * k[1], s[2] + h * k[2], s[3] + h * k[3]];

/** One step: classical Runge–Kutta 4 (accurate; energy error tiny and shown) or explicit Euler (for the lesson). */
export function step(s: State, dt: number, p: PendulumParams, integrator: Integrator = "rk4"): State {
  if (integrator === "euler") return axpy(s, derivatives(s, p), dt);
  const k1 = derivatives(s, p);
  const k2 = derivatives(axpy(s, k1, dt / 2), p);
  const k3 = derivatives(axpy(s, k2, dt / 2), p);
  const k4 = derivatives(axpy(s, k3, dt), p);
  return [0, 1, 2, 3].map((i) => s[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])) as State;
}

export function energy([t1, t2, w1, w2]: State, { m1, m2, l1, l2, g }: PendulumParams): { kinetic: number; potential: number; total: number } {
  const kinetic = 0.5 * m1 * l1 * l1 * w1 * w1 + 0.5 * m2 * (l1 * l1 * w1 * w1 + l2 * l2 * w2 * w2 + 2 * l1 * l2 * w1 * w2 * Math.cos(t1 - t2));
  const potential = -(m1 + m2) * g * l1 * Math.cos(t1) - m2 * g * l2 * Math.cos(t2);
  return { kinetic, potential, total: kinetic + potential };
}

/** Bob positions with y pointing up and the pivot at the origin. */
export function positions([t1, t2]: State, { l1, l2 }: Pick<PendulumParams, "l1" | "l2">): { x1: number; y1: number; x2: number; y2: number } {
  const x1 = l1 * Math.sin(t1);
  const y1 = -l1 * Math.cos(t1);
  return { x1, y1, x2: x1 + l2 * Math.sin(t2), y2: y1 - l2 * Math.cos(t2) };
}

/** True once either arm has gone over the top (angles are not wrapped). */
export const hasFlipped = ([t1, t2]: State) => Math.abs(t1) > Math.PI || Math.abs(t2) > Math.PI;

/** Time of the first flip within `tMax`, or undefined. Exactly what the fractal kernel computes per pixel. */
export function flipTime(start: State, p: PendulumParams, dt: number, tMax: number, integrator: Integrator = "rk4"): number | undefined {
  let s = start;
  const steps = Math.round(tMax / dt);
  for (let k = 1; k <= steps; k++) {
    s = step(s, dt, p, integrator);
    if (hasFlipped(s)) return k * dt;
  }
  return undefined;
}

/**
 * Energy forbids every flip: for equal point masses and lengths released from rest, E = −mgl(2 cos θ₁ + cos θ₂)
 * and the cheapest flip (lower arm over the top, upper arm straight down) needs −mgl. So no flip if
 * 2 cos θ₁ + cos θ₂ > 1. (For general masses and lengths the same reasoning is in `canNeverFlip`.)
 */
export const flipForbiddenEqual = (t1: number, t2: number) => 2 * Math.cos(t1) + Math.cos(t2) > 1;

/** General version from energy: released at `state`, can either arm ever reach the top? */
export function canNeverFlip(state: State, p: PendulumParams): boolean {
  const E = energy(state, p).total;
  // Lowest potential energy with arm 2 pointing up (θ₂ = π, θ₁ = 0) or arm 1 up (θ₁ = π, θ₂ = 0).
  const lower = -(p.m1 + p.m2) * p.g * p.l1 + p.m2 * p.g * p.l2;
  const upper = (p.m1 + p.m2) * p.g * p.l1 - p.m2 * p.g * p.l2;
  return E < Math.min(lower, upper);
}

/**
 * The same energy test as a curve in the map of starting angles (released from rest): no flip is possible where
 * a·cos θ₁ + b·cos θ₂ > c, with a = (m₁ + m₂)·l₁, b = m₂·l₂ and c = |a − b|. Equal masses and lengths: 2, 1, 1.
 */
export function flipBoundary({ m1, m2, l1, l2 }: Pick<PendulumParams, "m1" | "m2" | "l1" | "l2">): { a: number; b: number; c: number } {
  const a = (m1 + m2) * l1;
  const b = m2 * l2;
  return { a, b, c: Math.abs(a - b) };
}

/** Small-angle normal modes (equal masses and lengths): ω² = (g/l)(2 ∓ √2); slow mode in phase, θ₂ = √2·θ₁. */
export function normalModes(p: Pick<PendulumParams, "g" | "l1">) {
  const slow = Math.sqrt((p.g / p.l1) * (2 - Math.SQRT2));
  const fast = Math.sqrt((p.g / p.l1) * (2 + Math.SQRT2));
  return { slow: { omega: slow, period: (2 * Math.PI) / slow, ratio: Math.SQRT2 }, fast: { omega: fast, period: (2 * Math.PI) / fast, ratio: -Math.SQRT2 } };
}

/** Measures oscillation periods from upward zero crossings of a signal (θ₁), interpolated between samples. */
export class PeriodTracker {
  private last: { t: number; v: number } | undefined;
  private crossings: number[] = [];

  reset(): void {
    this.last = undefined;
    this.crossings = [];
  }

  sample(t: number, v: number): void {
    const prev = this.last;
    this.last = { t, v };
    if (!prev || !(prev.v < 0 && v >= 0)) return;
    this.crossings.push(prev.t + ((0 - prev.v) / (v - prev.v)) * (t - prev.t));
    if (this.crossings.length > 64) this.crossings.shift();
  }

  /** The last `n` full periods (oldest first). */
  periods(n = 8): number[] {
    const c = this.crossings;
    const out: number[] = [];
    for (let i = Math.max(1, c.length - n); i < c.length; i++) out.push(c[i] - c[i - 1]);
    return out;
  }
}

/** Angle difference folded into (−π, π], so the spread saturates at "completely different" instead of growing on. */
export const angleGap = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));

/** Distance in (θ₁, θ₂) between two pendulums: how different they look. */
export const separation = (a: State, b: State) => Math.hypot(angleGap(a[0], b[0]), angleGap(a[1], b[1]));

/**
 * The exponential growth rate of the separation (the largest Lyapunov exponent, per second): the least-squares
 * slope of ln(separation) against time over the samples where it is growing but not yet saturated.
 * Undefined until there are enough such samples.
 */
export function growthRate(times: number[], seps: number[], lo = 1e-7, hi = 0.3): number | undefined {
  const pts: [number, number][] = [];
  times.forEach((t, i) => {
    if (seps[i] > lo && seps[i] < hi) pts.push([t, Math.log(seps[i])]);
  });
  if (pts.length < 6) return undefined;
  const n = pts.length;
  const mt = pts.reduce((a, p) => a + p[0], 0) / n;
  const ml = pts.reduce((a, p) => a + p[1], 0) / n;
  let num = 0;
  let den = 0;
  for (const [t, l] of pts) {
    num += (t - mt) * (l - ml);
    den += (t - mt) * (t - mt);
  }
  return den > 0 ? num / den : undefined;
}

/** The fractal's view window: centre (θ₁, θ₂) and the width of the square it shows, in radians. */
export interface FractalWindow {
  cx: number;
  cy: number;
  span: number;
}

export const FULL_WINDOW: FractalWindow = { cx: 0, cy: 0, span: 2 * Math.PI };

/** Starting angles of pixel (i, j) of an n×n map (j = 0 is the top row; θ₂ increases upwards). As fractal-seed.wgsl. */
export function pixelAngles(i: number, j: number, n: number, w: FractalWindow): [number, number] {
  return [w.cx + ((i + 0.5) / n - 0.5) * w.span, w.cy + (0.5 - (j + 0.5) / n) * w.span];
}

/** Inverse of `pixelAngles` for a point in [0, 1]² of the map (u right, v down). */
export function mapAngles(u: number, v: number, w: FractalWindow): [number, number] {
  return [w.cx + (u - 0.5) * w.span, w.cy + (0.5 - v) * w.span];
}

/** The next 32-bit float above x (x finite). */
export function nextFloat32(x: number): number {
  const f = new Float32Array([x]);
  const bits = new Int32Array(f.buffer);
  if (f[0] === 0) bits[0] = 1;
  else bits[0] += f[0] > 0 ? 1 : -1;
  return f[0];
}

/**
 * Upper-arm start angles of a crowd of `n` pendulums `nudge` apart, as 32-bit floats for the GPU. A 32-bit float
 * near 2 rad cannot tell apart angles closer than about 2·10⁻⁷ rad, so neighbours that would round to the same
 * value are pushed one float step apart instead: every pendulum stays distinct (and fans out).
 */
export function crowdAngles(theta: number, nudge: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const want = Math.fround(theta + i * nudge);
    out[i] = i > 0 && want <= out[i - 1] ? nextFloat32(out[i - 1]) : want;
  }
  return out;
}

const RAD = Math.PI / 180;
export const deg = (r: number) => r / RAD;
export const rad = (d: number) => d * RAD;
