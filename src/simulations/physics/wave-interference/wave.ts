/**
 * CPU twin of wave.wgsl: the same phasor maths, used for the detector graph, the probe and the fringe
 * measurements (and compared against the shader in the GPU tests). Keep the two in sync.
 *
 * Simulation space: the screen is 4 units tall, y up, x spans `4 * aspect` centred on 0.
 */

export type WaveMode = "point" | "two-points" | "single-slit" | "double-slit";
export type WaveView = "amplitude" | "intensity" | "water";

export interface WaveParams {
  frequency: number;
  amplitude: number;
  damping: number;
  waveSpeed: number;
  mode: WaveMode;
  /** Distance between the two point sources / slit centres. */
  separation: number;
  slitWidth: number;
  view: WaveView;
  /** x position of the detector screen. */
  detectorX: number;
  /** Whether the detector screen and its graph are shown. */
  detector: boolean;
}

export const DEFAULTS: WaveParams = {
  frequency: 3.0,
  amplitude: 1.0,
  damping: 0.02,
  waveSpeed: 1.0,
  mode: "double-slit",
  separation: 0.8,
  slitWidth: 0.15,
  view: "amplitude",
  detectorX: 1.5,
  detector: true,
};

export const MODES: readonly WaveMode[] = ["point", "two-points", "single-slit", "double-slit"];
export const VIEWS: readonly WaveView[] = ["amplitude", "intensity", "water"];

/** x of the point sources and of the barrier (SOURCE_X in the shader). */
export const SOURCE_X = -1.5;
/** Secondary sources per slit (SLIT_SAMPLES in the shader). */
export const SLIT_SAMPLES = 24;
/** Visible half-height of the simulation space. */
export const HALF_HEIGHT = 2;

const TAU = 6.28318530718;

export type Phasor = [number, number];

export const wavelength = (p: Pick<WaveParams, "waveSpeed" | "frequency">) => p.waveSpeed / p.frequency;
const wavenumber = (p: WaveParams) => (TAU * p.frequency) / p.waveSpeed;

function cylindrical(p: WaveParams, sx: number, sy: number, x: number, y: number): Phasor {
  const r = Math.hypot(x - sx, y - sy);
  const a = (p.amplitude * Math.exp(-p.damping * r)) / (Math.sqrt(r) + 0.1);
  const phase = wavenumber(p) * r;
  return [a * Math.cos(phase), a * Math.sin(phase)];
}

function slit(p: WaveParams, centreY: number, x: number, y: number): Phasor {
  const dy = p.slitWidth / SLIT_SAMPLES;
  let c = 0;
  let s = 0;
  for (let i = 0; i < SLIT_SAMPLES; i++) {
    const sy = centreY + (i + 0.5 - 0.5 * SLIT_SAMPLES) * dy;
    const r = Math.hypot(x - SOURCE_X, y - sy);
    const a = (p.amplitude * Math.exp(-p.damping * r)) / (SLIT_SAMPLES * Math.sqrt(1 + r));
    const phase = wavenumber(p) * r;
    c += a * Math.cos(phase);
    s += a * Math.sin(phase);
  }
  return [c, s];
}

/** Complex amplitude (cos part, sin part) of the field at (x, y). */
export function phasor(p: WaveParams, x: number, y: number): Phasor {
  if (p.mode === "point") return cylindrical(p, SOURCE_X, 0, x, y);
  if (p.mode === "two-points") {
    const a = cylindrical(p, SOURCE_X, 0.5 * p.separation, x, y);
    const b = cylindrical(p, SOURCE_X, -0.5 * p.separation, x, y);
    return [a[0] + b[0], a[1] + b[1]];
  }
  if (x < SOURCE_X) {
    const phase = wavenumber(p) * (x - SOURCE_X);
    return [p.amplitude * Math.cos(phase), p.amplitude * Math.sin(phase)];
  }
  const first = slit(p, p.mode === "double-slit" ? 0.5 * p.separation : 0, x, y);
  if (p.mode === "single-slit") return first;
  const second = slit(p, -0.5 * p.separation, x, y);
  return [first[0] + second[0], first[1] + second[1]];
}

/** Time-averaged intensity relative to the source amplitude squared (what the Intensity view shows). */
export function intensity(p: WaveParams, x: number, y: number): number {
  const [c, s] = phasor(p, x, y);
  return (0.5 * (c * c + s * s)) / (p.amplitude * p.amplitude);
}

/** Peak amplitude |psi| at a point (independent of time). */
export function peakAmplitude(p: WaveParams, x: number, y: number): number {
  const [c, s] = phasor(p, x, y);
  return Math.hypot(c, s);
}

/** Intensity along the vertical line `x`, from y = -2 to +2 inclusive. */
export function intensityProfile(p: WaveParams, x: number, samples = 481): number[] {
  return Array.from({ length: samples }, (_, i) => intensity(p, x, -HALF_HEIGHT + (i / (samples - 1)) * 2 * HALF_HEIGHT));
}

/** y coordinate of profile sample `i`. */
export const profileY = (i: number, samples: number) => -HALF_HEIGHT + (i / (samples - 1)) * 2 * HALF_HEIGHT;

/** Indices of local maxima that reach at least `minFraction` of the profile's maximum. */
export function findPeaks(values: number[], minFraction = 0.3): number[] {
  const max = Math.max(...values);
  const peaks: number[] = [];
  for (let i = 1; i < values.length - 1; i++) {
    if (values[i] > values[i - 1] && values[i] >= values[i + 1] && values[i] >= minFraction * max) peaks.push(i);
  }
  return peaks;
}

/**
 * Mean distance between neighbouring bright fringes near the centre of the screen, or undefined with
 * fewer than two. Only the `maxPeaks` fringes closest to y = 0 are used (three by default: the centre and its neighbours): away from the axis the
 * fringes spread out (spacing ~ 1 / cos^2 of the angle), which the small-angle formula ignores.
 */
export function measuredFringeSpacing(values: number[], maxPeaks = 3): number | undefined {
  const centre = (values.length - 1) / 2;
  const peaks = findPeaks(values)
    .sort((a, b) => Math.abs(a - centre) - Math.abs(b - centre))
    .slice(0, maxPeaks)
    .sort((a, b) => a - b);
  if (peaks.length < 2) return undefined;
  const span = profileY(peaks[peaks.length - 1], values.length) - profileY(peaks[0], values.length);
  return span / (peaks.length - 1);
}

/** Small-angle double-slit fringe spacing  lambda * L / d  (undefined for other sources). */
export function theoreticalFringeSpacing(p: WaveParams): number | undefined {
  if (p.mode !== "double-slit" && p.mode !== "two-points") return undefined;
  const distance = p.detectorX - SOURCE_X;
  return distance > 0 ? (wavelength(p) * distance) / p.separation : undefined;
}

export interface DetectorReadouts {
  profile: number[];
  measuredSpacing?: number;
  theorySpacing?: number;
}

export function detectorReadouts(p: WaveParams): DetectorReadouts {
  const profile = intensityProfile(p, p.detectorX);
  return { profile, measuredSpacing: measuredFringeSpacing(profile), theorySpacing: theoreticalFringeSpacing(p) };
}

/** Canvas pixel (CSS px, origin top-left) -> simulation coordinates, matching wave.wgsl. */
export function viewToWorld(xPx: number, yPx: number, width: number, height: number): [number, number] {
  const aspect = width / height;
  return [(xPx / width - 0.5) * aspect * 4, (0.5 - yPx / height) * 4];
}

/** Inverse of `viewToWorld`. */
export function worldToView(x: number, y: number, width: number, height: number): [number, number] {
  const aspect = width / height;
  return [(x / (aspect * 4) + 0.5) * width, (0.5 - y / 4) * height];
}
