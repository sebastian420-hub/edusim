/**
 * CPU reference implementation of the Hodgkin-Huxley model. It mirrors neuron.wgsl step for step
 * (Rush-Larsen gating, forward-Euler voltage) and is used by the unit tests to check both the
 * physiology and the GPU shader.
 */

export interface HHParams {
  g_Na: number;
  g_K: number;
  g_L: number;
  E_Na: number;
  E_K: number;
  E_L: number;
  C_m: number;
  I_inj: number;
  temperature: number;
  /** 0 = DC, 1 = single pulse every period, 2 = twin pulses every period. */
  pulse_mode: 0 | 1 | 2;
}

export interface HHState {
  V: number;
  m: number;
  h: number;
  n: number;
}

export const DEFAULT_PARAMS: HHParams = {
  g_Na: 120,
  g_K: 36,
  g_L: 0.3,
  E_Na: 50,
  E_K: -77,
  E_L: -54.387,
  C_m: 1,
  I_inj: 10,
  temperature: 6.3,
  pulse_mode: 0,
};

export const REST_STATE: HHState = { V: -65, m: 0.05, h: 0.6, n: 0.31 };

/** Integration step (ms). */
export const DT_MS = 0.01;
/** A history sample is recorded every this many integration steps. */
export const SAMPLE_EVERY = 5;
/** Samples kept in the GPU ring buffer. */
export const HISTORY_SAMPLES = 2048;
/** Width of the visible time window (ms). */
export const WINDOW_MS = HISTORY_SAMPLES * SAMPLE_EVERY * DT_MS;

// Stimulus timing (ms), repeated every PULSE_PERIOD. Must match get_current() in neuron.wgsl.
export const PULSE_PERIOD = 25;
export const PULSE_1 = [5, 6] as const;
export const PULSE_2 = [13, 14] as const;

export function appliedCurrent(tMs: number, p: Pick<HHParams, "I_inj" | "pulse_mode">): number {
  if (p.pulse_mode === 0) return p.I_inj;
  const t = tMs - PULSE_PERIOD * Math.floor(tMs / PULSE_PERIOD);
  const inPulse1 = t >= PULSE_1[0] && t <= PULSE_1[1];
  if (p.pulse_mode === 1) return inPulse1 ? p.I_inj : 0;
  const inPulse2 = t >= PULSE_2[0] && t <= PULSE_2[1];
  return inPulse1 || inPulse2 ? p.I_inj : 0;
}

/** Q10 = 3 temperature scaling of the gating kinetics, relative to the original 6.3 degC data. */
export function temperatureFactor(celsius: number): number {
  return Math.pow(3, (celsius - 6.3) / 10);
}

export interface Rates {
  am: number;
  bm: number;
  ah: number;
  bh: number;
  an: number;
  bn: number;
}

/** Voltage-dependent opening/closing rates (1/ms), voltage measured from the -65 mV rest. */
export function rates(V: number): Rates {
  const Vs = V + 65;
  const am = Math.abs(Vs - 25) < 0.001 ? 1 : (0.1 * (25 - Vs)) / (Math.exp((25 - Vs) / 10) - 1);
  const bm = 4 * Math.exp(-Vs / 18);
  const ah = 0.07 * Math.exp(-Vs / 20);
  const bh = 1 / (Math.exp((30 - Vs) / 10) + 1);
  const an = Math.abs(Vs - 10) < 0.001 ? 0.1 : (0.01 * (10 - Vs)) / (Math.exp((10 - Vs) / 10) - 1);
  const bn = 0.125 * Math.exp(-Vs / 80);
  return { am, bm, ah, bh, an, bn };
}

function relax(x: number, a: number, b: number, dt: number): number {
  const inf = a / (a + b);
  const tau = 1 / (a + b);
  return inf + (x - inf) * Math.exp(-dt / tau);
}

/** Advances the neuron by one step of `dt` ms starting at time `tMs`. */
export function stepHH(s: HHState, p: HHParams, tMs: number, dt = DT_MS): HHState {
  const phi = temperatureFactor(p.temperature);
  const r = rates(s.V);
  const m = relax(s.m, r.am * phi, r.bm * phi, dt);
  const h = relax(s.h, r.ah * phi, r.bh * phi, dt);
  const n = relax(s.n, r.an * phi, r.bn * phi, dt);

  const I_Na = p.g_Na * m * m * m * h * (s.V - p.E_Na);
  const I_K = p.g_K * n * n * n * n * (s.V - p.E_K);
  const I_L = p.g_L * (s.V - p.E_L);
  const V = s.V + (dt * (appliedCurrent(tMs, p) - I_Na - I_K - I_L)) / p.C_m;
  return { V, m, h, n };
}

export interface Trace {
  t: number[];
  V: number[];
}

/** Simulates `durationMs` and records the voltage every `sampleEvery` steps. */
export function simulate(
  p: HHParams,
  durationMs: number,
  { init = REST_STATE, dt = DT_MS, sampleEvery = SAMPLE_EVERY } = {},
): Trace {
  const steps = Math.round(durationMs / dt);
  const trace: Trace = { t: [], V: [] };
  let s = init;
  for (let i = 0; i < steps; i++) {
    s = stepHH(s, p, i * dt, dt);
    if ((i + 1) % sampleEvery === 0) {
      trace.t.push((i + 1) * dt);
      trace.V.push(s.V);
    }
  }
  return trace;
}

/** Counts upward crossings of `threshold` (mV). */
export function countSpikes(V: number[], threshold = 0): number {
  let count = 0;
  for (let i = 1; i < V.length; i++) {
    if (V[i - 1] < threshold && V[i] >= threshold) count++;
  }
  return count;
}

// ───────────────────────── measurements (used by the readouts, f–I curve and challenges) ─────────────────────────

/** Times (ms) at which the voltage crosses `threshold` upwards, linearly interpolated between samples. */
export function spikeTimes(trace: Trace, threshold = 0): number[] {
  const times: number[] = [];
  for (let i = 1; i < trace.V.length; i++) {
    if (trace.V[i - 1] < threshold && trace.V[i] >= threshold) {
      const f = (threshold - trace.V[i - 1]) / (trace.V[i] - trace.V[i - 1]);
      times.push(trace.t[i - 1] + f * (trace.t[i] - trace.t[i - 1]));
    }
  }
  return times;
}

/**
 * Steady-state firing rate (Hz) under sustained (DC) current, from the interval between the last two
 * spikes of a run (the first ~half is discarded as the onset transient). 0 when fewer than two spikes.
 */
export function steadyFiringRate(p: HHParams, durationMs = 300): number {
  const times = spikeTimes(simulate({ ...p, pulse_mode: 0 }, durationMs));
  if (times.length < 2) return 0;
  const interval = times[times.length - 1] - times[times.length - 2];
  return interval > 0 ? 1000 / interval : 0;
}

/** Spikes fired in the last complete stimulus cycle (pulse / twin modes), after a warm-up cycle. */
export function spikesPerCycle(p: HHParams): number {
  const cycles = 4;
  const times = spikeTimes(simulate(p, PULSE_PERIOD * cycles));
  const lastCycleStart = PULSE_PERIOD * (cycles - 1);
  return times.filter((t) => t >= lastCycleStart).length;
}

/** Steady firing rate (Hz) for each injected DC current in `currents` (other parameters as in `p`). */
export function firingRateCurve(p: HHParams, currents: number[], durationMs = 250): number[] {
  return currents.map((I_inj) => steadyFiringRate({ ...p, I_inj }, durationMs));
}

/** Lowest current in the curve that produces repetitive firing (the onset / rheobase), if any. */
export function rheobase(currents: number[], rates: number[]): number | undefined {
  const i = rates.findIndex((r) => r > 0);
  return i === -1 ? undefined : currents[i];
}

/**
 * Sample at horizontal position `fraction` (0 = oldest, 1 = newest) of the ring buffer the GPU
 * maintains: `history` holds `HISTORY_SAMPLES` samples of (V, m, h, n), and `head` is the index the
 * next sample will be written to — which is also where the oldest sample lives.
 */
export function sampleAt(history: Float32Array, head: number, fraction: number) {
  const k = Math.min(HISTORY_SAMPLES - 1, Math.max(0, Math.round(fraction * (HISTORY_SAMPLES - 1))));
  const i = ((head + k) % HISTORY_SAMPLES) * 4;
  return {
    V: history[i],
    m: history[i + 1],
    h: history[i + 2],
    n: history[i + 3],
    /** How long before "now" this sample was recorded (ms). */
    msAgo: (HISTORY_SAMPLES - 1 - k) * SAMPLE_EVERY * DT_MS,
  };
}

export interface FiAnalysis {
  currents: number[];
  rates: number[];
  /** Lowest sustained-firing current, refined by bisection to ~0.03 µA/cm²; undefined if none up to the sweep's end. */
  rheobase?: number;
}

/**
 * Firing-rate vs injected-current curve (0..`maxCurrent` in 1 µA/cm² steps, DC stimulation) and its
 * onset current. Depends only on conductances and temperature, not on the injected current itself.
 */
export function fiAnalysis(p: HHParams, maxCurrent = 30, durationMs = 250): FiAnalysis {
  const currents = Array.from({ length: maxCurrent + 1 }, (_, i) => i);
  const rates = firingRateCurve(p, currents, durationMs);
  const first = rates.findIndex((r) => r > 0);
  if (first === -1) return { currents, rates };
  let lo = first === 0 ? 0 : currents[first - 1];
  let hi = currents[first];
  for (let i = 0; i < 6; i++) {
    const mid = (lo + hi) / 2;
    if (steadyFiringRate({ ...p, I_inj: mid }, durationMs) > 0) hi = mid;
    else lo = mid;
  }
  return { currents, rates, rheobase: hi };
}
