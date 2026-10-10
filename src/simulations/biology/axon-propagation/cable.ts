/**
 * CPU twin of the axon shaders: Hodgkin–Huxley membranes joined into a cable. Every fibre is a chain of
 * compartments (uniform for bare axons; nodes of Ranvier and myelinated internodes for myelinated ones).
 *
 * One step (Δt) per fibre:
 *  1. the m, h, n gates relax towards their steady state at the old voltage (Rush–Larsen, as the HH simulation);
 *  2. the voltages are solved *implicitly* from the current balance of every compartment,
 *       (Cᵢ/Δt + gᵢ + Gᵢ₋₁ + Gᵢ)·Vᵢ' − Gᵢ₋₁·Vᵢ₋₁' − Gᵢ·Vᵢ₊₁' = Cᵢ/Δt·Vᵢ + eᵢ + Iᵢ,
 *     where gᵢ is the compartment's total ionic conductance, eᵢ = Σ g·E its driving term, Gᵢ the axial
 *     conductance to the next compartment and Iᵢ the stimulus. A tridiagonal system, solved exactly by the Thomas
 *     algorithm: stable for any Δt (an explicit step would need Δt < 0.004 ms for the squid axon at 0.5 mm).
 *
 * Units: mV, ms, cm; conductances in mS, capacitances in µF, currents in µA (mS·mV = µA, µF·mV/ms = µA).
 * The GPU kernel (cable.wgsl) does the same arithmetic in 32-bit; keep the two in sync.
 */
import { DEFAULT_PARAMS, rates, temperatureFactor } from "../hodgkin-huxley/hh";

/** Axial resistivity of squid axoplasm, Ω·cm (Hodgkin & Huxley 1952). */
export const RI = 35.4;
/** Squid giant axon diameter used by Hodgkin & Huxley for the propagated action potential, µm. */
export const SQUID_DIAMETER = 476;
/** Their computed conduction velocity at 18.3 °C (measured: 21.2 m/s). */
export const HH_VELOCITY = 18.8;
export const HH_TEMPERATURE = 18.3;
export const REST_V = -65;
/** Integration step, ms. */
export const DT = 0.01;

/** Membrane of a bare axon and of a node of Ranvier (channel densities in mS/cm², capacitance µF/cm²). */
export interface Membrane {
  cm: number;
  gNa: number;
  gK: number;
  gL: number;
}

export const BARE: Membrane = { cm: 1, gNa: DEFAULT_PARAMS.g_Na, gK: DEFAULT_PARAMS.g_K, gL: DEFAULT_PARAMS.g_L };
/** Nodes of Ranvier pack sodium channels far more densely than bare membrane. */
export const NODE: Membrane = { cm: 1, gNa: 10 * BARE.gNa, gK: 10 * BARE.gK, gL: 10 * BARE.gL };
/**
 * A myelinated internode of a 10 µm fibre: ~50 wraps of membrane in series divide capacitance and leak about
 * 100-fold. Myelin thickness grows with the fibre (constant g-ratio, as Rushton assumed), so the wraps, and the
 * division, scale with the diameter: see `myelinFor`.
 */
export const MYELIN: Membrane = { cm: 0.01, gNa: 0, gK: 0, gL: 0.003 };
export const REFERENCE_DIAMETER = 10;
/** Internode length relative to the fibre diameter. */
export const INTERNODE_PER_DIAMETER = 100;
/** Node of Ranvier length, cm: about 1 µm whatever the fibre size. */
export const NODE_LENGTH = 1e-4;

/** Internode membrane of a fibre of `diameter` µm: thicker myelin (more wraps) on thicker fibres. */
export const myelinFor = (diameter: number): Membrane => {
  const k = REFERENCE_DIAMETER / diameter;
  return { cm: MYELIN.cm * k, gNa: 0, gK: 0, gL: MYELIN.gL * k };
};
/** Compartments per internode. */
export const INTERNODE_SEGMENTS = 10;

export const { E_Na, E_K, E_L } = DEFAULT_PARAMS;

/** One fibre's compartments, as flat arrays (index = compartment along the fibre). */
export interface Fibre {
  diameter: number; // µm
  myelinated: boolean;
  n: number;
  /** Centre of each compartment along the fibre, cm. */
  x: Float64Array;
  length: Float64Array;
  /** Membrane area, cm². */
  area: Float64Array;
  cm: Float64Array;
  gNa: Float64Array;
  gK: Float64Array;
  gL: Float64Array;
  /** Axial conductance from compartment i to i + 1, mS (0 for the last). */
  axial: Float64Array;
  /** Node of Ranvier (or bare membrane) = 1, internode = 0: where spikes are regenerated. */
  active: Uint8Array;
}

/** Length constant at rest of a bare fibre, cm: λ = √(a·Rm / 2Rᵢ). */
export function lengthConstant(diameterUm: number, m: Membrane = BARE): number {
  const a = (diameterUm * 1e-4) / 2;
  const rm = 1000 / m.gL; // Ω·cm²
  return Math.sqrt((a * rm) / (2 * RI));
}

function build(diameter: number, myelinated: boolean, lengths: number[], membranes: Membrane[], active: number[]): Fibre {
  const n = lengths.length;
  const a = (diameter * 1e-4) / 2;
  const f: Fibre = {
    diameter,
    myelinated,
    n,
    x: new Float64Array(n),
    length: Float64Array.from(lengths),
    area: new Float64Array(n),
    cm: new Float64Array(n),
    gNa: new Float64Array(n),
    gK: new Float64Array(n),
    gL: new Float64Array(n),
    axial: new Float64Array(n),
    active: Uint8Array.from(active),
  };
  let at = 0;
  for (let i = 0; i < n; i++) {
    f.x[i] = at + lengths[i] / 2;
    at += lengths[i];
    f.area[i] = 2 * Math.PI * a * lengths[i];
    f.cm[i] = membranes[i].cm;
    f.gNa[i] = membranes[i].gNa;
    f.gK[i] = membranes[i].gK;
    f.gL[i] = membranes[i].gL;
  }
  // Between compartment centres: R = Rᵢ·distance / (π a²) Ω → G = 1000 / R mS.
  for (let i = 0; i < n - 1; i++) f.axial[i] = (1000 * Math.PI * a * a) / (RI * (lengths[i] / 2 + lengths[i + 1] / 2));
  return f;
}

/** A bare (unmyelinated) axon of `length` cm in `n` equal compartments. */
export function bareFibre(diameter: number, length: number, n: number): Fibre {
  return build(diameter, false, Array(n).fill(length / n), Array(n).fill(BARE), Array(n).fill(1));
}

/**
 * A myelinated fibre at least `length` cm long: node, internode, node, … ending on a node. Internode length and
 * myelin thickness grow with the diameter while nodes stay ~1 µm (Rushton's similarity): then every internode
 * charges in the same time, and the speed is proportional to the diameter.
 */
export function myelinatedFibre(diameter: number, length: number, segments = INTERNODE_SEGMENTS): Fibre {
  const internode = INTERNODE_PER_DIAMETER * diameter * 1e-4;
  const MYELIN = myelinFor(diameter);
  const lengths: number[] = [NODE_LENGTH];
  const membranes: Membrane[] = [NODE];
  const active: number[] = [1];
  let at = NODE_LENGTH;
  while (at < length) {
    for (let k = 0; k < segments; k++) {
      lengths.push(internode / segments);
      membranes.push(MYELIN);
      active.push(0);
    }
    lengths.push(NODE_LENGTH);
    membranes.push(NODE);
    active.push(1);
    at += internode + NODE_LENGTH;
  }
  return build(diameter, true, lengths, membranes, active);
}

/** Sodium block on [x0, x1] (cm): `fraction` of the channels (1 = TTX, 0.9 ≈ lidocaine). */
export function blockSodium(f: Fibre, x0: number, x1: number, fraction: number): void {
  for (let i = 0; i < f.n; i++) if (f.x[i] >= x0 && f.x[i] <= x1) f.gNa[i] *= 1 - fraction;
}

/**
 * Demyelination of the internodes whose centre lies in [x0, x1]: `remaining` = fraction of the myelin left
 * (1 = healthy, 0 = bare membrane). Capacitance and leak scale with the inverse of the myelin thickness.
 */
export function demyelinate(f: Fibre, x0: number, x1: number, remaining: number): void {
  const r = Math.max(remaining, 1e-3);
  const healthy = myelinFor(f.diameter);
  for (let i = 0; i < f.n; i++) {
    if (f.active[i] || f.x[i] < x0 || f.x[i] > x1) continue;
    f.cm[i] = Math.min(BARE.cm, healthy.cm / r);
    f.gL[i] = Math.min(BARE.gL, healthy.gL / r);
  }
}

/** Gate values at rest (steady state at V). */
export function restingGates(V = REST_V): { m: number; h: number; n: number } {
  const r = rates(V);
  return { m: r.am / (r.am + r.bm), h: r.ah / (r.ah + r.bh), n: r.an / (r.an + r.bn) };
}

export interface CableState {
  V: Float64Array;
  m: Float64Array;
  h: Float64Array;
  n: Float64Array;
}

export function restState(f: Fibre): CableState {
  const g = restingGates();
  return { V: new Float64Array(f.n).fill(REST_V), m: new Float64Array(f.n).fill(g.m), h: new Float64Array(f.n).fill(g.h), n: new Float64Array(f.n).fill(g.n) };
}

const relax = (x: number, a: number, b: number, dt: number) => {
  const inf = a / (a + b);
  return inf + (x - inf) * Math.exp(-dt * (a + b));
};

/** Scratch space for the tridiagonal solve, reused between steps. */
export interface Scratch {
  c: Float64Array;
  d: Float64Array;
}
export const scratchFor = (n: number): Scratch => ({ c: new Float64Array(n), d: new Float64Array(n) });

/**
 * Advances a fibre by one step. `stim[i]` is the current injected into compartment i (µA) during this step.
 * Returns nothing; `s` is updated in place. `membraneCurrent`, if given, receives each compartment's total
 * transmembrane current (µA, outward positive) for the extracellular recording.
 */
export function stepCable(f: Fibre, s: CableState, celsius: number, stim: Float64Array | null, scratch: Scratch, dt = DT, membraneCurrent?: Float64Array): void {
  const phi = temperatureFactor(celsius);
  const n = f.n;
  const { c, d } = scratch;
  // Thomas forward sweep, assembling each row as we go: lower = −Gᵢ₋₁, diag, upper = −Gᵢ.
  let prevC = 0;
  let prevD = 0;
  for (let i = 0; i < n; i++) {
    const V = s.V[i];
    const r = rates(V);
    const m = relax(s.m[i], r.am * phi, r.bm * phi, dt);
    const h = relax(s.h[i], r.ah * phi, r.bh * phi, dt);
    const nn = relax(s.n[i], r.an * phi, r.bn * phi, dt);
    s.m[i] = m;
    s.h[i] = h;
    s.n[i] = nn;
    const A = f.area[i];
    const gNa = f.gNa[i] * m * m * m * h * A;
    const gK = f.gK[i] * nn * nn * nn * nn * A;
    const gL = f.gL[i] * A;
    const C = (f.cm[i] * A) / dt;
    const lower = i > 0 ? f.axial[i - 1] : 0;
    const upper = f.axial[i];
    const diag = C + gNa + gK + gL + lower + upper;
    const rhs = C * V + gNa * E_Na + gK * E_K + gL * E_L + (stim ? stim[i] : 0);
    const denom = diag + lower * prevC; // row: −Gᵢ₋₁·Vᵢ₋₁ + diag·Vᵢ − Gᵢ·Vᵢ₊₁, so b − a·c′ = diag + Gᵢ₋₁·c′
    c[i] = -upper / denom;
    d[i] = (rhs + lower * prevD) / denom;
    prevC = c[i];
    prevD = d[i];
  }
  // Back substitution.
  let next = d[n - 1];
  const old = membraneCurrent ? Float64Array.from(s.V) : null;
  s.V[n - 1] = next;
  for (let i = n - 2; i >= 0; i--) {
    next = d[i] - c[i] * next;
    s.V[i] = next;
  }
  if (membraneCurrent && old) {
    // Net axial inflow = transmembrane current (Kirchhoff), minus any injected stimulus.
    for (let i = 0; i < n; i++) {
      const left = i > 0 ? f.axial[i - 1] * (s.V[i - 1] - s.V[i]) : 0;
      const right = i < n - 1 ? f.axial[i] * (s.V[i + 1] - s.V[i]) : 0;
      membraneCurrent[i] = left + right + (stim ? stim[i] : 0);
    }
  }
}

/** Stimulus: `amplitude` µA into compartments [first, last] for `durationMs` from time `startMs`. */
export interface Stimulus {
  first: number;
  last: number;
  startMs: number;
  durationMs: number;
  amplitude: number;
}

/** Fills `out` with the stimulus currents active at time `t` (start of the step). */
export function stimulusAt(stimuli: readonly Stimulus[], t: number, out: Float64Array): boolean {
  out.fill(0);
  let any = false;
  for (const st of stimuli) {
    if (t + 1e-9 < st.startMs || t + 1e-9 >= st.startMs + st.durationMs) continue;
    const k = st.last - st.first + 1;
    for (let i = st.first; i <= st.last; i++) out[i] += st.amplitude / k;
    any = true;
  }
  return any;
}

/** Threshold for "a spike is here": arrival times are when V first rises through it. */
export const SPIKE_THRESHOLD = -20;

export interface RunResult {
  /** First upward crossing of SPIKE_THRESHOLD per compartment (ms), NaN if never. */
  arrival: Float64Array;
  /** Number of upward crossings per compartment. */
  crossings: Uint16Array;
  /** Peak voltage per compartment. */
  peak: Float64Array;
  state: CableState;
}

/** Runs a fibre from rest for `durationMs`, recording arrival times, crossings and peaks. */
export function runFibre(f: Fibre, celsius: number, stimuli: readonly Stimulus[], durationMs: number, dt = DT, onStep?: (t: number, s: CableState) => void): RunResult {
  const s = restState(f);
  const scratch = scratchFor(f.n);
  const stim = new Float64Array(f.n);
  const arrival = new Float64Array(f.n).fill(NaN);
  const crossings = new Uint16Array(f.n);
  const peak = new Float64Array(f.n).fill(REST_V);
  const steps = Math.round(durationMs / dt);
  for (let k = 0; k < steps; k++) {
    const t = k * dt;
    const before = Float64Array.from(s.V);
    stepCable(f, s, celsius, stimulusAt(stimuli, t, stim) ? stim : null, scratch, dt);
    for (let i = 0; i < f.n; i++) {
      if (before[i] < SPIKE_THRESHOLD && s.V[i] >= SPIKE_THRESHOLD) {
        // Interpolate the crossing inside the step.
        const at = t + dt * ((SPIKE_THRESHOLD - before[i]) / (s.V[i] - before[i]));
        if (Number.isNaN(arrival[i])) arrival[i] = at;
        crossings[i]++;
      }
      if (s.V[i] > peak[i]) peak[i] = s.V[i];
    }
    onStep?.(t + dt, s);
  }
  return { arrival, crossings, peak, state: s };
}

/**
 * Conduction velocity (m/s) from arrival times: the least-squares slope of x against arrival time over the
 * active compartments (nodes, for a myelinated fibre) between fractions `from` and `to` of the fibre, away from
 * the stimulus and the sealed end. Undefined if the spike did not get through that stretch.
 */
export function conductionVelocity(f: Fibre, arrival: Float64Array, from = 0.3, to = 0.7): number | undefined {
  const total = f.x[f.n - 1];
  const pts: [number, number][] = [];
  for (let i = 0; i < f.n; i++) {
    if (!f.active[i] || f.x[i] < from * total || f.x[i] > to * total) continue;
    if (Number.isNaN(arrival[i])) return undefined;
    pts.push([arrival[i], f.x[i]]);
  }
  if (pts.length < 2) return undefined;
  const mt = pts.reduce((a, p) => a + p[0], 0) / pts.length;
  const mx = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  let num = 0;
  let den = 0;
  for (const [t, x] of pts) {
    num += (t - mt) * (x - mx);
    den += (t - mt) * (t - mt);
  }
  return den > 0 ? (num / den) * 10 : undefined; // cm/ms → m/s
}

/**
 * A brief (0.2 ms) current pulse into the stretch starting at compartment `at`, scaled to the fibre so that
 * `factor` is the stimulus in multiples of its threshold (1 = just at threshold, at 18.3 °C).
 */
export function kick(f: Fibre, startMs = 0.5, factor = 2, at = 0): Stimulus {
  // Charge the first stretch (about a twentieth of the fibre, at least one active compartment) by ~40 mV in 0.2 ms.
  let last = at;
  const total = f.x[f.n - 1];
  while (last < f.n - 1 && (f.x[last] - f.x[at] < total / 40 || !f.active[last])) last++;
  let capacitance = 0;
  for (let i = at; i <= last; i++) capacitance += f.cm[i] * f.area[i];
  // Measured thresholds at 18.3 °C (cable.test.ts): 0.55 of this charge for bare fibres, 1.19 for myelinated ones.
  const threshold = f.myelinated ? 1.19 : 0.55;
  const amplitude = factor * threshold * ((capacitance * 40) / 0.2); // µF·mV/ms = µA
  return { first: at, last, startMs, durationMs: 0.2, amplitude };
}

/**
 * A fibre's contribution to the potential at an extracellular electrode at `electrode` cm along the nerve and
 * `height` cm away (line-source approximation, up to the constant 1/4πσ): Σ Iₘ / distance, with each
 * compartment's membrane current from Kirchhoff (net axial inflow plus any injected stimulus). As cable.wgsl.
 */
export function extracellular(f: Fibre, V: Float64Array, stim: Float64Array | null, electrode: number, height: number): number {
  let sum = 0;
  for (let i = 0; i < f.n; i++) {
    let im = stim ? stim[i] : 0;
    if (i > 0) im += f.axial[i - 1] * (V[i - 1] - V[i]);
    if (i < f.n - 1) im += f.axial[i] * (V[i + 1] - V[i]);
    const dx = f.x[i] - electrode;
    sum += im / Math.sqrt(dx * dx + height * height);
  }
  return sum;
}
