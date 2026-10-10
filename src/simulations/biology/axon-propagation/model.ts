/**
 * What the simulation builds from its settings, shared by the simulation and the tests: the fibre(s), the
 * stimuli of a sweep, and how long a sweep lasts.
 */
import { bareFibre, blockSodium, demyelinate, INTERNODE_PER_DIAMETER, kick, lengthConstant, myelinatedFibre, NODE_LENGTH } from "./cable";
import type { Fibre, Stimulus } from "./cable";
import type { AxonSettings } from "./settings";
import { DRUG_BLOCK, effectiveDiameter } from "./settings";

/** Bare axons are 8 length constants long; myelinated ones 30 internodes. */
export const BARE_LENGTHS = 8;
export const BARE_COMPARTMENTS = 400;
export const INTERNODES = 30;
/** The first stimulus of every sweep, ms. */
export const FIRST_PULSE = 0.5;

/** Sweep length (ms): the spike crosses a bare axon in ~4.6 ms and a myelinated one in ~1.2 ms at 18 °C, whatever its size. */
export function sweepMs(s: Pick<AxonSettings, "myelin" | "temp" | "pulses" | "gap">): number {
  // Colder is slower (the gates' Q10 of 3); leave room for it and for the second pulse of a pair.
  const cold = Math.max(1, Math.pow(3, (18.3 - s.temp) / 20));
  const transit = (s.myelin ? 1.3 : 4.8) * cold;
  return Math.ceil((FIRST_PULSE + transit * 1.6 + (s.pulses === "pair" ? s.gap : 0) + 1) * 2) / 2;
}

/** The axon of the Axon view, with its treated stretch applied. */
export function axonFibre(s: AxonSettings): Fibre {
  const d = effectiveDiameter(s);
  const f = s.myelin ? myelinatedFibre(d, INTERNODES * (INTERNODE_PER_DIAMETER * d * 1e-4 + NODE_LENGTH) - 1e-6) : bareFibre(d, BARE_LENGTHS * lengthConstant(d), BARE_COMPARTMENTS);
  const L = fibreLength(f);
  const [a, b] = [Math.min(s.from, s.to) * L, Math.max(s.from, s.to) * L];
  if (DRUG_BLOCK[s.drug] > 0) blockSodium(f, a, b, DRUG_BLOCK[s.drug]);
  if (s.myelin && s.myelinLeft < 1) demyelinate(f, a, b, s.myelinLeft);
  return f;
}

export const fibreLength = (f: Fibre) => f.x[f.n - 1] + f.length[f.n - 1] / 2;

/** Nearest compartment to a fraction of the fibre's length (for electrodes); active ones for myelinated fibres. */
export function compartmentAt(f: Fibre, fraction: number): number {
  const x = fraction * fibreLength(f);
  let best = 0;
  for (let i = 0; i < f.n; i++) if (f.active[i] && Math.abs(f.x[i] - x) < Math.abs(f.x[best] - x)) best = i;
  return best;
}

export interface Sweep {
  /** Stimulating electrode A (near end) and B (far end, or the second pulse of a pair at A). */
  a: Stimulus;
  b: Stimulus | null;
  ms: number;
}

/** The stimuli of one sweep. */
export function sweepFor(s: AxonSettings, f: Fibre): Sweep {
  const a = kick(f, FIRST_PULSE, s.stim);
  let b: Stimulus | null = null;
  if (s.pulses === "pair") b = { ...a, startMs: FIRST_PULSE + s.gap };
  if (s.pulses === "both") {
    // The same pulse at the far end: start where its stretch ends exactly at the last compartment.
    const far = kick(f, FIRST_PULSE, s.stim, 0);
    const k = far.last - far.first;
    b = { ...far, first: f.n - 1 - k, last: f.n - 1 };
  }
  return { a, b, ms: sweepMs(s) };
}

// ── The nerve: a bundle of fibres of many sizes, like a frog sciatic nerve ──

/** Nerve length, cm. */
export const NERVE_LENGTH = 2;
/** Recording electrode distance from the fibres, cm. */
export const NERVE_HEIGHT = 0.05;
export const NERVE_SWEEP_MS = 40;

/** A small, fast deterministic generator, so a nerve (and its link) is the same every time. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type FibreClass = "Aα/β" | "Aδ" | "C";

export interface NerveFibre {
  diameter: number;
  myelinated: boolean;
  kind: FibreClass;
}

/**
 * Fibre sizes of the nerve: 30 % large myelinated A fibres (Aα/β, 6–14 µm), 20 % thin myelinated Aδ (1.5–4 µm)
 * and 50 % unmyelinated C fibres (0.4–1.2 µm). Sorted by diameter, largest first.
 */
export function nerveComposition(count: number, seed = 7): NerveFibre[] {
  const random = rng(seed);
  const out: NerveFibre[] = [];
  for (let k = 0; k < count; k++) {
    const u = k / count;
    const r = random();
    if (u < 0.3) out.push({ diameter: 6 + 8 * r, myelinated: true, kind: "Aα/β" });
    else if (u < 0.5) out.push({ diameter: 1.5 + 2.5 * r, myelinated: true, kind: "Aδ" });
    else out.push({ diameter: 0.4 + 0.8 * r, myelinated: false, kind: "C" });
  }
  return out.sort((x, y) => y.diameter - x.diameter);
}

/** Internode segments in the nerve view (fewer than the Axon view's 10, to fit a thousand fibres). */
export const NERVE_SEGMENTS = 4;

export function nerveFibre(spec: NerveFibre): Fibre {
  if (spec.myelinated) return myelinatedFibre(spec.diameter, NERVE_LENGTH, NERVE_SEGMENTS);
  // Compartments of an eighth of a length constant.
  const n = Math.ceil(NERVE_LENGTH / (lengthConstant(spec.diameter) / 8));
  return bareFibre(spec.diameter, NERVE_LENGTH, n);
}

/**
 * An external electrode excites big fibres more easily than thin ones; modelled as a stimulus (in multiples of
 * each fibre's threshold) growing with √(diameter / 10 µm). Weak shocks recruit only the largest fibres.
 */
export const recruitment = (strength: number, diameter: number) => strength * Math.sqrt(diameter / 10);

export function nerveStimulus(f: Fibre, strength: number): Stimulus {
  return kick(f, FIRST_PULSE, recruitment(strength, f.diameter));
}
