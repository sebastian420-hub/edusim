import { boolField, enumField, numberField } from "@/lib/urlState";
import type { FieldCodec, Schema } from "@/lib/urlState";

export type AxonView = "axon" | "nerve";
export type Pulses = "single" | "pair" | "both";
export type Drug = "none" | "ttx" | "lidocaine";
export const NERVE_COUNTS = [100, 300, 1000] as const;
export type NerveCount = (typeof NERVE_COUNTS)[number];

/** Myelinated fibres in nature are at most about 20 µm thick. */
export const MAX_MYELINATED = 20;

/** Everything a shared link reproduces. Positions along the axon are fractions of its length (0 = stimulus end). */
export interface AxonSettings {
  view: AxonView;
  /** Fibre diameter, µm (476 = the squid giant axon). */
  diameter: number;
  myelin: boolean;
  /** °C. */
  temp: number;
  /** Stimulus strength in multiples of the threshold. */
  stim: number;
  pulses: Pulses;
  /** Interval between the two pulses of a pair, ms. */
  gap: number;
  /** Recording electrodes. */
  e1: number;
  e2: number;
  /** The treated stretch, and what is applied to it. */
  from: number;
  to: number;
  drug: Drug;
  /** Myelin left in the treated stretch (1 = healthy), for myelinated fibres. */
  myelinLeft: number;
  /** Slow motion: simulated ms per real second. */
  speed: number;
  /** Nerve view. */
  fibres: NerveCount;
  nerveStim: number;
  /** Recording electrode distance from the stimulus, cm. */
  distance: number;
}

export const AXON_DEFAULTS: AxonSettings = {
  view: "axon",
  diameter: 476,
  myelin: false,
  temp: 18.3,
  stim: 2,
  pulses: "single",
  gap: 6,
  e1: 0.3,
  e2: 0.7,
  from: 0.45,
  to: 0.55,
  drug: "none",
  myelinLeft: 1,
  speed: 2,
  fibres: 300,
  nerveStim: 6,
  distance: 1.5,
};

const countField: FieldCodec<NerveCount> = { decode: (raw) => NERVE_COUNTS.find((c) => String(c) === raw), encode: String };

export const AXON_SCHEMA: Schema<AxonSettings> = {
  view: enumField(["axon", "nerve"] as const),
  diameter: numberField(0.5, 1000),
  myelin: boolField,
  temp: numberField(0, 40),
  stim: numberField(0.2, 6),
  pulses: enumField(["single", "pair", "both"] as const),
  gap: numberField(0.5, 20),
  e1: numberField(0.02, 0.98),
  e2: numberField(0.02, 0.98),
  from: numberField(0, 1),
  to: numberField(0, 1),
  drug: enumField(["none", "ttx", "lidocaine"] as const),
  myelinLeft: numberField(0, 1),
  speed: numberField(0.2, 20),
  fibres: countField,
  nerveStim: numberField(0.2, 12),
  distance: numberField(0.3, 1.9),
};

/** The diameter actually used: myelinated fibres are capped at 20 µm. */
export const effectiveDiameter = (s: Pick<AxonSettings, "diameter" | "myelin">) => (s.myelin ? Math.min(s.diameter, MAX_MYELINATED) : s.diameter);

/** Fraction of sodium channels each drug blocks. */
export const DRUG_BLOCK: Record<Drug, number> = { none: 0, ttx: 1, lidocaine: 0.9 };
