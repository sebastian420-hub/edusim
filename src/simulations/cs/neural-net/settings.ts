import { enumField, numberField } from "@/lib/urlState";
import type { FieldCodec, Schema } from "@/lib/urlState";
import { FEATURES, featureCount, PLANE_SETS } from "./net";
import type { Activation, Init, NetSpec, PlaneSet } from "./net";
import { DIGIT_PIXELS } from "./digits";
import type { NetConfig } from "./sim";

export type NetView = "playground" | "digits";
export const ACTIVATIONS: readonly Activation[] = ["relu", "tanh", "sigmoid", "linear"];
export const DIGIT_NETS = ["linear", "h128", "h128-64"] as const;
export type DigitNet = (typeof DIGIT_NETS)[number];
export const BATCH_SIZES = [1, 4, 16, 64, 256, 2000] as const;
export const DIGIT_BATCHES = [8, 32, 128] as const;
/** Hidden layers on the plane: up to 6 of up to 32 neurons. */
export const MAX_HIDDEN_LAYERS = 6;
export const MAX_NEURONS = 32;

/** Everything a shared link reproduces. */
export interface NetSettings {
  view: NetView;
  // Playground: the data.
  data: PlaneSet;
  points: number;
  /** Fraction of labels flipped at random. */
  noise: number;
  /** Fraction of the points used for training (the rest are the test set). */
  train: number;
  /** Input features, a bit per FEATURES entry. */
  features: number;
  // Playground: the network and its training.
  layers: number[];
  act: Activation;
  lr: number;
  /** Batch size (2000 = the whole training set: plain gradient descent). */
  batch: number;
  mom: number;
  l2: number;
  init: Init;
  seed: number;
  /** Training steps per frame. */
  speed: number;
  // Digits.
  dnet: DigitNet;
  dlr: number;
  dbatch: number;
}

export const NET_DEFAULTS: NetSettings = {
  view: "playground",
  data: "spiral",
  points: 400,
  noise: 0,
  train: 0.5,
  features: 0b11,
  layers: [16, 16],
  act: "relu",
  lr: 0.03,
  batch: 16,
  mom: 0.9,
  l2: 0,
  init: "random",
  seed: 1,
  speed: 5,
  dnet: "h128",
  dlr: 0.05,
  dbatch: 32,
};

const intField = (min: number, max: number): FieldCodec<number> => {
  const base = numberField(min, max);
  return { decode: (raw) => (base.decode(raw) === undefined ? undefined : Math.round(base.decode(raw)!)), encode: (v) => String(Math.round(v)) };
};

/** Learning rates span decades: three significant digits. */
const rateField = (min: number, max: number): FieldCodec<number> => {
  const base = numberField(min, max);
  return { decode: base.decode, encode: (v) => String(Number(v.toPrecision(3))) };
};

/** Hidden layer widths as "16-16", or "none". Invalid entries make the whole field invalid. */
export const layersField: FieldCodec<number[]> = {
  decode(raw) {
    if (raw === "none") return [];
    const parts = raw.split("-");
    if (parts.length > MAX_HIDDEN_LAYERS || parts.some((p) => !/^\d+$/.test(p))) return undefined;
    return parts.map((p) => Math.min(MAX_NEURONS, Math.max(1, Number(p))));
  },
  encode: (v) => (v.length ? v.join("-") : "none"),
};

const featuresField: FieldCodec<number> = {
  decode(raw) {
    const n = Number(raw);
    if (!/^\d+$/.test(raw) || n < 1 || n >= 1 << FEATURES.length) return undefined;
    return n;
  },
  encode: String,
};

const choiceField = (values: readonly number[]): FieldCodec<number> => ({ decode: (raw) => values.find((v) => String(v) === raw), encode: String });

export const NET_SCHEMA: Schema<NetSettings> = {
  view: enumField(["playground", "digits"] as const),
  data: enumField(PLANE_SETS),
  points: intField(100, 2000),
  noise: numberField(0, 0.4),
  train: numberField(0.1, 0.9),
  features: featuresField,
  layers: layersField,
  act: enumField(ACTIVATIONS),
  lr: rateField(1e-4, 30),
  batch: choiceField(BATCH_SIZES),
  mom: numberField(0, 0.99),
  l2: rateField(0, 0.1),
  init: enumField(["random", "zeros"] as const),
  seed: intField(1, 999999),
  speed: intField(1, 50),
  dnet: enumField(DIGIT_NETS),
  dlr: rateField(1e-4, 3),
  dbatch: choiceField(DIGIT_BATCHES),
};

export const DIGIT_HIDDEN: Record<DigitNet, number[]> = { linear: [], h128: [128], "h128-64": [128, 64] };

/** The plane network the settings describe. */
export const planeSpec = (s: Pick<NetSettings, "features" | "layers" | "act">): NetSpec => ({ inputs: featureCount(s.features), hidden: s.layers, outputs: 1, activation: s.act });
/** The digits network: always ReLU (the modern default for images). */
export const digitSpec = (s: Pick<NetSettings, "dnet">): NetSpec => ({ inputs: DIGIT_PIXELS, hidden: DIGIT_HIDDEN[s.dnet], outputs: 10, activation: "relu" });

/** What the simulation needs from the settings (plus whether test points are drawn, which is not in a link). */
export function netConfigOf(s: NetSettings, showTest = true): NetConfig {
  return {
    view: s.view,
    plane: { data: s.data, points: s.points, noise: s.noise, train: s.train, features: s.features, spec: planeSpec(s), init: s.init, seed: s.seed },
    opt: { lr: s.lr, momentum: s.mom, l2: s.l2, batch: s.batch },
    digits: { spec: digitSpec(s), opt: { lr: s.dlr, momentum: 0.9, l2: 0, batch: s.dbatch }, seed: s.seed },
    speed: s.speed,
    showTest,
  };
}
