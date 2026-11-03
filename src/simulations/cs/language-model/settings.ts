import { enumField, numberField } from "@/lib/urlState";
import type { FieldCodec, Schema } from "@/lib/urlState";
import { MODEL_KINDS } from "./model";
import type { ModelKind, ModelSpec } from "./model";
import { workMegabytes } from "./plan";
import type { LmConfig, LmView } from "./sim";
import { TEXT_IDS } from "./text";
import type { TextId } from "./text";

export const VIEWS: readonly LmView[] = ["train", "predict", "tables", "shannon"];
export const CONTEXTS = [16, 32, 64, 128] as const;
export const WIDTHS = [32, 64, 128] as const;
export const HEADS = [1, 2, 4] as const;
export const BATCHES = [8, 16, 32] as const;
export const HIDDEN = [64, 128, 256] as const;
export const SPEEDS = [0, 1, 2, 4, 8, 16, 32] as const;
/** Widest attention head the kernels hold. */
export const MAX_HEAD_WIDTH = 64;
/** Working memory a model may use (a storage binding holds 128 MB). */
export const MEMORY_LIMIT_MB = 110;

/** Everything a shared link reproduces (your own text stays in the browser). */
export interface LmSettings {
  view: LmView;
  text: TextId;
  model: ModelKind;
  /** Letters per window: the transformer's context. */
  ctx: number;
  /** MLP: letters of context and hidden width. */
  k: number;
  hidden: number;
  /** Transformer. */
  layers: number;
  heads: number;
  width: number;
  batch: number;
  lr: number;
  seed: number;
  /** Steps per frame, 0 = automatic. */
  speed: number;
  /** Writing temperature. */
  temp: number;
  /** Count tables: letters of context shown. */
  tk: number;
}

/** A good learning rate for each model (Adam): what switching model sets. */
export const LR_FOR: Record<ModelKind, number> = { bigram: 0.05, mlp: 0.003, gpt: 0.003 };

export const LM_DEFAULTS: LmSettings = {
  view: "train",
  text: "shakespeare",
  model: "gpt",
  ctx: 64,
  k: 8,
  hidden: 128,
  layers: 2,
  heads: 4,
  width: 64,
  batch: 16,
  lr: LR_FOR.gpt,
  seed: 1,
  speed: 0,
  temp: 1,
  tk: 4,
};

const intField = (min: number, max: number): FieldCodec<number> => {
  const base = numberField(min, max);
  return { decode: (raw) => (base.decode(raw) === undefined ? undefined : Math.round(base.decode(raw)!)), encode: (v) => String(Math.round(v)) };
};
const rateField = (min: number, max: number): FieldCodec<number> => {
  const base = numberField(min, max);
  return { decode: base.decode, encode: (v) => String(Number(v.toPrecision(3))) };
};
const choiceField = (values: readonly number[]): FieldCodec<number> => ({ decode: (raw) => values.find((v) => String(v) === raw), encode: String });

export const LM_SCHEMA: Schema<LmSettings> = {
  view: enumField(VIEWS),
  text: enumField(TEXT_IDS),
  model: enumField(MODEL_KINDS),
  ctx: choiceField(CONTEXTS),
  k: intField(1, 16),
  hidden: choiceField(HIDDEN),
  layers: intField(1, 4),
  heads: choiceField(HEADS),
  width: choiceField(WIDTHS),
  batch: choiceField(BATCHES),
  lr: rateField(1e-4, 1),
  seed: intField(1, 999999),
  speed: choiceField(SPEEDS),
  temp: numberField(0, 2),
  tk: intField(0, 6),
};

/** The model a set of settings describes (heads raised if a head would be wider than the kernels hold). */
export function specOf(s: LmSettings): Omit<ModelSpec, "V"> {
  const heads = Math.max(s.heads, Math.ceil(s.width / MAX_HEAD_WIDTH));
  return { kind: s.model, T: s.ctx, k: s.k, e: 16, h: s.hidden, L: s.layers, H: heads, d: s.width };
}

/** The batch that fits in memory (the chosen one, or smaller for a large model with a long context). */
export function fittedBatch(s: LmSettings): number {
  const spec = { ...specOf(s), V: 96 };
  let b = s.batch;
  while (b > 1 && workMegabytes(spec, b) > MEMORY_LIMIT_MB) b = Math.floor(b / 2);
  return b;
}

/** What the simulation needs: the settings, plus your own text (which is not in a link). */
export function lmConfigOf(s: LmSettings, ownText = ""): LmConfig {
  return { view: s.view, text: s.text, ownText, spec: specOf(s), batch: fittedBatch(s), lr: s.lr, seed: s.seed, speed: s.speed, temperature: s.temp, tableK: s.tk };
}
