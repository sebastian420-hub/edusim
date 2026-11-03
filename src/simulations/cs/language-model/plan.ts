/**
 * The memory plan of a model on a text: where everything lives in the four buffers and the programs that run over
 * them. Shared by the GPU engine (engine.ts) and the CPU twin (twin.ts), so both lay out memory identically.
 *
 * tokens: [text][writing lines: margin, context, letters written][prompt][window starts][writing counter]
 */
import { CONTEXT_MARGIN } from "./text";
import { forwardProgram, layout, optLayout, trainProgram, writeProgram } from "./model";
import type { Layout, ModelSpec, Program } from "./model";

/** Largest buffer a WebGPU device must allow binding (128 MiB). */
export const MAX_BINDING_BYTES = 134_217_728;

export interface PlanOptions {
  /** Training batch (windows per step); evaluation runs in chunks of this many windows. */
  B: number;
  /** Lines written at once, and letters per line. */
  G: number;
  W: number;
  /** Windows evaluated by evaluate(). */
  E: number;
}

export interface Plan extends PlanOptions {
  spec: ModelSpec;
  lay: Layout;
  nText: number;
  tokens: {
    size: number;
    write: number;
    /** Tokens per writing line: margin, then T of context, then W written. */
    lineStride: number;
    prompt: number;
    startsTrain: number;
    startsEval: number;
    startsWrite: number;
    startsPredict: number;
    counter: number;
  };
  /** Evaluation losses (E·T), then the surprise of each written letter (G·W). */
  work: { size: number; losses: number; surprise: number };
  opt: ReturnType<typeof optLayout>;
  train: Program;
  /** Evaluation: chunks of C windows (C ≥ B, as many as fit); chunk c writes its per-row losses at losses + c·C·T. */
  evalChunks: Program[];
}

export function makePlan(spec: ModelSpec, nText: number, o: PlanOptions): Plan {
  const lay = layout(spec);
  const T = spec.T;
  const lineStride = CONTEXT_MARGIN + T + o.W;
  const write = nText;
  const prompt = write + o.G * lineStride;
  const startsTrain = prompt + CONTEXT_MARGIN + T + 1;
  const startsEval = startsTrain + o.B;
  const startsWrite = startsEval + o.E;
  const startsPredict = startsWrite + o.G;
  const counter = startsPredict + 1;
  const tokens = { size: counter + 1, write, lineStride, prompt, startsTrain, startsEval, startsWrite, startsPredict, counter };

  const train = trainProgram(spec, lay, o.B, { starts: startsTrain, gen: 0 });
  let top = train.work;
  // Evaluation runs forward only, so it can take bigger chunks than a training batch in the same memory.
  let C = o.B;
  while (C * 2 <= Math.max(o.B, 64) && forwardProgram(spec, lay, C * 2, T, { starts: 0, gen: 0 }, 0).work <= train.work) C *= 2;
  const chunks = Math.ceil(o.E / C);
  const sizes = Array.from({ length: chunks }, (_, c) => Math.min(C, o.E - c * C));
  // Losses go after everything any program uses (writing and prediction use at most a training step's memory).
  const probe = sizes.map((b) => forwardProgram(spec, lay, b, T, { starts: 0, gen: 0 }, 0).work);
  top = Math.max(top, ...probe, writeProgram(spec, lay, Math.max(1, o.G), { starts: 0, gen: 1 }, 1, 0).work);
  const losses = top;
  const evalChunks = sizes.map((b, c) => forwardProgram(spec, lay, b, T, { starts: startsEval + c * C, gen: 0 }, losses + c * C * T));
  const work = { size: losses + o.E * T + o.G * o.W, losses, surprise: losses + o.E * T };
  if (work.size * 4 > MAX_BINDING_BYTES) throw new Error(`the model needs ${Math.ceil((work.size * 4) / 2 ** 20)} MB of working memory; at most 128 MB fit`);
  return { ...o, spec, lay, nText, tokens, work, opt: optLayout(lay.P), train, evalChunks };
}

/** Megabytes of working memory a model would need (for the settings to keep within the limit). */
export function workMegabytes(spec: ModelSpec, B: number): number {
  return (trainProgram(spec, layout(spec), B, { starts: 0, gen: 0 }).work * 4) / 2 ** 20;
}

/** The writing program of a plan, at a temperature and seed. */
export const writeOf = (p: Plan, temperature: number, seed: number) => writeProgram(p.spec, p.lay, p.G, { starts: p.tokens.startsWrite, gen: p.tokens.counter }, temperature, seed, p.work.surprise, p.W);

/** A forward pass of the prompt window (T' letters at tokens[prompt + margin …]). */
export const predictOf = (p: Plan, length: number) => forwardProgram(p.spec, p.lay, 1, length, { starts: p.tokens.startsPredict, gen: 0 }, null);

/** Writing line g's first token (its margin), and where its window starts once n letters are written. */
export const lineStart = (p: Plan, g: number) => p.tokens.write + g * p.tokens.lineStride;
