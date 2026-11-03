/**
 * The CPU twin as a whole machine: a plan's buffers in 64-bit, and the same operations the GPU engine offers
 * (load, step, evaluate, write, predict), run by cpu.ts. Used by the tests and to measure the challenges.
 */
import { runOps } from "./cpu";
import type { Buffers } from "./cpu";
import { stepValues } from "./model";
import { lineStart, makePlan, predictOf, writeOf } from "./plan";
import type { Plan, PlanOptions } from "./plan";
import type { ModelSpec } from "./model";
import { CONTEXT_MARGIN } from "./text";

export function createTwin(spec: ModelSpec, text: Uint8Array, opts: PlanOptions) {
  const plan: Plan = makePlan(spec, text.length, opts);
  const buf: Buffers = {
    params: new Float64Array(plan.lay.P),
    work: new Float64Array(plan.work.size),
    opt: new Float64Array(plan.opt.size),
    tokens: new Uint32Array(plan.tokens.size),
  };
  buf.tokens.set(text);
  let steps = 0;
  let writing = { temperature: 1, seed: 1 };

  return {
    plan,
    buf,
    get steps() {
      return steps;
    },
    load(p: ArrayLike<number>) {
      buf.params.set(p);
      buf.opt.fill(0);
      steps = 0;
    },
    /** One AdamW step on the windows starting at `starts` (B of them). */
    step(starts: ArrayLike<number>, lr: number) {
      buf.tokens.set(starts, plan.tokens.startsTrain);
      buf.opt.set(stepValues(steps, lr), plan.opt.step);
      runOps(plan.train.ops, buf);
      steps++;
    },
    /** Per-row losses of the E evaluation windows (window-major, T per window). */
    evaluate(starts: ArrayLike<number>): Float64Array {
      buf.tokens.set(starts, plan.tokens.startsEval);
      for (const chunk of plan.evalChunks) runOps(chunk.ops, buf);
      return buf.work.slice(plan.work.losses, plan.work.losses + plan.E * spec.T);
    },
    /** Starts G lines of writing from their contexts (margin + T letters each). */
    startWriting(contexts: readonly ArrayLike<number>[], temperature: number, seed: number) {
      for (let g = 0; g < plan.G; g++) {
        buf.tokens.set(contexts[g], lineStart(plan, g));
        buf.tokens[plan.tokens.startsWrite + g] = lineStart(plan, g) + CONTEXT_MARGIN;
      }
      buf.tokens[plan.tokens.counter] = 0;
      writing = { temperature, seed };
    },
    /** Writes n more letters on every line. */
    write(n: number) {
      const prog = writeOf(plan, writing.temperature, writing.seed);
      const k = Math.min(n, plan.W - buf.tokens[plan.tokens.counter]);
      for (let i = 0; i < k; i++) runOps(prog.ops, buf);
    },
    /** The letters written so far on each line. */
    written(): Uint32Array[] {
      const n = buf.tokens[plan.tokens.counter];
      return Array.from({ length: plan.G }, (_, g) => buf.tokens.slice(lineStart(plan, g) + CONTEXT_MARGIN + spec.T, lineStart(plan, g) + CONTEXT_MARGIN + spec.T + n));
    },
    /** The model's surprise (−ln p, nats) at each letter it wrote, per line. */
    surprise(): Float64Array[] {
      const n = Math.min(plan.W, buf.tokens[plan.tokens.counter]);
      return Array.from({ length: plan.G }, (_, g) => buf.work.slice(plan.work.surprise + g * plan.W, plan.work.surprise + g * plan.W + n));
    },
    /** The scores for the letter after `prompt` (its last ≤ T letters), and each layer's attention. */
    predict(prompt: readonly number[], pad: number) {
      const T = Math.max(1, Math.min(spec.T, prompt.length));
      const window = prompt.slice(-T);
      const region = plan.tokens.prompt;
      buf.tokens.fill(pad, region, region + CONTEXT_MARGIN);
      buf.tokens.set(window.length ? window : [pad], region + CONTEXT_MARGIN);
      buf.tokens[plan.tokens.startsPredict] = region + CONTEXT_MARGIN;
      const prog = predictOf(plan, T);
      runOps(prog.ops, buf);
      const V = spec.V;
      return {
        logits: buf.work.slice(prog.acts.logits + (T - 1) * V, prog.acts.logits + T * V),
        attention: prog.acts.attention.map((p) => buf.work.slice(p, p + spec.H * T * T)),
        T,
      };
    },
  };
}

export type Twin = ReturnType<typeof createTwin>;
