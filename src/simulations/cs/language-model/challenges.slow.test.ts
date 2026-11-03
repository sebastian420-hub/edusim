/**
 * The challenge claims that need real training: run on the GPU (about an hour on the software renderer), so only with
 * LM_SLOW=1. `pnpm test:lm-slow` runs them; their measured numbers are recorded in docs/LANGUAGE-MODEL-PLAN.md.
 */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { tryInitGpu } from "@/test/gpu";
import type { NodeGpu } from "@/test/gpu";
import { COPY_LOSS, LM_CHALLENGES, LOOP_REPETITION, VERBATIM } from "./challenges";
import type { LmReadouts } from "./challenges";
import { createEngine } from "./engine";
import { initParams } from "./model";
import { bestTable, tableResult } from "./ngram";
import { longestCopy, repetition } from "./sample";
import { LM_DEFAULTS, lmConfigOf } from "./settings";
import type { LmSettings } from "./settings";
import { curvePoint, evaluationWindows, LINE_LETTERS, LINES } from "./sim";
import { batchStream, CONTEXT_MARGIN, copyingCorpus, decode, namesCorpus, seedContext, shakespeareCorpus, sonnetCorpus } from "./text";
import type { Corpus } from "./text";

const SLOW = !!process.env.LM_SLOW;
/** Steps the copying game is trained for (the two-layer model's induction head forms at about 2 000–3 000). */
const COPY_STEPS = 3500;
const read = (f: string) => gunzipSync(readFileSync(`public/data/${f}`)).toString("utf8");
const challenge = (id: string) => LM_CHALLENGES.find((c) => c.id === id)!;
const settingsFor = (id: string, patch: Partial<LmSettings> = {}): LmSettings => ({ ...LM_DEFAULTS, ...challenge(id).setup, ...patch });

let gpu: NodeGpu | null = null;
beforeAll(async () => {
  if (SLOW) gpu = await tryInitGpu();
});
afterAll(() => gpu?.dispose());

/** Trains as the page does (same windows, batches and measurements) and returns what the goals read. */
async function trainAsThePage(s: LmSettings, c: Corpus, steps: number, opts: { write?: number } = {}) {
  const config = lmConfigOf(s);
  const spec = { ...config.spec, V: c.vocab.length };
  const { starts, heldTargets } = evaluationWindows(c, spec.T);
  const engine = createEngine(gpu!, spec, c.tokens, { B: config.batch, G: LINES, W: LINE_LETTERS, E: starts.length });
  engine.load(initParams(spec, s.seed));
  const stream = batchStream(c, spec.T, config.batch, s.seed);
  for (let k = 0; k < steps; k++) {
    engine.step(stream.next(), s.lr);
    if (k % 50 === 49) await engine.readOpt(); // keep the queue short
  }
  const now = curvePoint(c, spec.T, starts, await engine.evaluate(starts), steps);
  let lines: string[] = [];
  if (opts.write !== undefined) {
    engine.startWriting(Array.from({ length: LINES }, (_, g) => seedContext(c, CONTEXT_MARGIN + spec.T, 100 + g)), opts.write, 3);
    engine.write(LINE_LETTERS);
    lines = (await engine.written()).map((l) => decode(c, l));
  }
  engine.dispose();
  return { now, lines, heldTargets };
}

describe.skipIf(!SLOW)("language-model challenges, trained on the GPU", () => {
  it("beat the table: the transformer's loss on unseen names drops below the best count table", async () => {
    const s = settingsFor("beat-table");
    const c = namesCorpus(read("names.dat"));
    const { now, heldTargets } = await trainAsThePage(s, c, 2500);
    const best = bestTable([0, 1, 2, 3, 4].map((k) => tableResult(c, k, 0, 1, heldTargets)));
    expect(best.k).toBe(3);
    expect(now.held).toBeLessThan(best.rowsLoss!);
    const r: LmReadouts = { ready: true, text: "names", kind: "gpt", layers: 2, steps: 2500, now, refs: { uniform: Math.log(27), unigram: null, bigram: null, best: { k: 3, loss: best.rowsLoss! } }, writing: null, tables: [], tablesSeen: [], shannon: null };
    expect(challenge("beat-table").goal.check({ params: s, readouts: r })).toBe(true);
  }, 3_600_000);

  it("temperature: trained 500 steps on Shakespeare, at temperature 0 it loops; at 1 it does not", async () => {
    const s = settingsFor("temperature");
    const c = shakespeareCorpus(read("shakespeare.dat"));
    const cold = await trainAsThePage(s, c, 500, { write: 0 });
    const rep0 = cold.lines.reduce((a, l) => a + repetition(l), 0) / cold.lines.length;
    expect(rep0).toBeGreaterThanOrEqual(LOOP_REPETITION);
    const warm = await trainAsThePage(s, c, 500, { write: 1 });
    const rep1 = warm.lines.reduce((a, l) => a + repetition(l), 0) / warm.lines.length;
    expect(rep1).toBeLessThan(LOOP_REPETITION);
  }, 3_600_000);

  it("learn by heart: on Sonnet 18 the transformer writes 80 letters or more verbatim", async () => {
    const s = settingsFor("by-heart");
    const c = sonnetCorpus();
    const { lines } = await trainAsThePage(s, c, 500, { write: s.temp });
    const source = decode(c, c.tokens);
    expect(Math.max(...lines.map((l) => longestCopy(l, source).length))).toBeGreaterThanOrEqual(VERBATIM);
  }, 3_600_000);

  it("copy what you just saw: two layers learn to copy (an induction head), one layer can't", async () => {
    const s = settingsFor("copy");
    const c = copyingCorpus();
    const one = await trainAsThePage(s, c, COPY_STEPS);
    const two = await trainAsThePage({ ...s, layers: 2 }, c, COPY_STEPS);
    expect(one.now.repeated!).toBeGreaterThan(2 * COPY_LOSS);
    expect(two.now.repeated!).toBeLessThan(COPY_LOSS);
    // The first copy stays unpredictable for both.
    expect(two.now.fresh!).toBeGreaterThan(3);
  }, 3_600_000);
});

it("the slow challenge tests are opt-in (LM_SLOW=1)", () => {
  expect(typeof SLOW).toBe("boolean");
});
