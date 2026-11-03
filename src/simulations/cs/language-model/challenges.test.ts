import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { BIGRAM_TOLERANCE, COPY_LOSS, LM_CHALLENGES, LOOP_REPETITION, VERBATIM } from "./challenges";
import type { LmReadouts } from "./challenges";
import { initParams } from "./model";
import { tableLadder, tableResult } from "./ngram";
import type { TableResult } from "./ngram";
import { LM_DEFAULTS, lmConfigOf } from "./settings";
import type { LmSettings } from "./settings";
import { curvePoint, evaluationWindows } from "./sim";
import { createTwin } from "./twin";
import { batchStream, shakespeareCorpus } from "./text";

const read = (f: string) => gunzipSync(readFileSync(`public/data/${f}`)).toString("utf8");
const challenge = (id: string) => LM_CHALLENGES.find((c) => c.id === id)!;
const settingsFor = (id: string, patch: Partial<LmSettings> = {}): LmSettings => ({ ...LM_DEFAULTS, ...challenge(id).setup, ...patch });
const passes = (id: string, params: LmSettings, r: LmReadouts) => challenge(id).goal.check({ params, readouts: r });

const blank: LmReadouts = { ready: true, text: "shakespeare", kind: "gpt", layers: 2, steps: 0, now: null, refs: null, writing: null, tables: [], tablesSeen: [], shannon: null };
const writing = (patch: Partial<NonNullable<LmReadouts["writing"]>>) => ({ temperature: 1, lines: [], letters: 240, repetition: 0, surprise: 1, copy: null, names: null, ...patch });

describe("language-model challenges", () => {
  it("each has exactly one correct prediction, and a distinct id", () => {
    for (const c of LM_CHALLENGES) expect(c.prediction?.options.filter((o) => o.correct)).toHaveLength(1);
    expect(new Set(LM_CHALLENGES.map((c) => c.id)).size).toBe(LM_CHALLENGES.length);
    expect(LM_CHALLENGES).toHaveLength(7);
  });

  it("how surprised: the neural bigram learns the count table (2.45 nats), on the letters the page measures", () => {
    const s = settingsFor("bigram");
    const c = shakespeareCorpus(read("shakespeare.dat"));
    const config = lmConfigOf(s);
    const spec = { ...config.spec, V: c.vocab.length };
    const { starts, heldTargets } = evaluationWindows(c, spec.T);
    const twin = createTwin(spec, c.tokens, { B: config.batch, G: 1, W: 1, E: starts.length });
    twin.load(initParams(spec, s.seed));
    const stream = batchStream(c, spec.T, config.batch, s.seed);
    for (let k = 0; k < 400; k++) twin.step(stream.next(), s.lr);
    const now = curvePoint(c, spec.T, starts, twin.evaluate(starts), 400);
    const table = tableResult(c, 1, 0, 1, heldTargets).rowsLoss!;
    expect(table).toBeGreaterThan(2.4);
    expect(table).toBeLessThan(2.55);
    expect(Math.abs(now.held - table)).toBeLessThanOrEqual(BIGRAM_TOLERANCE);
    const r: LmReadouts = { ...blank, kind: "bigram", steps: 400, now, refs: { uniform: Math.log(65), unigram: null, bigram: table, best: null } };
    expect(passes("bigram", s, r)).toBe(true);
    // Untrained, it is at guessing level, far from the table.
    expect(passes("bigram", s, { ...r, steps: 0, now: { ...now, held: Math.log(65) } })).toBe(false);
    // The transformer can't take this one.
    expect(passes("bigram", { ...s, model: "gpt" }, { ...r, kind: "gpt" })).toBe(false);
  }, 120_000);

  it("more context, until it memorises: the best Shakespeare table is k = 4, found after looking further", () => {
    const c = shakespeareCorpus(read("shakespeare.dat"));
    const g = tableLadder(c, 6, 10);
    const tables: TableResult[] = [];
    for (let r = g.next(); !r.done; r = g.next()) tables.push(r.value);
    const s = settingsFor("tables");
    expect(s.view).toBe("tables");
    const r = { ...blank, tables };
    expect(passes("tables", { ...s, tk: 4 }, { ...r, tablesSeen: [1, 4] })).toBe(false); // never looked beyond
    expect(passes("tables", { ...s, tk: 4 }, { ...r, tablesSeen: [1, 6, 4] })).toBe(true);
    expect(passes("tables", { ...s, tk: 5 }, { ...r, tablesSeen: [1, 6, 5] })).toBe(false);
    expect(passes("tables", { ...s, tk: 4 }, { ...r, tables: tables.slice(0, 5), tablesSeen: [6, 4] })).toBe(false); // still counting
  }, 60_000);

  it("goals read what they claim (beat the table, temperature, by heart, copying, Shannon)", () => {
    const beat = settingsFor("beat-table");
    const refs = { uniform: Math.log(27), unigram: 2.82, bigram: 2.46, best: { k: 3, loss: 2.13 } };
    const point = (held: number, extra: object = {}) => ({ step: 1000, train: held, held, ...extra });
    expect(passes("beat-table", beat, { ...blank, text: "names", now: point(2.1), refs })).toBe(true);
    expect(passes("beat-table", beat, { ...blank, text: "names", now: point(2.2), refs })).toBe(false);
    expect(passes("beat-table", beat, { ...blank, text: "shakespeare", now: point(1.5), refs })).toBe(false);

    const temp = settingsFor("temperature");
    const looping = { ...blank, steps: 600, writing: writing({ temperature: 0, repetition: LOOP_REPETITION + 0.1 }) };
    expect(passes("temperature", temp, looping)).toBe(true);
    expect(passes("temperature", temp, { ...looping, steps: 100 })).toBe(false); // an untrained model loops for no reason
    expect(passes("temperature", temp, { ...looping, writing: writing({ temperature: 1, repetition: 0.5 }) })).toBe(false);
    expect(passes("temperature", temp, { ...looping, writing: writing({ temperature: 0, letters: 100, repetition: 0.5 }) })).toBe(false);

    const heart = settingsFor("by-heart");
    expect(heart.text).toBe("sonnet");
    expect(passes("by-heart", heart, { ...blank, text: "sonnet", writing: writing({ copy: VERBATIM }) })).toBe(true);
    expect(passes("by-heart", heart, { ...blank, text: "sonnet", writing: writing({ copy: VERBATIM - 1 }) })).toBe(false);

    const copy = settingsFor("copy");
    expect(copy.layers).toBe(1);
    const copying = { ...blank, text: "copying" as const, now: point(2, { repeated: COPY_LOSS - 0.1, fresh: 3.26 }) };
    expect(passes("copy", copy, { ...copying, layers: 1 })).toBe(false);
    expect(passes("copy", { ...copy, layers: 2 }, { ...copying, layers: 2 })).toBe(true);

    const shannon = settingsFor("shannon");
    const played = { played: 50, you: { lower: 1, upper: 2 }, model: { lower: 1.5, upper: 2.5 }, modelBits: 2.4 };
    expect(passes("shannon", shannon, { ...blank, shannon: played })).toBe(true);
    expect(passes("shannon", shannon, { ...blank, shannon: { ...played, played: 49 } })).toBe(false);
  });
});
