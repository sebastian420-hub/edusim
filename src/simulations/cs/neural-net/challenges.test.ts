import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { NET_CHALLENGES } from "./challenges";
import type { NetReadouts } from "./challenges";
import { digitsDataset, parseDigits } from "./digits";
import { evaluate, neuronSpread, planeDataset, train } from "./net";
import { digitSpec, NET_DEFAULTS, planeSpec } from "./settings";
import type { NetSettings } from "./settings";

const challenge = (id: string) => NET_CHALLENGES.find((c) => c.id === id)!;
const settingsFor = (id: string, patch: Partial<NetSettings> = {}): NetSettings => ({ ...NET_DEFAULTS, ...challenge(id).setup, ...patch });

/** Trains the twin as the Playground does and measures what the goals read. */
function playground(s: NetSettings, steps: number): NetReadouts {
  const data = planeDataset(s.data, s.points, s.noise, s.train, s.features, s.seed);
  const spec = planeSpec(s);
  const r = train(spec, data, { lr: s.lr, momentum: s.mom, l2: s.l2, batch: Math.min(s.batch, data.nTrain), seed: s.seed, init: s.init }, steps);
  const tr = evaluate(spec, r.lay, r.p, data, 0, data.nTrain);
  const te = evaluate(spec, r.lay, r.p, data, data.nTrain, data.n - data.nTrain);
  return {
    view: "playground",
    steps,
    hidden: spec.hidden.length,
    now: { step: steps, epoch: r.stream.epochs, trainLoss: tr.loss, trainAcc: tr.accuracy, testLoss: te.loss, testAcc: te.accuracy },
    spread: neuronSpread(r.lay, r.p),
  };
}

const passes = (id: string, s: NetSettings, r: NetReadouts) => challenge(id).goal.check({ params: s, readouts: r });

describe("neural network challenges", () => {
  it("each has exactly one correct prediction", () => {
    for (const c of NET_CHALLENGES) expect(c.prediction?.options.filter((o) => o.correct)).toHaveLength(1);
  });

  it("a straight line can't do XOR (about 50 %); a hidden layer of 4 can", () => {
    const setup = settingsFor("line-xor");
    const r = playground(setup, 1000);
    expect(r.now!.testAcc).toBeLessThan(0.65);
    expect(r.now!.trainLoss).toBeGreaterThan(0.68); // ≈ ln 2
    expect(passes("line-xor", setup, r)).toBe(false);
    const hidden = settingsFor("line-xor", { layers: [4] });
    expect(passes("line-xor", hidden, playground(hidden, 600))).toBe(true);
    // The x·y feature is the other challenge's answer, not this one's.
    const xy = settingsFor("line-xor", { features: 0b10011 });
    expect(passes("line-xor", xy, playground(xy, 600))).toBe(false);
  });

  it("the right input: x·y solves XOR with no hidden layer, x² does not", () => {
    const xy = settingsFor("features", { features: 0b10011 });
    expect(passes("features", xy, playground(xy, 600))).toBe(true);
    const x2 = settingsFor("features", { features: 0b00111 });
    expect(passes("features", x2, playground(x2, 600))).toBe(false);
  });

  it("too big a step: learning rate 10 leaves the circle at a guess; the setup's 0.1 learns it", () => {
    const setup = settingsFor("big-step");
    const ok = playground(setup, 600);
    expect(ok.now!.trainAcc).toBeGreaterThanOrEqual(0.95);
    expect(passes("big-step", setup, ok)).toBe(false);
    const big = settingsFor("big-step", { lr: 10 });
    const r = playground(big, 300);
    expect(r.now!.trainAcc).toBeLessThanOrEqual(0.6);
    expect(passes("big-step", big, r)).toBe(true);
  });

  it("memorising: the setup overfits within 4 000 steps; with the training share raised it is no longer allowed", () => {
    const setup = settingsFor("overfit");
    const r = playground(setup, 4000);
    expect(r.now!.trainAcc).toBeGreaterThanOrEqual(0.95);
    expect(r.now!.testAcc).toBeLessThan(r.now!.trainAcc - 0.2);
    expect(passes("overfit", setup, r)).toBe(true);
    const more = settingsFor("overfit", { train: 0.5 });
    expect(passes("overfit", more, playground(more, 2000))).toBe(false);
  });

  it("zero start: clones that learn nothing; a random start solves XOR", () => {
    const setup = settingsFor("zeros");
    const r = playground(setup, 500);
    expect(r.spread).toBe(0);
    expect(r.now!.testAcc).toBeLessThan(0.6);
    expect(passes("zeros", setup, r)).toBe(false);
    const random = settingsFor("zeros", { init: "random" });
    expect(passes("zeros", random, playground(random, 500))).toBe(true);
  });

  it("deep and stuck: six sigmoid layers stay near 50 %; six tanh layers reach 90 %", () => {
    const setup = settingsFor("deep");
    const r = playground(setup, 800);
    expect(r.now!.testAcc).toBeLessThan(0.6);
    expect(passes("deep", setup, r)).toBe(false);
    const tanh = settingsFor("deep", { act: "tanh" });
    expect(passes("deep", tanh, playground(tanh, 800))).toBe(true);
    const shallow = settingsFor("deep", { layers: [4] }); // not the point
    expect(passes("deep", shallow, playground(shallow, 800))).toBe(false);
  });

  it("read my handwriting: the straight line stays below 92 %, one hidden layer passes", () => {
    const digits = parseDigits(gunzipSync(readFileSync("public/data/mnist-12k.dat")));
    const data = digitsDataset(digits);
    const run = (s: NetSettings, epochs: number): NetReadouts => {
      const spec = digitSpec(s);
      const r = train(spec, data, { lr: s.dlr, momentum: 0.9, l2: 0, batch: s.dbatch, seed: s.seed }, Math.round((epochs * digits.nTrain) / s.dbatch));
      const te = evaluate(spec, r.lay, r.p, data, digits.nTrain, digits.count - digits.nTrain);
      return { view: "digits", steps: 0, hidden: spec.hidden.length, now: { step: 0, epoch: epochs, trainLoss: 0, trainAcc: 0, testLoss: te.loss, testAcc: te.accuracy }, spread: null };
    };
    const setup = settingsFor("digits");
    const line = run(setup, 5);
    expect(line.now!.testAcc).toBeGreaterThan(0.85); // "about 9 in 10"
    expect(passes("digits", setup, line)).toBe(false);
    const hidden = settingsFor("digits", { dnet: "h128" });
    expect(passes("digits", hidden, run(hidden, 5))).toBe(true);
  }, 120_000);
});
