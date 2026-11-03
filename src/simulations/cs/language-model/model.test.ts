import { describe, expect, it } from "vitest";
import { gelu, geluSlope, pickLetter, runOps, sampleUniform } from "./cpu";
import { gaussian, initParams, layout, parameterCount, stepValues } from "./model";
import type { ModelSpec } from "./model";
import { makePlan } from "./plan";
import { createTwin } from "./twin";
import { CONTEXT_MARGIN, rng } from "./text";

const base: ModelSpec = { kind: "gpt", V: 7, T: 6, k: 3, e: 4, h: 5, L: 2, H: 2, d: 8 };
const SPECS: [string, ModelSpec][] = [
  ["neural bigram", { ...base, kind: "bigram" }],
  ["MLP", { ...base, kind: "mlp" }],
  ["transformer, 2 layers", base],
  ["transformer, 1 layer, 1 head", { ...base, L: 1, H: 1, d: 6 }],
];

function randomText(n: number, V: number, seed: number) {
  const r = rng(seed);
  return Uint8Array.from({ length: n }, () => Math.floor(r() * V));
}

/** Parameters with enough spread for every nonlinearity to matter (the real init is too gentle for a check). */
function lively(spec: ModelSpec, seed: number) {
  const p = initParams(spec, seed);
  const g = gaussian(rng(seed + 99));
  for (let i = 0; i < p.length; i++) p[i] += 0.4 * g();
  return p;
}

/** The training program up to (not including) the optimiser: forward, loss and its gradient, backward. */
function lossAndGrad(spec: ModelSpec, params: Float64Array, starts: number[]) {
  const text = randomText(200, spec.V, 5);
  const twin = createTwin(spec, text, { B: starts.length, G: 1, W: 4, E: 1 });
  twin.load(params);
  twin.buf.tokens.set(starts, twin.plan.tokens.startsTrain);
  const ops = twin.plan.train.ops;
  const cut = ops.findIndex((o) => o.kernel === "normPartial");
  const ce = ops.find((o) => o.kernel === "crossEntropy")!;
  const lossOf = (p: Float64Array) => {
    twin.buf.params.set(p);
    runOps(ops.slice(0, ops.indexOf(ce) + 1), twin.buf);
    let s = 0;
    for (let r = 0; r < starts.length * spec.T; r++) s += twin.buf.work[ce.c + r];
    return s / (starts.length * spec.T);
  };
  twin.buf.params.set(params);
  runOps(ops.slice(0, cut), twin.buf);
  const grad = twin.buf.opt.slice(0, twin.plan.lay.P);
  return { grad, lossOf };
}

describe("backpropagation through every op (twin, centred differences)", () => {
  for (const [name, spec] of SPECS) {
    it(name, () => {
      const params = lively(spec, 3);
      const { grad, lossOf } = lossAndGrad(spec, params, [20, 57, 101]);
      const h = 1e-5;
      let worst = 0;
      const scale = Math.max(...Array.from(grad, Math.abs));
      expect(scale).toBeGreaterThan(1e-4);
      for (let i = 0; i < params.length; i++) {
        const up = params.slice();
        up[i] += h;
        const down = params.slice();
        down[i] -= h;
        const numeric = (lossOf(up) - lossOf(down)) / (2 * h);
        worst = Math.max(worst, Math.abs(numeric - grad[i]) / Math.max(Math.abs(numeric) + Math.abs(grad[i]), 1e-3 * scale));
      }
      expect(worst).toBeLessThan(1e-6);
    });
  }
});

describe("models", () => {
  it("lay out decayed matrices first and count parameters as documented", () => {
    const shakespeare: ModelSpec = { kind: "gpt", V: 65, T: 64, k: 8, e: 16, h: 128, L: 2, H: 4, d: 64 };
    expect(parameterCount(shakespeare)).toBeGreaterThan(100_000);
    expect(parameterCount(shakespeare)).toBeLessThan(120_000);
    const big = { ...shakespeare, L: 4, d: 128 };
    expect(parameterCount(big)).toBeGreaterThan(780_000);
    expect(parameterCount(big)).toBeLessThan(850_000);
    const lay = layout(shakespeare);
    expect(lay.at.wte).toBe(0);
    expect(lay.nDecay).toBe(lay.at.ln1_0);
    expect(parameterCount({ ...shakespeare, kind: "bigram" })).toBe(65 * 65);
  });

  it("start the bigram at uniform guessing: loss ln V", () => {
    const spec: ModelSpec = { ...base, kind: "bigram" };
    const text = randomText(300, spec.V, 1);
    const twin = createTwin(spec, text, { B: 2, G: 1, W: 1, E: 2 });
    twin.load(initParams(spec, 1));
    const losses = twin.evaluate([30, 90]);
    for (const l of losses) expect(l).toBeCloseTo(Math.log(spec.V), 12);
  });

  it("warm the learning rate up over the first steps and correct Adam's bias", () => {
    expect(stepValues(0, 1e-3)[0]).toBeCloseTo(1e-5, 12);
    expect(stepValues(99, 1e-3)[0]).toBeCloseTo(1e-3, 12);
    expect(stepValues(5000, 1e-3)[0]).toBeCloseTo(1e-3, 12);
    expect(stepValues(0, 1)[1]).toBeCloseTo(0.1, 12);
    expect(stepValues(0, 1)[2]).toBeCloseTo(0.01, 12);
  });

  it("clip the gradient to norm 1 and apply decoupled weight decay to matrices only", () => {
    const spec: ModelSpec = { ...base, kind: "mlp" };
    const text = randomText(300, spec.V, 2);
    const twin = createTwin(spec, text, { B: 2, G: 1, W: 1, E: 1 });
    const p0 = lively(spec, 4);
    twin.load(p0);
    twin.step([40, 120], 0.01);
    const o = twin.plan.opt;
    const norm = twin.buf.opt[o.norm];
    let sq = 0;
    for (let i = 0; i < twin.plan.lay.P; i++) sq += twin.buf.opt[i] ** 2;
    expect(norm).toBeCloseTo(Math.sqrt(sq), 10);
    expect(twin.buf.opt[o.scale]).toBeCloseTo(Math.min(1, 1 / (norm + 1e-6)), 12);
    // First Adam step: every parameter with a gradient moves by lr · warm-up (m̂/√v̂ = ±1), plus decay on matrices.
    const lr = stepValues(0, 0.01)[0];
    const lay = twin.plan.lay;
    for (const i of [0, lay.at.W1 + 3, lay.at.b2 + 1]) {
      const g = twin.buf.opt[i];
      if (Math.abs(g) < 1e-12) continue;
      const decayed = i < lay.nDecay ? p0[i] * (1 - lr * 0.1) : p0[i];
      expect(twin.buf.params[i]).toBeCloseTo(decayed - lr * Math.sign(g) * (1 / (1 + 1e-8 / Math.abs(g * twin.buf.opt[o.scale]))), 9);
    }
  });

  it("write letters on several lines at once, each reading its own last letters", () => {
    const spec: ModelSpec = { ...base, kind: "bigram" };
    const text = randomText(300, spec.V, 3);
    const twin = createTwin(spec, text, { B: 2, G: 3, W: 5, E: 1 });
    // A bigram table that always moves to the next letter: v → v + 1 (mod V).
    const p = new Float64Array(spec.V * spec.V);
    for (let v = 0; v < spec.V; v++) p[v * spec.V + ((v + 1) % spec.V)] = 50;
    twin.load(p);
    const ctx = (last: number) => Uint8Array.from({ length: CONTEXT_MARGIN + spec.T }, (_, i) => (i === CONTEXT_MARGIN + spec.T - 1 ? last : 0));
    twin.startWriting([ctx(0), ctx(3), ctx(5)], 1, 7);
    twin.write(5);
    expect(twin.written().map((l) => Array.from(l))).toEqual([
      [1, 2, 3, 4, 5],
      [4, 5, 6, 0, 1],
      [6, 0, 1, 2, 3],
    ]);
  });

  it("predict from a prompt shorter than the context", () => {
    const spec = base;
    const twin = createTwin(spec, randomText(300, spec.V, 4), { B: 2, G: 1, W: 1, E: 1 });
    twin.load(lively(spec, 5));
    const { logits, attention, T } = twin.predict([1, 2, 3], 0);
    expect(T).toBe(3);
    expect(logits.length).toBe(spec.V);
    expect(attention.length).toBe(spec.L);
    // Causal: each query's weights sum to 1 over the letters so far and are 0 after it.
    const a = attention[0];
    for (let h = 0; h < spec.H; h++) {
      for (let i = 0; i < T; i++) {
        let s = 0;
        for (let j = 0; j < T; j++) {
          if (j > i) expect(a[(h * T + i) * T + j]).toBe(0);
          s += a[(h * T + i) * T + j];
        }
        expect(s).toBeCloseTo(1, 12);
      }
    }
  });

  it("refuse a model too big for a storage binding", () => {
    expect(() => makePlan({ kind: "gpt", V: 65, T: 128, k: 8, e: 16, h: 128, L: 4, H: 4, d: 128 }, 1000, { B: 64, G: 8, W: 64, E: 64 })).toThrow(/128 MB/);
  });
});

describe("sampling", () => {
  it("takes the most likely letter at temperature 0 and follows softmax(scores / T) otherwise", () => {
    const scores = [0.1, 2, -1, 1.5];
    expect(pickLetter(scores, 0, 4, 0, 0.99)).toBe(1);
    const counts = [0, 0, 0, 0];
    for (let n = 0; n < 20000; n++) counts[pickLetter(scores, 0, 4, 0.7, sampleUniform(11, 0, n))]++;
    const w = scores.map((s) => Math.exp(s / 0.7));
    const total = w.reduce((a, b) => a + b);
    for (let v = 0; v < 4; v++) expect(counts[v] / 20000).toBeCloseTo(w[v] / total, 1);
  });

  it("draws uniform numbers that are exact in 24 bits and differ between lines and positions", () => {
    const u = sampleUniform(3, 0, 0);
    expect(u).toBeGreaterThanOrEqual(0);
    expect(u).toBeLessThan(1);
    expect(u * 16777216).toBe(Math.round(u * 16777216));
    expect(sampleUniform(3, 1, 0)).not.toBe(u);
    expect(sampleUniform(3, 0, 1)).not.toBe(u);
  });

  it("uses the tanh form of GELU, with its exact slope", () => {
    for (const x of [-3, -0.5, 0, 0.7, 2.5]) {
      const h = 1e-6;
      expect(geluSlope(x)).toBeCloseTo((gelu(x + h) - gelu(x - h)) / (2 * h), 8);
    }
    expect(gelu(0)).toBe(0);
    expect(gelu(10)).toBeCloseTo(10, 6);
  });
});

describe("letter maps", () => {
  it("find the two main directions of the letter vectors", async () => {
    const { project2d } = await import("./embedding");
    // Points on a tilted plane in 3D: two components explain everything.
    const rows = 20;
    const data = new Float64Array(rows * 3);
    for (let r = 0; r < rows; r++) {
      const a = Math.cos(r) * 3;
      const b = Math.sin(r * 1.7);
      data.set([a + b, a - b, 0.5 * a], r * 3);
    }
    const p = project2d({ rows, cols: 3, data });
    expect(p.explained).toBeCloseTo(1, 6);
    // Distances between points are preserved by the projection (they lie in the plane).
    const d3 = Math.hypot(data[0] - data[3], data[1] - data[4], data[2] - data[5]);
    expect(Math.hypot(p.x[0] - p.x[1], p.y[0] - p.y[1])).toBeCloseTo(d3, 6);
  });
});
