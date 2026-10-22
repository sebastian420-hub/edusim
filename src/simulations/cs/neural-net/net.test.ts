import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { confusion, digitsDataset, DIGIT_PIXELS, DIGIT_SIZE, parseDigits, toDigit } from "./digits";
import {
  batchStream,
  evaluate,
  FEATURES,
  forward,
  gradient,
  initParams,
  layerGradientSizes,
  layout,
  neuronSpread,
  numericGradient,
  permutation,
  planeDataset,
  probabilityAt,
  PROBE_S,
  rng,
  stepProfile,
  train,
  update,
  weightMask,
} from "./net";
import type { Activation, Dataset, NetSpec } from "./net";

const XY = 0b11;
const all = (n: number) => Uint32Array.from({ length: n }, (_, i) => i);

/** Random inputs and labels for the gradient check. */
function randomData(n: number, dim: number, classes: number, seed: number): Dataset {
  const r = rng(seed);
  return { n, dim, nTrain: n, x: Float64Array.from({ length: n * dim }, () => 2 * r() - 1), y: Uint8Array.from({ length: n }, () => Math.floor(r() * classes)), classes };
}

describe("network layout", () => {
  it("packs each layer's weights (row per neuron) then its biases", () => {
    const lay = layout({ inputs: 3, hidden: [4, 2], outputs: 1, activation: "tanh" });
    expect(lay.sizes).toEqual([3, 4, 2, 1]);
    expect(lay.layers.map((l) => [l.w, l.b])).toEqual([
      [0, 12],
      [16, 24],
      [26, 28],
    ]);
    expect(lay.count).toBe(29);
    expect(lay.maxWidth).toBe(4);
  });

  it("seeded: the same seed gives the same weights, and biases start at zero", () => {
    const spec: NetSpec = { inputs: 2, hidden: [8], outputs: 1, activation: "relu" };
    const a = initParams(spec, 7);
    expect(initParams(spec, 7)).toEqual(a);
    expect(initParams(spec, 8)).not.toEqual(a);
    const lay = layout(spec);
    for (const L of lay.layers) for (let j = 0; j < L.nOut; j++) expect(a[L.b + j]).toBe(0);
    // He-uniform for ReLU: |w| ≤ √(6 / fan-in).
    for (let k = 0; k < 16; k++) expect(Math.abs(a[k])).toBeLessThanOrEqual(Math.sqrt(6 / 2));
  });

  it("permutation and batch order: every training sample once per pass, the same order from the same seed", () => {
    const p = permutation(50, rng(3));
    expect([...p].sort((x, y) => x - y)).toEqual([...all(50)]);
    const s = batchStream(30, 7, 5);
    const seen: number[] = [];
    for (let k = 0; k < 30; k++) seen.push(...s.next());
    // 210 samples = 7 passes: each sample exactly 7 times.
    const counts = new Array(30).fill(0);
    for (const i of seen) counts[i]++;
    expect(counts.every((c) => c === 7)).toBe(true);
    expect(s.epochs).toBe(7);
    const t = batchStream(30, 7, 5);
    expect([...t.next()]).toEqual(seen.slice(0, 7));
  });
});

describe("backpropagation", () => {
  const cases: { act: Activation; hidden: number[]; outputs: number }[] = [
    { act: "tanh", hidden: [], outputs: 1 },
    { act: "tanh", hidden: [5, 3], outputs: 1 },
    { act: "sigmoid", hidden: [4, 4, 4], outputs: 1 },
    { act: "relu", hidden: [6], outputs: 1 },
    { act: "linear", hidden: [3], outputs: 1 },
    { act: "tanh", hidden: [5], outputs: 10 },
    { act: "relu", hidden: [7, 4], outputs: 10 },
  ];
  for (const c of cases) {
    it(`matches centred finite differences (${c.act}, ${JSON.stringify(c.hidden)}, ${c.outputs} output${c.outputs > 1 ? "s" : ""})`, () => {
      const spec: NetSpec = { inputs: 3, hidden: c.hidden, outputs: c.outputs, activation: c.act };
      const lay = layout(spec);
      const data = randomData(12, 3, Math.max(2, c.outputs), 11);
      const p = initParams(spec, 4);
      // Non-zero biases so every path is exercised.
      for (const L of lay.layers) for (let j = 0; j < L.nOut; j++) p[L.b + j] = 0.1 * (j - 1);
      const analytic = gradient(spec, lay, p, data, all(12)).grad;
      const numeric = numericGradient(spec, lay, p, data, all(12));
      // Relative error per parameter (cs231n's check), floored at 10⁻⁴ of the largest component: deep sigmoid
      // nets have components so small that the finite difference itself is only good to ~10⁻¹¹.
      const floor = 1e-4 * Math.max(...analytic.map(Math.abs));
      let worst = 0;
      for (let k = 0; k < p.length; k++) worst = Math.max(worst, Math.abs(analytic[k] - numeric[k]) / Math.max(Math.abs(analytic[k]), Math.abs(numeric[k]), floor));
      expect(worst).toBeLessThan(1e-6);
    });
  }

  it("the squared-error loss (twin only) has the right gradient too", () => {
    const spec: NetSpec = { inputs: 3, hidden: [], outputs: 1, activation: "linear" };
    const lay = layout(spec);
    const data = randomData(10, 3, 2, 2);
    const p = initParams(spec, 1);
    const a = gradient(spec, lay, p, data, all(10), "squared").grad;
    const n = numericGradient(spec, lay, p, data, all(10), "squared");
    for (let k = 0; k < p.length; k++) expect(a[k]).toBeCloseTo(n[k], 8);
  });

  it("SGD with momentum: v ← μv + g + λw (weights only), w ← w − ηv", () => {
    const lay = layout({ inputs: 1, hidden: [], outputs: 1, activation: "linear" });
    const p = Float64Array.from([2, 1]); // weight 2, bias 1
    const v = Float64Array.from([0.5, 0.5]);
    update(p, v, [1, 1], { lr: 0.1, momentum: 0.9, l2: 0.5 }, weightMask(lay));
    expect(v[0]).toBeCloseTo(0.45 + 1 + 1, 12); // weight: decay applies
    expect(v[1]).toBeCloseTo(0.45 + 1, 12); // bias: none
    expect(p[0]).toBeCloseTo(2 - 0.1 * 2.45, 12);
    expect(p[1]).toBeCloseTo(1 - 0.1 * 1.45, 12);
  });

  it("the decision map's probability is the sigmoid of the output logit", () => {
    const spec: NetSpec = { inputs: 7, hidden: [4], outputs: 1, activation: "tanh" };
    const lay = layout(spec);
    const p = initParams(spec, 9);
    const z = forward(spec, lay, p, [0.3, -0.2, 0.09, 0.04, -0.06, Math.sin(0.3 * Math.PI), Math.sin(-0.2 * Math.PI)])[1][0];
    expect(probabilityAt(spec, lay, p, (1 << FEATURES.length) - 1, 0.3, -0.2)).toBeCloseTo(1 / (1 + Math.exp(-z)), 12);
  });
});

describe("what the challenges claim", () => {
  it("a straight line cannot do XOR: the loss can never go below ln 2 on the four corners", () => {
    // By symmetry (x, y) → (−x, −y) the best linear model has zero weights, so it says 50 % everywhere: loss ln 2.
    const data: Dataset = { n: 4, dim: 2, nTrain: 4, x: Float64Array.from([1, 1, -1, -1, 1, -1, -1, 1]), y: Uint8Array.from([1, 1, 0, 0]), classes: 2 };
    const spec: NetSpec = { inputs: 2, hidden: [], outputs: 1, activation: "linear" };
    const lay = layout(spec);
    const r = rng(1);
    for (let k = 0; k < 2000; k++) {
      const p = Float64Array.from({ length: 3 }, () => 10 * (2 * r() - 1));
      expect(evaluate(spec, lay, p, data, 0, 4).loss).toBeGreaterThanOrEqual(Math.LN2 - 1e-12);
    }
  });

  it("…so on XOR data it guesses: about 50 %; a hidden layer or the x·y feature solves it", () => {
    for (const seed of [1, 2, 3]) {
      const line = planeDataset("xor", 400, 0, 0.5, XY, seed);
      const opt = { lr: 0.1, momentum: 0.9, l2: 0, batch: 16, seed };
      const flat: NetSpec = { inputs: 2, hidden: [], outputs: 1, activation: "tanh" };
      const r = train(flat, line, opt, 500);
      const test = evaluate(flat, r.lay, r.p, line, line.nTrain, line.n - line.nTrain);
      expect(test.accuracy).toBeLessThan(0.65);
      expect(test.loss).toBeGreaterThan(0.68);
      const deep: NetSpec = { ...flat, hidden: [4] };
      const d = train(deep, line, opt, 500);
      expect(evaluate(deep, d.lay, d.p, line, line.nTrain, line.n - line.nTrain).accuracy).toBeGreaterThanOrEqual(0.95);
      const xy = planeDataset("xor", 400, 0, 0.5, XY | (1 << 4), seed);
      const withXY: NetSpec = { ...flat, inputs: 3 };
      const f = train(withXY, xy, opt, 500);
      expect(evaluate(withXY, f.lay, f.p, xy, xy.nTrain, xy.n - xy.nTrain).accuracy).toBeGreaterThanOrEqual(0.95);
    }
  });

  it("…and the circle needs x² and y² (or a hidden layer)", () => {
    const flat = planeDataset("circle", 400, 0, 0.5, XY, 1);
    const spec: NetSpec = { inputs: 2, hidden: [], outputs: 1, activation: "tanh" };
    const opt = { lr: 0.1, momentum: 0.9, l2: 0, batch: 16, seed: 1 };
    const r = train(spec, flat, opt, 500);
    expect(evaluate(spec, r.lay, r.p, flat, flat.nTrain, 200).accuracy).toBeLessThan(0.7);
    const squares = planeDataset("circle", 400, 0, 0.5, XY | 4 | 8, 1);
    const s = train({ ...spec, inputs: 4 }, squares, opt, 500);
    expect(evaluate({ ...spec, inputs: 4 }, s.lay, s.p, squares, squares.nTrain, 200).accuracy).toBeGreaterThanOrEqual(0.95);
  });

  it("the step-size threshold: on a quadratic bowl, steps below 2/λmax always go downhill, above it the loss grows", () => {
    // Linear model + squared error = a quadratic bowl whose curvature is (1/n)·Σ x̃x̃ᵀ, x̃ = (features, 1).
    const data = planeDataset("blobs", 200, 0, 1 - 1 / 200, XY, 4);
    const spec: NetSpec = { inputs: 2, hidden: [], outputs: 1, activation: "linear" };
    const lay = layout(spec);
    const n = data.nTrain;
    const H = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let s = 0; s < n; s++) {
      const v = [data.x[2 * s], data.x[2 * s + 1], 1];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) H[3 * i + j] += (v[i] * v[j]) / n;
    }
    let e = [1, 1, 1];
    let lambda = 0;
    for (let k = 0; k < 200; k++) {
      const he = [0, 1, 2].map((i) => H[3 * i] * e[0] + H[3 * i + 1] * e[1] + H[3 * i + 2] * e[2]);
      lambda = Math.hypot(...he);
      e = he.map((x) => x / lambda);
    }
    const run = (lr: number) => {
      const p = initParams(spec, 2);
      const v = new Float64Array(lay.count);
      const losses = [evaluate(spec, lay, p, data, 0, n, "squared").loss];
      for (let k = 0; k < 300; k++) {
        update(p, v, gradient(spec, lay, p, data, all(n), "squared").grad, { lr, momentum: 0, l2: 0 }, weightMask(lay));
        losses.push(evaluate(spec, lay, p, data, 0, n, "squared").loss);
      }
      return losses;
    };
    const below = run((0.95 * 2) / lambda);
    for (let k = 1; k < below.length; k++) expect(below[k]).toBeLessThanOrEqual(below[k - 1] + 1e-12);
    const above = run((1.05 * 2) / lambda);
    expect(above.at(-1)!).toBeGreaterThan(10 * above[0]);
  });

  it("too big a step on a real network: η = 10 leaves it guessing (≤ 60 %) and confidently wrong (loss > ln 2)", () => {
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const data = planeDataset("circle", 400, 0, 0.5, XY, seed);
      const spec: NetSpec = { inputs: 2, hidden: [8], outputs: 1, activation: "tanh" };
      const r = train(spec, data, { lr: 10, momentum: 0.9, l2: 0, batch: 16, seed }, 300);
      const big = evaluate(spec, r.lay, r.p, data, 0, data.nTrain);
      expect(big.accuracy).toBeLessThanOrEqual(0.6);
      expect(big.loss).toBeGreaterThan(Math.LN2);
      const ok = train(spec, data, { lr: 0.1, momentum: 0.9, l2: 0, batch: 16, seed }, 600);
      expect(evaluate(spec, ok.lay, ok.p, data, 0, data.nTrain).accuracy).toBeGreaterThanOrEqual(0.95);
    }
  });

  it("memorising: 40 noisy spiral points and a big network — train accuracy near 100 %, test far behind", () => {
    const data = planeDataset("spiral", 400, 0.2, 0.1, XY, 1);
    expect(data.nTrain).toBe(40);
    const spec: NetSpec = { inputs: 2, hidden: [16, 16], outputs: 1, activation: "relu" };
    const r = train(spec, data, { lr: 0.03, momentum: 0.9, l2: 0, batch: 16, seed: 1 }, 4000);
    const tr = evaluate(spec, r.lay, r.p, data, 0, data.nTrain);
    const te = evaluate(spec, r.lay, r.p, data, data.nTrain, data.n - data.nTrain);
    expect(tr.accuracy).toBeGreaterThanOrEqual(0.95);
    expect(te.accuracy).toBeLessThan(tr.accuracy - 0.2);
    expect(te.loss).toBeGreaterThan(5 * tr.loss);
  });

  it("zero start: every hidden neuron stays a clone, nothing is learnt; a random start learns", () => {
    const data = planeDataset("xor", 400, 0, 0.5, XY, 1);
    const spec: NetSpec = { inputs: 2, hidden: [4, 4], outputs: 1, activation: "tanh" };
    const opt = { lr: 0.1, momentum: 0.9, l2: 0, batch: 16, seed: 1 };
    const z = train(spec, data, { ...opt, init: "zeros" }, 200);
    expect(neuronSpread(z.lay, z.p)).toBe(0);
    expect(evaluate(spec, z.lay, z.p, data, data.nTrain, 200).accuracy).toBeLessThan(0.6);
    const r = train(spec, data, opt, 500);
    expect(neuronSpread(r.lay, r.p)).toBeGreaterThan(0.1);
    expect(evaluate(spec, r.lay, r.p, data, data.nTrain, 200).accuracy).toBeGreaterThanOrEqual(0.95);
  });

  it("vanishing gradients: six sigmoid layers pass back under 1 % of the error to the first layer; tanh trains", () => {
    for (const seed of [1, 2, 3]) {
      const data = planeDataset("circle", 400, 0, 0.5, XY, seed);
      const deep = (activation: Activation): NetSpec => ({ inputs: 2, hidden: [4, 4, 4, 4, 4, 4], outputs: 1, activation });
      const sig = deep("sigmoid");
      const lay = layout(sig);
      const g = layerGradientSizes(lay, gradient(sig, lay, initParams(sig, seed), data, all(data.nTrain)).grad);
      expect(g[0]).toBeLessThan(0.01 * g[g.length - 1]);
      const opt = { lr: 0.03, momentum: 0.9, l2: 0, batch: 16, seed };
      const s = train(sig, data, opt, 500);
      expect(evaluate(sig, s.lay, s.p, data, data.nTrain, 200).accuracy).toBeLessThan(0.6);
      const t = train(deep("tanh"), data, opt, 500);
      expect(evaluate(deep("tanh"), t.lay, t.p, data, data.nTrain, 200).accuracy).toBeGreaterThanOrEqual(0.9);
    }
  });

  it("the default spiral is untangled by two ReLU layers of 16 within 3000 steps, for any seed", () => {
    for (const seed of [11, 12, 13, 14]) {
      const data = planeDataset("spiral", 400, 0, 0.5, XY, seed);
      const spec: NetSpec = { inputs: 2, hidden: [16, 16], outputs: 1, activation: "relu" };
      const r = train(spec, data, { lr: 0.03, momentum: 0.9, l2: 0, batch: 16, seed }, 3000);
      expect(evaluate(spec, r.lay, r.p, data, data.nTrain, 200).accuracy).toBeGreaterThanOrEqual(0.95);
    }
  });

  it("the step probe: s = 1 is the loss now, s = 0 the loss before the step", () => {
    const data = planeDataset("circle", 200, 0, 0.5, XY, 2);
    const spec: NetSpec = { inputs: 2, hidden: [4], outputs: 1, activation: "tanh" };
    const lay = layout(spec);
    const p = initParams(spec, 2);
    const before = Float64Array.from(p);
    const v = new Float64Array(lay.count);
    update(p, v, gradient(spec, lay, p, data, all(data.nTrain)).grad, { lr: 0.5, momentum: 0, l2: 0 }, weightMask(lay));
    const profile = stepProfile(spec, lay, p, v, 0.5, data, PROBE_S);
    expect(profile[8]).toBeCloseTo(evaluate(spec, lay, p, data, 0, data.nTrain).loss, 12);
    expect(profile[0]).toBeCloseTo(evaluate(spec, lay, before, data, 0, data.nTrain).loss, 12);
    expect(profile[8]).toBeLessThan(profile[0]); // a small full-batch step goes downhill
  });
});

describe("handwritten digits", () => {
  const digits = parseDigits(gunzipSync(readFileSync("public/data/mnist-12k.dat")));

  it("the shipped subset: 10 000 training and 2 000 test digits, every class present", () => {
    expect(digits.count).toBe(12000);
    expect(digits.nTrain).toBe(10000);
    const counts = new Array(10).fill(0);
    for (const l of digits.labels) counts[l]++;
    for (const c of counts) expect(c).toBeGreaterThan(900);
    // MNIST digits are centred by mass in 28×28 with an empty border.
    let edge = 0;
    for (let k = 0; k < 100; k++) for (let i = 0; i < DIGIT_SIZE; i++) edge += digits.pixels[k * DIGIT_PIXELS + i];
    expect(edge).toBe(0);
  });

  it("a drawing becomes an MNIST-style digit: 20 px tall, centred by mass", () => {
    const W = 200;
    const ink = new Float64Array(W * W);
    // A thick vertical bar, off-centre, 120 px tall.
    for (let y = 20; y < 140; y++) for (let x = 130; x < 150; x++) ink[y * W + x] = 1;
    const d = toDigit(ink, W, W)!;
    let top = DIGIT_SIZE;
    let bottom = -1;
    let m = 0;
    let cx = 0;
    let cy = 0;
    for (let y = 0; y < DIGIT_SIZE; y++)
      for (let x = 0; x < DIGIT_SIZE; x++) {
        const v = d[y * DIGIT_SIZE + x];
        if (v > 0) {
          top = Math.min(top, y);
          bottom = Math.max(bottom, y);
        }
        m += v;
        cx += v * (x + 0.5);
        cy += v * (y + 0.5);
      }
    expect(bottom - top + 1).toBe(20);
    expect(Math.abs(cx / m - 14)).toBeLessThanOrEqual(0.5); // whole-pixel shift
    expect(Math.abs(cy / m - 14)).toBeLessThanOrEqual(0.5);
    expect(toDigit(new Float64Array(16), 4, 4)).toBeNull();
  });

  it("confusion matrix counts truth × guess", () => {
    const m = confusion([1, 1, 2, 3], [1, 2, 2, 3], 4);
    expect(m[1]).toEqual([0, 1, 1, 0]);
    expect(m[2][2]).toBe(1);
    expect(m[3][3]).toBe(1);
  });

  it("a straight-line model reads about 9 digits in 10 and no better than 90 %; one hidden layer passes 92 %", () => {
    const data = digitsDataset(digits);
    const linear: NetSpec = { inputs: DIGIT_PIXELS, hidden: [], outputs: 10, activation: "relu" };
    const l = train(linear, data, { lr: 0.01, momentum: 0.9, l2: 0, batch: 32, seed: 1 }, Math.round((10 * 10000) / 32));
    const lAcc = evaluate(linear, l.lay, l.p, data, 10000, 2000).accuracy;
    expect(lAcc).toBeGreaterThan(0.85);
    expect(lAcc).toBeLessThan(0.9);
    const hidden: NetSpec = { ...linear, hidden: [128] };
    const h = train(hidden, data, { lr: 0.05, momentum: 0.9, l2: 0, batch: 32, seed: 1 }, Math.round((5 * 10000) / 32));
    expect(evaluate(hidden, h.lay, h.p, data, 10000, 2000).accuracy).toBeGreaterThan(0.92);
  }, 120_000);
});
