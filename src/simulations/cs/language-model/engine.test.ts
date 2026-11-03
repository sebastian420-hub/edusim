import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { tryInitGpu } from "@/test/gpu";
import type { NodeGpu } from "@/test/gpu";
import { runOp } from "./cpu";
import { createEngine } from "./engine";
import { gaussian, initParams } from "./model";
import type { ModelSpec, Op } from "./model";
import { createTwin } from "./twin";
import { batchStream, buildCorpus, CONTEXT_MARGIN, rng } from "./text";

let gpu: NodeGpu | null = null;
beforeAll(async () => {
  gpu = await tryInitGpu();
});
afterAll(() => gpu?.dispose());

const needsGpu = (name: string, fn: (gpu: NodeGpu) => Promise<void>, timeout?: number) =>
  it(
    name,
    async (ctx) => {
      if (!gpu) return ctx.skip();
      await fn(gpu);
    },
    timeout,
  );

// Shapes that are not multiples of anything: 11 letters, width 12 in 3 heads of 4, windows of 7.
const base: ModelSpec = { kind: "gpt", V: 11, T: 7, k: 3, e: 5, h: 9, L: 2, H: 3, d: 12 };
const SPECS: [string, ModelSpec][] = [
  ["neural bigram", { ...base, kind: "bigram" }],
  ["MLP", { ...base, kind: "mlp" }],
  ["transformer", base],
];
const OPTS = { B: 3, G: 2, W: 6, E: 4 };

function randomText(n: number, V: number, seed: number) {
  const r = rng(seed);
  return Uint8Array.from({ length: n }, () => Math.floor(r() * V));
}

function lively(spec: ModelSpec, seed: number) {
  const p = initParams(spec, seed);
  const g = gaussian(rng(seed + 99));
  for (let i = 0; i < p.length; i++) p[i] += 0.3 * g();
  return p;
}

/** Largest difference relative to the largest value of the reference. */
function relErr(gpuValues: ArrayLike<number>, twin: ArrayLike<number>) {
  let scale = 1e-6;
  let worst = 0;
  for (let i = 0; i < twin.length; i++) scale = Math.max(scale, Math.abs(twin[i]));
  for (let i = 0; i < twin.length; i++) worst = Math.max(worst, Math.abs(gpuValues[i] - twin[i]));
  return worst / scale;
}

describe("language-model kernels vs the CPU twin", () => {
  for (const [name, spec] of SPECS) {
    needsGpu(`${name}: every op of a training step, from identical inputs`, async (gpu) => {
      const text = randomText(400, spec.V, 1);
      const twin = createTwin(spec, text, OPTS);
      const engine = createEngine(gpu, spec, text, OPTS);
      twin.load(lively(spec, 2));
      twin.buf.tokens.set([40, 133, 271], twin.plan.tokens.startsTrain);
      twin.buf.opt.set([3e-3, 0.1, 0.01, 0.1], twin.plan.opt.step);
      const kinds = new Set<string>();
      for (const op of twin.plan.train.ops as Op[]) {
        engine.writeBuffer("params", Float32Array.from(twin.buf.params));
        engine.writeBuffer("work", Float32Array.from(twin.buf.work));
        engine.writeBuffer("opt", Float32Array.from(twin.buf.opt));
        engine.writeBuffer("tokens", Uint32Array.from(twin.buf.tokens));
        engine.runOps([op]);
        runOp(op, twin.buf);
        const [p, w, o] = await Promise.all([engine.readParams(), engine.readWork(0, twin.plan.work.size), engine.readOpt()]);
        expect(relErr(w, twin.buf.work), `${op.kernel} (work)`).toBeLessThan(2e-5);
        expect(relErr(o, twin.buf.opt), `${op.kernel} (opt)`).toBeLessThan(2e-5);
        expect(relErr(p, twin.buf.params), `${op.kernel} (params)`).toBeLessThan(2e-5);
        kinds.add(op.kernel);
      }
      if (spec.kind === "gpt") expect([...kinds].sort()).toEqual(["act", "adam", "attention", "attentionBack", "colsum", "crossEntropy", "embed", "embedBack", "layernorm", "layernormBack", "matmul", "normFinish", "normPartial"]);
      engine.dispose();
    });
  }

  needsGpu("the matrix product, in every combination of transposes, sources and accumulation", async (gpu) => {
    const spec = base;
    const text = randomText(400, spec.V, 3);
    const twin = createTwin(spec, text, OPTS);
    const engine = createEngine(gpu, spec, text, OPTS);
    const r = rng(9);
    const P = twin.plan.lay.P;
    const W = twin.plan.work.size;
    for (let i = 0; i < P; i++) twin.buf.params[i] = r() - 0.5;
    for (let i = 0; i < W; i++) twin.buf.work[i] = r() - 0.5;
    for (let i = 0; i < twin.buf.opt.length; i++) twin.buf.opt[i] = r() - 0.5;
    const [M, N, K] = [37, 45, 70];
    for (let flags = 0; flags < 128; flags++) {
      const transA = flags & 1;
      const transB = flags & 2;
      const op: Op = { kernel: "matmul", M, N, K, flags, a: 0, f: transA ? M : K, b: 5, g: transB ? K : N, c: 4000, h: N, d: 7, e: 2000, B: 0, T: 0, H: 0, D: 0, V: 0, ctx: 0, starts: 0, gen: 0, s0: 0, s1: 0, s2: 0, s3: 0 };
      engine.writeBuffer("params", Float32Array.from(twin.buf.params));
      engine.writeBuffer("work", Float32Array.from(twin.buf.work));
      engine.writeBuffer("opt", Float32Array.from(twin.buf.opt));
      engine.runOps([op]);
      runOp(op, twin.buf);
      const out = flags & 8 ? await engine.readOpt() : await engine.readWork(0, W);
      const ref = flags & 8 ? twin.buf.opt : twin.buf.work;
      expect(relErr(out.subarray(4000, 4000 + M * N), ref.subarray(4000, 4000 + M * N)), `flags ${flags}`).toBeLessThan(2e-6);
    }
    engine.dispose();
  });

  for (const [name, spec] of SPECS) {
    needsGpu(`${name}: 20 training steps, evaluation and prediction agree with the twin`, async (gpu) => {
      const corpus = buildCorpus("own", Array.from(randomText(3000, spec.V, 4), (t) => "abcdefghijk"[t]).join(""));
      const twin = createTwin(spec, corpus.tokens, OPTS);
      const engine = createEngine(gpu, spec, corpus.tokens, OPTS);
      const p0 = lively(spec, 5);
      twin.load(p0);
      engine.load(p0);
      const stream = batchStream(corpus, spec.T, OPTS.B, 1);
      for (let k = 0; k < 20; k++) {
        const starts = stream.next();
        twin.step(starts, 3e-3);
        engine.step(starts, 3e-3);
      }
      expect(relErr(await engine.readParams(), twin.buf.params)).toBeLessThan(1e-3);
      const ev = [30, 400, 1200, 2500];
      expect(relErr(await engine.evaluate(ev), twin.evaluate(ev))).toBeLessThan(1e-3);
      const prompt = [1, 4, 1, 5, 9, 2, 6, 5, 3];
      const a = await engine.predict(prompt, 0);
      const b = twin.predict(prompt, 0);
      expect(a.T).toBe(b.T);
      expect(relErr(a.logits, b.logits)).toBeLessThan(1e-3);
      a.attention.forEach((att, l) => expect(relErr(att, b.attention[l])).toBeLessThan(1e-3));
      engine.dispose();
    });
  }

  needsGpu("prediction read-backs survive training queued right after them", async (gpu) => {
    const spec = base;
    const text = randomText(400, spec.V, 8);
    const twin = createTwin(spec, text, OPTS);
    const engine = createEngine(gpu, spec, text, OPTS);
    const p = lively(spec, 9);
    twin.load(p);
    engine.load(p);
    const prompt = [2, 7, 1, 8, 2, 8, 1];
    const pending = engine.predict(prompt, 0);
    // A training step reuses the same working memory as the prediction.
    engine.step([40, 133, 271], 1e-3);
    const a = await pending;
    const b = twin.predict(prompt, 0);
    a.attention.forEach((att, l) => expect(relErr(att, b.attention[l])).toBeLessThan(1e-3));
    expect(relErr(a.logits, b.logits)).toBeLessThan(1e-3);
    engine.dispose();
  });

  for (const temperature of [0, 1]) {
    needsGpu(`writing at temperature ${temperature} gives the twin's letters`, async (gpu) => {
      const spec = base;
      const text = randomText(400, spec.V, 6);
      const twin = createTwin(spec, text, OPTS);
      const engine = createEngine(gpu, spec, text, OPTS);
      const p = lively(spec, 7);
      twin.load(p);
      engine.load(p);
      const contexts = [0, 1].map((g) => text.subarray(50 + 90 * g, 50 + 90 * g + CONTEXT_MARGIN + spec.T));
      twin.startWriting(contexts, temperature, 13);
      engine.startWriting(contexts, temperature, 13);
      twin.write(4);
      engine.write(2);
      engine.write(2);
      const g = await engine.written();
      const t = twin.written();
      expect(g.map((l) => Array.from(l))).toEqual(t.map((l) => Array.from(l)));
      expect(g[0].length).toBe(4);
      const sg = await engine.surprise();
      const st = twin.surprise();
      sg.forEach((line, k) => expect(relErr(line.subarray(0, 4), st[k])).toBeLessThan(1e-4));
      expect(st[0].every((x) => x > 0)).toBe(true);
      engine.dispose();
    });
  }

  needsGpu(
    "throughput of the default transformer (logged)",
    async (gpu) => {
      const spec: ModelSpec = { kind: "gpt", V: 65, T: 64, k: 8, e: 16, h: 128, L: 2, H: 4, d: 64 };
      const corpus = buildCorpus("own", Array.from(randomText(20000, 26, 8), (t) => String.fromCharCode(97 + t)).join(""));
      const engine = createEngine(gpu, { ...spec, V: corpus.vocab.length }, corpus.tokens, { B: 16, G: 8, W: 64, E: 16 });
      engine.load(initParams({ ...spec, V: corpus.vocab.length }, 1));
      const stream = batchStream(corpus, spec.T, 16, 1);
      engine.step(stream.next(), 1e-3);
      await engine.readOpt();
      const t0 = performance.now();
      for (let k = 0; k < 5; k++) engine.step(stream.next(), 1e-3);
      await engine.readOpt();
      console.log(`default transformer: ${((performance.now() - t0) / 5).toFixed(0)} ms per step on this device`);
      engine.dispose();
    },
    120_000,
  );
});
