/**
 * The models of "How AI Writes" and how each one runs, written once as a list of operations ("ops") that both the
 * GPU (lm.wgsl, one kernel per op kind) and the 64-bit CPU twin (cpu.ts, a port of every kernel) execute. An op is
 * a fixed record of shapes, offsets and flags, the same bytes the GPU reads as its uniform, so the two backends
 * cannot drift apart: they run the same program over the same buffers.
 *
 * Buffers (all flat arrays):
 * - params: every weight. Matrices that get weight decay first, then the rest (layer norms, biases).
 * - work:   the activations of a batch, then (training) their gradients, then the evaluation losses.
 * - opt:    the gradient [0, P), Adam's first and second moments [P, 3P), the gradient-norm scratch and step values.
 * - tokens: the text (one u32 per letter), the writing area, the prompt, the window starts, the writing counter.
 *
 * Models (all predict the next letter of every position of a window of T letters):
 * - bigram: a V × V table of scores (logits) for the next letter given this one.
 * - mlp:    Bengio et al. (2003): embeddings of the last k letters, concatenated, a tanh layer, then the scores.
 * - gpt:    a transformer: letter and position embeddings, L pre-norm blocks (causal multi-head self-attention, then
 *           a GELU MLP four times as wide), a final layer norm and the scores.
 */
import { rng } from "./text";

export const MODEL_KINDS = ["bigram", "mlp", "gpt"] as const;
export type ModelKind = (typeof MODEL_KINDS)[number];

export interface ModelSpec {
  kind: ModelKind;
  /** Vocabulary size. */
  V: number;
  /** Window length: letters per training window, and the transformer's context. */
  T: number;
  /** MLP: letters of context, embedding width, hidden width. */
  k: number;
  e: number;
  h: number;
  /** Transformer: layers, heads, width. */
  L: number;
  H: number;
  d: number;
}

/** Adam's constants (nanoGPT's for character models), the clip on the gradient norm and the warm-up. */
export const ADAM = { beta1: 0.9, beta2: 0.99, eps: 1e-8, weightDecay: 0.1 } as const;
export const CLIP_NORM = 1;
export const WARMUP_STEPS = 100;
export const LN_EPS = 1e-5;
/** Partial sums of the gradient norm (a fixed-order two-stage reduction). */
export const NORM_PARTS = 256;

type Init = { kind: "normal"; std: number } | { kind: "zeros" } | { kind: "ones" } | { kind: "sinusoid"; rms: number; width: number };
interface Tensor {
  name: string;
  size: number;
  decay: boolean;
  init: Init;
}

const normal = (std: number): Init => ({ kind: "normal", std });
const zeros: Init = { kind: "zeros" };
const ones: Init = { kind: "ones" };

function tensors(s: ModelSpec): Tensor[] {
  const { V, T, k, e, h, L, d } = s;
  const t: Tensor[] = [];
  const add = (name: string, size: number, decay: boolean, init: Init) => t.push({ name, size, decay, init });
  if (s.kind === "bigram") {
    // Not decayed: weight decay would pull the table towards uniform, away from the counts it should reach.
    add("table", V * V, false, zeros);
  } else if (s.kind === "mlp") {
    add("C", V * e, true, normal(1));
    add("W1", k * e * h, true, normal(5 / 3 / Math.sqrt(k * e)));
    add("W2", h * V, true, normal(0.01));
    add("b1", h, false, zeros);
    add("b2", V, false, zeros);
  } else {
    // Matrices start at 1/√(fan-in) (PyTorch's scale) rather than nanoGPT's 0.02: a tiny model trained for a minute
    // has no time to escape the flat start that tiny attention weights give (the copying game's induction heads,
    // above all). The output projections of each block are scaled down by √(2L) so the residual stream starts calm,
    // and the head stays small so training starts at exactly the guessing loss.
    const inner = 1 / Math.sqrt(d);
    const resid = (fanIn: number) => 1 / Math.sqrt(fanIn) / Math.sqrt(2 * L);
    add("wte", V * d, true, normal(0.02));
    // Positions start as the original Transformer's sinusoids (at the same scale as a random start): "one letter
    // back" is then a rotation the attention can learn quickly, which the copying game's induction heads need.
    add("wpe", T * d, true, { kind: "sinusoid", rms: 0.02, width: d });
    for (let l = 0; l < L; l++) {
      add(`qkv${l}`, d * 3 * d, true, normal(inner));
      add(`proj${l}`, d * d, true, normal(resid(d)));
      add(`fc${l}`, d * 4 * d, true, normal(inner));
      add(`out${l}`, 4 * d * d, true, normal(resid(4 * d)));
    }
    add("head", d * V, true, normal(0.02));
    for (let l = 0; l < L; l++) {
      add(`ln1_${l}`, d, false, ones);
      add(`ln1b_${l}`, d, false, zeros);
      add(`qkvb${l}`, 3 * d, false, zeros);
      add(`projb${l}`, d, false, zeros);
      add(`ln2_${l}`, d, false, ones);
      add(`ln2b_${l}`, d, false, zeros);
      add(`fcb${l}`, 4 * d, false, zeros);
      add(`outb${l}`, d, false, zeros);
    }
    add("lnf", d, false, ones);
    add("lnfb", d, false, zeros);
    add("headb", V, false, zeros);
  }
  return t;
}

export interface Layout {
  /** Number of parameters. */
  P: number;
  /** Parameters [0, nDecay) get weight decay. */
  nDecay: number;
  /** Offset of each tensor. */
  at: Record<string, number>;
  size: Record<string, number>;
}

export function layout(s: ModelSpec): Layout {
  const at: Record<string, number> = {};
  const size: Record<string, number> = {};
  let P = 0;
  let nDecay = 0;
  for (const t of tensors(s)) {
    at[t.name] = P;
    size[t.name] = t.size;
    P += t.size;
    if (t.decay) nDecay = P;
  }
  return { P, nDecay, at, size };
}

/** Box–Muller normals from the seeded generator. */
export function gaussian(r: () => number) {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    const u = 1 - r();
    const v = r();
    const m = Math.sqrt(-2 * Math.log(u));
    spare = m * Math.sin(2 * Math.PI * v);
    return m * Math.cos(2 * Math.PI * v);
  };
}

/** A fresh model: the same numbers for the GPU and the twin from the same seed. */
export function initParams(s: ModelSpec, seed: number): Float64Array {
  const lay = layout(s);
  const p = new Float64Array(lay.P);
  const g = gaussian(rng(seed * 2654435761));
  for (const t of tensors(s)) {
    const o = lay.at[t.name];
    const init = t.init;
    if (init.kind === "sinusoid") {
      const amp = init.rms * Math.SQRT2;
      for (let i = 0; i < t.size; i++) {
        const pos = Math.floor(i / init.width);
        const c = i % init.width;
        const angle = pos / Math.pow(10000, (2 * Math.floor(c / 2)) / init.width);
        p[o + i] = amp * (c % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
      }
      continue;
    }
    for (let i = 0; i < t.size; i++) p[o + i] = init.kind === "normal" ? init.std * g() : init.kind === "ones" ? 1 : 0;
  }
  return p;
}

// ── Ops ──

export const KERNELS = ["embed", "embedBack", "layernorm", "layernormBack", "matmul", "colsum", "act", "attention", "attentionBack", "crossEntropy", "normPartial", "normFinish", "adam", "sample", "advance"] as const;
export type Kernel = (typeof KERNELS)[number];

/** Matmul flags. */
export const MM = { transA: 1, transB: 2, bInWork: 4, cInOpt: 8, bias: 16, residual: 32, accumulate: 64 } as const;
/** Act modes (in flags). */
export const ACT = { gelu: 0, geluBack: 1, tanh: 2, tanhBack: 3 } as const;

/**
 * One op: the fields of the uniform struct `Op` in lm.wgsl, in order. Their meaning depends on the kernel (see
 * each builder below and the kernels in cpu.ts / lm.wgsl).
 */
export interface Op {
  kernel: Kernel;
  M: number;
  N: number;
  K: number;
  flags: number;
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
  g: number;
  h: number;
  B: number;
  T: number;
  H: number;
  D: number;
  V: number;
  ctx: number;
  starts: number;
  gen: number;
  s0: number;
  s1: number;
  s2: number;
  s3: number;
}

/** The u32 fields of Op in struct order (then four f32s). */
export const OP_U32_FIELDS = ["M", "N", "K", "flags", "a", "b", "c", "d", "e", "f", "g", "h", "B", "T", "H", "D", "V", "ctx", "starts", "gen"] as const;
export const OP_F32_FIELDS = ["s0", "s1", "s2", "s3"] as const;

const op = (kernel: Kernel, fields: Partial<Omit<Op, "kernel">>): Op => ({ kernel, M: 0, N: 0, K: 0, flags: 0, a: 0, b: 0, c: 0, d: 0, e: 0, f: 0, g: 0, h: 0, B: 0, T: 0, H: 0, D: 0, V: 0, ctx: 0, starts: 0, gen: 0, s0: 0, s1: 0, s2: 0, s3: 0, ...fields });

/** Threads an op runs (matmul: 64 per 32 × 32 output tile). */
export function invocations(o: Op): number {
  switch (o.kernel) {
    case "embed":
      return o.B * o.T * o.N;
    case "embedBack":
      return o.V * o.K + (o.flags & 1 ? o.T * o.K : 0);
    case "layernorm":
      return o.M;
    case "layernormBack":
      return o.M + o.N;
    case "matmul":
      return 64 * Math.ceil(o.M / 32) * Math.ceil(o.N / 32);
    case "colsum":
      return o.N;
    case "act":
      return o.M;
    case "attention":
      return o.B * o.H * o.T;
    case "attentionBack":
      return 2 * o.B * o.H * o.T;
    case "crossEntropy":
      return o.B * o.T;
    case "normPartial":
      return NORM_PARTS;
    case "normFinish":
    case "advance":
      return 1;
    case "adam":
      return o.K;
    case "sample":
      return o.B;
  }
}

/** Where things live in the opt buffer. */
export function optLayout(P: number) {
  return { grad: 0, m: P, v: 2 * P, partials: 3 * P, norm: 3 * P + NORM_PARTS, scale: 3 * P + NORM_PARTS + 1, step: 3 * P + NORM_PARTS + 4, size: 3 * P + NORM_PARTS + 8 };
}

/** Where a batch's tokens come from: window starts at tokens[starts + b], shifted by tokens[gen] when gen > 0. */
export interface Source {
  starts: number;
  gen: number;
}

/** A forward pass's activations (offsets in work) that readers need. */
export interface Activations {
  logits: number;
  /** Per layer, the attention probabilities [B][H][T][T] (transformer only). */
  attention: number[];
}

export interface Program {
  ops: Op[];
  /** Floats of work the program uses (from 0). */
  work: number;
  acts: Activations;
}

/** A bump allocator over the work buffer. */
function allocator(from = 0) {
  let top = from;
  return {
    take(n: number) {
      const at = top;
      top += n;
      return at;
    },
    get top() {
      return top;
    },
  };
}

interface Forward {
  ops: Op[];
  acts: Activations;
  /** Everything backward() needs. */
  saved: Record<string, number>;
}

/** The forward pass of B windows of T letters, from token source `src`, allocating activations from `alloc`. */
function forward(s: ModelSpec, lay: Layout, B: number, T: number, src: Source, alloc: ReturnType<typeof allocator>): Forward {
  const rows = B * T;
  const ops: Op[] = [];
  const saved: Record<string, number> = {};
  const { V } = s;
  if (s.kind === "bigram") {
    const logits = alloc.take(rows * V);
    ops.push(op("embed", { B, T, N: V, K: V, ctx: 1, a: lay.at.table, c: logits, ...src }));
    return { ops, acts: { logits, attention: [] }, saved: { logits } };
  }
  if (s.kind === "mlp") {
    const { k, e, h } = s;
    const emb = alloc.take(rows * k * e);
    const pre = alloc.take(rows * h);
    const act = alloc.take(rows * h);
    const logits = alloc.take(rows * V);
    ops.push(op("embed", { B, T, N: k * e, K: e, ctx: k, a: lay.at.C, c: emb, ...src }));
    ops.push(op("matmul", { M: rows, N: h, K: k * e, a: emb, f: k * e, b: lay.at.W1, g: h, c: pre, h: h, d: lay.at.b1, flags: MM.bias }));
    ops.push(op("act", { M: rows * h, a: pre, c: act, flags: ACT.tanh }));
    ops.push(op("matmul", { M: rows, N: V, K: h, a: act, f: h, b: lay.at.W2, g: V, c: logits, h: V, d: lay.at.b2, flags: MM.bias }));
    Object.assign(saved, { emb, pre, act, logits });
    return { ops, acts: { logits, attention: [] }, saved };
  }
  const { L, H, d } = s;
  const D = d / H;
  const x0 = alloc.take(rows * d);
  ops.push(op("embed", { B, T, N: d, K: d, ctx: 1, a: lay.at.wte, b: lay.at.wpe, c: x0, flags: 1, ...src }));
  saved.x0 = x0;
  const attention: number[] = [];
  let x = x0;
  for (let l = 0; l < L; l++) {
    const y1 = alloc.take(rows * d);
    const st1 = alloc.take(rows * 2);
    const qkv = alloc.take(rows * 3 * d);
    const p = alloc.take(B * H * T * T);
    const ao = alloc.take(rows * d);
    const xm = alloc.take(rows * d);
    const y2 = alloc.take(rows * d);
    const st2 = alloc.take(rows * 2);
    const fc = alloc.take(rows * 4 * d);
    const ge = alloc.take(rows * 4 * d);
    const xo = alloc.take(rows * d);
    ops.push(op("layernorm", { M: rows, N: d, a: x, b: lay.at[`ln1_${l}`], c: y1, d: st1 }));
    ops.push(op("matmul", { M: rows, N: 3 * d, K: d, a: y1, f: d, b: lay.at[`qkv${l}`], g: 3 * d, c: qkv, h: 3 * d, d: lay.at[`qkvb${l}`], flags: MM.bias }));
    ops.push(op("attention", { B, T, H, D, N: d, a: qkv, c: ao, d: p, s0: 1 / Math.sqrt(D) }));
    ops.push(op("matmul", { M: rows, N: d, K: d, a: ao, f: d, b: lay.at[`proj${l}`], g: d, c: xm, h: d, d: lay.at[`projb${l}`], e: x, flags: MM.bias | MM.residual }));
    ops.push(op("layernorm", { M: rows, N: d, a: xm, b: lay.at[`ln2_${l}`], c: y2, d: st2 }));
    ops.push(op("matmul", { M: rows, N: 4 * d, K: d, a: y2, f: d, b: lay.at[`fc${l}`], g: 4 * d, c: fc, h: 4 * d, d: lay.at[`fcb${l}`], flags: MM.bias }));
    ops.push(op("act", { M: rows * 4 * d, a: fc, c: ge, flags: ACT.gelu }));
    ops.push(op("matmul", { M: rows, N: d, K: 4 * d, a: ge, f: 4 * d, b: lay.at[`out${l}`], g: d, c: xo, h: d, d: lay.at[`outb${l}`], e: xm, flags: MM.bias | MM.residual }));
    Object.assign(saved, { [`xin${l}`]: x, [`y1_${l}`]: y1, [`st1_${l}`]: st1, [`qkv${l}`]: qkv, [`p${l}`]: p, [`ao${l}`]: ao, [`xm${l}`]: xm, [`y2_${l}`]: y2, [`st2_${l}`]: st2, [`fc${l}`]: fc, [`ge${l}`]: ge });
    attention.push(p);
    x = xo;
  }
  const yf = alloc.take(rows * d);
  const stf = alloc.take(rows * 2);
  const logits = alloc.take(rows * V);
  ops.push(op("layernorm", { M: rows, N: d, a: x, b: lay.at.lnf, c: yf, d: stf }));
  ops.push(op("matmul", { M: rows, N: V, K: d, a: yf, f: d, b: lay.at.head, g: V, c: logits, h: V, d: lay.at.headb, flags: MM.bias }));
  Object.assign(saved, { xL: x, yf, stf, logits });
  return { ops, acts: { logits, attention }, saved };
}

/** dW = Aᵀ · dY into opt, and (if any) the bias gradient: the column sums of dY. */
function weightGrads(ops: Op[], rows: number, nIn: number, nOut: number, input: number, dOut: number, w: number, bias: number | null) {
  ops.push(op("matmul", { M: nIn, N: nOut, K: rows, a: input, f: nIn, b: dOut, g: nOut, c: w, h: nOut, flags: MM.transA | MM.bInWork | MM.cInOpt }));
  if (bias !== null) ops.push(op("colsum", { M: rows, N: nOut, a: dOut, f: nOut, c: bias, flags: MM.cInOpt }));
}

/** dX = dY · Wᵀ (into work). */
const inputGrad = (rows: number, nIn: number, nOut: number, dOut: number, w: number, dIn: number) =>
  op("matmul", { M: rows, N: nIn, K: nOut, a: dOut, f: nOut, b: w, g: nOut, c: dIn, h: nIn, flags: MM.transB });

/** The backward pass, from the gradient of the logits (already in work at dLogits). */
function backward(s: ModelSpec, lay: Layout, B: number, T: number, src: Source, f: Forward, dLogits: number, alloc: ReturnType<typeof allocator>): Op[] {
  const rows = B * T;
  const ops: Op[] = [];
  const sv = f.saved;
  const { V } = s;
  if (s.kind === "bigram") {
    ops.push(op("embedBack", { B, T, N: V, K: V, V, ctx: 1, a: lay.at.table, c: dLogits, ...src }));
    return ops;
  }
  if (s.kind === "mlp") {
    const { k, e, h } = s;
    const dAct = alloc.take(rows * h);
    const dPre = alloc.take(rows * h);
    const dEmb = alloc.take(rows * k * e);
    weightGrads(ops, rows, h, V, sv.act, dLogits, lay.at.W2, lay.at.b2);
    ops.push(inputGrad(rows, h, V, dLogits, lay.at.W2, dAct));
    ops.push(op("act", { M: rows * h, a: sv.act, e: dAct, c: dPre, flags: ACT.tanhBack }));
    weightGrads(ops, rows, k * e, h, sv.emb, dPre, lay.at.W1, lay.at.b1);
    ops.push(inputGrad(rows, k * e, h, dPre, lay.at.W1, dEmb));
    ops.push(op("embedBack", { B, T, N: k * e, K: e, V, ctx: k, a: lay.at.C, c: dEmb, ...src }));
    return ops;
  }
  const { L, H, d } = s;
  const D = d / H;
  // Head and final layer norm.
  const dyf = alloc.take(rows * d);
  weightGrads(ops, rows, d, V, sv.yf, dLogits, lay.at.head, lay.at.headb);
  ops.push(inputGrad(rows, d, V, dLogits, lay.at.head, dyf));
  let dx = alloc.take(rows * d);
  ops.push(op("layernormBack", { M: rows, N: d, a: sv.xL, b: lay.at.lnf, c: dx, d: sv.stf, e: dyf }));
  for (let l = L - 1; l >= 0; l--) {
    const dge = alloc.take(rows * 4 * d);
    const dfc = alloc.take(rows * 4 * d);
    const dy2 = alloc.take(rows * d);
    const dxm = alloc.take(rows * d);
    const dao = alloc.take(rows * d);
    const dqkv = alloc.take(rows * 3 * d);
    const dy1 = alloc.take(rows * d);
    const dxin = alloc.take(rows * d);
    // MLP block: xo = xm + gelu(ln2(xm) · Wfc + bfc) · Wout + bout.
    weightGrads(ops, rows, 4 * d, d, sv[`ge${l}`], dx, lay.at[`out${l}`], lay.at[`outb${l}`]);
    ops.push(inputGrad(rows, 4 * d, d, dx, lay.at[`out${l}`], dge));
    ops.push(op("act", { M: rows * 4 * d, a: sv[`fc${l}`], e: dge, c: dfc, flags: ACT.geluBack }));
    weightGrads(ops, rows, d, 4 * d, sv[`y2_${l}`], dfc, lay.at[`fc${l}`], lay.at[`fcb${l}`]);
    ops.push(inputGrad(rows, d, 4 * d, dfc, lay.at[`fc${l}`], dy2));
    ops.push(op("layernormBack", { M: rows, N: d, a: sv[`xm${l}`], b: lay.at[`ln2_${l}`], c: dxm, d: sv[`st2_${l}`], e: dy2, f: dx, flags: 1 }));
    // Attention block: xm = xin + attention(ln1(xin)) · Wproj + bproj.
    weightGrads(ops, rows, d, d, sv[`ao${l}`], dxm, lay.at[`proj${l}`], lay.at[`projb${l}`]);
    ops.push(inputGrad(rows, d, d, dxm, lay.at[`proj${l}`], dao));
    ops.push(op("attentionBack", { B, T, H, D, N: d, a: sv[`qkv${l}`], b: sv[`ao${l}`], c: dqkv, d: sv[`p${l}`], e: dao, s0: 1 / Math.sqrt(D) }));
    weightGrads(ops, rows, d, 3 * d, sv[`y1_${l}`], dqkv, lay.at[`qkv${l}`], lay.at[`qkvb${l}`]);
    ops.push(inputGrad(rows, d, 3 * d, dqkv, lay.at[`qkv${l}`], dy1));
    ops.push(op("layernormBack", { M: rows, N: d, a: sv[`xin${l}`], b: lay.at[`ln1_${l}`], c: dxin, d: sv[`st1_${l}`], e: dy1, f: dxm, flags: 1 }));
    dx = dxin;
  }
  ops.push(op("embedBack", { B, T, N: d, K: d, V, ctx: 1, a: lay.at.wte, b: lay.at.wpe, c: dx, flags: 1, ...src }));
  return ops;
}

/** The cross-entropy op: per-row loss to `loss`; with dLogits, also its gradient (the batch mean). */
const crossEntropy = (B: number, T: number, V: number, logits: number, loss: number, dLogits: number | null, src: Source) =>
  op("crossEntropy", { B, T, V, a: logits, c: loss, e: dLogits ?? 0, flags: dLogits === null ? 0 : 1, s0: 1 / (B * T), ...src });

/**
 * One training step: forward, loss and its gradient, backward, the clipped global gradient norm and AdamW. The
 * step's learning rate and bias corrections are values in opt (written before each step, see stepValues()).
 */
export function trainProgram(s: ModelSpec, lay: Layout, B: number, src: Source): Program {
  const T = s.T;
  const alloc = allocator();
  const f = forward(s, lay, B, T, src, alloc);
  const loss = alloc.take(B * T);
  const dLogits = alloc.take(B * T * s.V);
  const ops = [...f.ops, crossEntropy(B, T, s.V, f.acts.logits, loss, dLogits, src), ...backward(s, lay, B, T, src, f, dLogits, alloc)];
  const o = optLayout(lay.P);
  ops.push(op("normPartial", { K: lay.P, a: o.grad, c: o.partials }));
  ops.push(op("normFinish", { c: o.partials, s0: CLIP_NORM }));
  ops.push(op("adam", { K: lay.P, M: lay.nDecay, a: o.grad, b: o.m, c: o.v, d: o.scale, e: o.step, s0: ADAM.beta1, s1: ADAM.beta2, s2: ADAM.eps }));
  return { ops, work: alloc.top, acts: f.acts };
}

/**
 * A forward pass of B windows of T letters (T ≤ the model's context) and, when `loss` is given, the per-row loss
 * written to work[loss …] (outside the program's own work).
 */
export function forwardProgram(s: ModelSpec, lay: Layout, B: number, T: number, src: Source, loss: number | null): Program {
  const alloc = allocator();
  const f = forward(s, lay, B, T, src, alloc);
  const ops = [...f.ops];
  if (loss !== null) ops.push(crossEntropy(B, T, s.V, f.acts.logits, loss, null, src));
  return { ops, work: alloc.top, acts: f.acts };
}

/**
 * Writing one letter on each of B lines: a forward pass over each line's last T letters (the window starts at
 * tokens[starts + b] + tokens[gen]), a sample from the last position's scores at `temperature`, written after the
 * window, then the counter at tokens[gen] moves on. Run it n times to write n letters without leaving the GPU. With a
 * stride, the model's own surprise at each letter it picked (−ln p at temperature 1) goes to work[surprise + b·stride + n].
 */
export function writeProgram(s: ModelSpec, lay: Layout, B: number, src: Source, temperature: number, seed: number, surprise = 0, stride = 0): Program {
  if (src.gen <= 0) throw new Error("writing needs a counter slot");
  const alloc = allocator();
  const f = forward(s, lay, B, s.T, src, alloc);
  const ops = [...f.ops, op("sample", { B, T: s.T, V: s.V, a: f.acts.logits, K: seed >>> 0, s0: temperature, c: surprise, h: stride, flags: stride > 0 ? 1 : 0, ...src }), op("advance", { gen: src.gen })];
  return { ops, work: alloc.top, acts: f.acts };
}

/** The per-step values the Adam op reads from opt[step …]: learning rate (with warm-up), bias corrections, decay. */
export function stepValues(step: number, lr: number): [number, number, number, number] {
  const t = step + 1;
  const warm = Math.min(1, t / WARMUP_STEPS);
  return [lr * warm, 1 - Math.pow(ADAM.beta1, t), 1 - Math.pow(ADAM.beta2, t), ADAM.weightDecay];
}

/** Parameters of a spec, without building it. */
export const parameterCount = (s: ModelSpec) => layout(s).P;

/** Floats of work a training step needs. */
export const trainWork = (s: ModelSpec, B: number) => trainProgram(s, layout(s), B, { starts: 0, gen: 0 }).work;
