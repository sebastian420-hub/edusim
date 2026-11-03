/**
 * The 64-bit CPU twin of lm.wgsl: every kernel, line for line, in the same summation order, over the same buffers
 * (Float64Array instead of f32). Programs from model.ts run here unchanged, so the GPU can be checked op by op and
 * step by step, and every claim of the simulation can be tested exactly.
 */
import { LN_EPS, NORM_PARTS } from "./model";
import type { Op } from "./model";

export interface Buffers {
  params: Float64Array;
  work: Float64Array;
  opt: Float64Array;
  tokens: Uint32Array;
}

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
/** tanh of a clamped argument (the GPU clamps so its tanh cannot overflow). */
export const safeTanh = (x: number) => Math.tanh(Math.min(15, Math.max(-15, x)));

export const gelu = (x: number) => 0.5 * x * (1 + safeTanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
export function geluSlope(x: number) {
  const t = safeTanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x));
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * SQRT_2_OVER_PI * (1 + 3 * 0.044715 * x * x);
}

/** PCG hash (u32 → u32), the same integer arithmetic as the shader. */
export function pcg(v: number): number {
  const s = (Math.imul(v >>> 0, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}

/** The writing kernel's uniform number in [0, 1) for line b at counter position n: 24 bits, exact in f32. */
export const sampleUniform = (seed: number, b: number, n: number) => (pcg((seed ^ pcg((b ^ pcg(n >>> 0)) >>> 0)) >>> 0) >>> 8) / 16777216;

const windowStart = (o: Op, t: Uint32Array, b: number) => t[o.starts + b] + (o.gen > 0 ? t[o.gen] : 0);

function embed(o: Op, { params, work, tokens }: Buffers) {
  for (let b = 0; b < o.B; b++) {
    const s = windowStart(o, tokens, b);
    for (let t = 0; t < o.T; t++) {
      const row = o.c + (b * o.T + t) * o.N;
      for (let j = 0; j < o.ctx; j++) {
        const tok = tokens[s + t - j];
        for (let i = 0; i < o.K; i++) {
          let v = params[o.a + tok * o.K + i];
          if (o.flags & 1 && j === 0) v += params[o.b + t * o.K + i];
          work[row + j * o.K + i] = v;
        }
      }
    }
  }
}

function embedBack(o: Op, { work, opt, tokens }: Buffers) {
  const starts = Array.from({ length: o.B }, (_, b) => windowStart(o, tokens, b));
  for (let v = 0; v < o.V; v++) {
    for (let i = 0; i < o.K; i++) {
      let sum = 0;
      for (let b = 0; b < o.B; b++) {
        for (let t = 0; t < o.T; t++) {
          for (let j = 0; j < o.ctx; j++) {
            if (tokens[starts[b] + t - j] === v) sum += work[o.c + (b * o.T + t) * o.N + j * o.K + i];
          }
        }
      }
      opt[o.a + v * o.K + i] = sum;
    }
  }
  if (o.flags & 1) {
    for (let t = 0; t < o.T; t++) {
      for (let i = 0; i < o.K; i++) {
        let sum = 0;
        for (let b = 0; b < o.B; b++) sum += work[o.c + (b * o.T + t) * o.N + i];
        opt[o.b + t * o.K + i] = sum;
      }
    }
  }
}

function layernorm(o: Op, { params, work }: Buffers) {
  const N = o.N;
  for (let r = 0; r < o.M; r++) {
    const x = o.a + r * N;
    let mean = 0;
    for (let i = 0; i < N; i++) mean += work[x + i];
    mean /= N;
    let v = 0;
    for (let i = 0; i < N; i++) v += (work[x + i] - mean) * (work[x + i] - mean);
    const rstd = 1 / Math.sqrt(v / N + LN_EPS);
    for (let i = 0; i < N; i++) work[o.c + r * N + i] = (work[x + i] - mean) * rstd * params[o.b + i] + params[o.b + N + i];
    work[o.d + 2 * r] = mean;
    work[o.d + 2 * r + 1] = rstd;
  }
}

function layernormBack(o: Op, { params, work, opt }: Buffers) {
  const N = o.N;
  for (let r = 0; r < o.M; r++) {
    const mean = work[o.d + 2 * r];
    const rstd = work[o.d + 2 * r + 1];
    let m1 = 0;
    let m2 = 0;
    for (let i = 0; i < N; i++) {
      const dyg = work[o.e + r * N + i] * params[o.b + i];
      m1 += dyg;
      m2 += dyg * (work[o.a + r * N + i] - mean) * rstd;
    }
    m1 /= N;
    m2 /= N;
    for (let i = 0; i < N; i++) {
      const xhat = (work[o.a + r * N + i] - mean) * rstd;
      const dyg = work[o.e + r * N + i] * params[o.b + i];
      let dx = rstd * (dyg - m1 - xhat * m2);
      if (o.flags & 1) dx += work[o.f + r * N + i];
      work[o.c + r * N + i] = dx;
    }
  }
  for (let i = 0; i < N; i++) {
    let dg = 0;
    let db = 0;
    for (let r = 0; r < o.M; r++) {
      const dy = work[o.e + r * N + i];
      dg += dy * (work[o.a + r * N + i] - work[o.d + 2 * r]) * work[o.d + 2 * r + 1];
      db += dy;
    }
    opt[o.b + i] = dg;
    opt[o.b + N + i] = db;
  }
}

function matmul(o: Op, { params, work, opt }: Buffers) {
  const transA = o.flags & 1;
  const transB = o.flags & 2;
  const B = o.flags & 4 ? work : params;
  const C = o.flags & 8 ? opt : work;
  for (let m = 0; m < o.M; m++) {
    for (let n = 0; n < o.N; n++) {
      let s = 0;
      for (let k = 0; k < o.K; k++) {
        const a = transA ? work[o.a + k * o.f + m] : work[o.a + m * o.f + k];
        const b = transB ? B[o.b + n * o.g + k] : B[o.b + k * o.g + n];
        s += a * b;
      }
      if (o.flags & 16) s += params[o.d + n];
      if (o.flags & 32) s += work[o.e + m * o.h + n];
      const at = o.c + m * o.h + n;
      C[at] = o.flags & 64 ? C[at] + s : s;
    }
  }
}

function colsum(o: Op, { work, opt }: Buffers) {
  const C = o.flags & 8 ? opt : work;
  for (let n = 0; n < o.N; n++) {
    let s = 0;
    for (let m = 0; m < o.M; m++) s += work[o.a + m * o.f + n];
    C[o.c + n] = o.flags & 64 ? C[o.c + n] + s : s;
  }
}

function act(o: Op, { work }: Buffers) {
  for (let i = 0; i < o.M; i++) {
    const x = work[o.a + i];
    switch (o.flags) {
      case 0:
        work[o.c + i] = gelu(x);
        break;
      case 1:
        work[o.c + i] = work[o.e + i] * geluSlope(x);
        break;
      case 2:
        work[o.c + i] = safeTanh(x);
        break;
      default:
        work[o.c + i] = work[o.e + i] * (1 - x * x);
    }
  }
}

function attention(o: Op, { work }: Buffers) {
  const { B, T, H, D, N } = o;
  const s = new Float64Array(T);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < T; i++) {
        const q = o.a + (b * T + i) * 3 * N + h * D;
        let mx = -Infinity;
        for (let j = 0; j <= i; j++) {
          const k = o.a + (b * T + j) * 3 * N + N + h * D;
          let dot = 0;
          for (let c = 0; c < D; c++) dot += work[q + c] * work[k + c];
          s[j] = dot * o.s0;
          mx = Math.max(mx, s[j]);
        }
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          s[j] = Math.exp(s[j] - mx);
          sum += s[j];
        }
        const p = o.d + ((b * H + h) * T + i) * T;
        for (let j = 0; j < T; j++) work[p + j] = j <= i ? s[j] / sum : 0;
        for (let c = 0; c < D; c++) {
          let acc = 0;
          for (let j = 0; j <= i; j++) acc += work[p + j] * work[o.a + (b * T + j) * 3 * N + 2 * N + h * D + c];
          work[o.c + (b * T + i) * N + h * D + c] = acc;
        }
      }
    }
  }
}

function attentionBack(o: Op, { work }: Buffers) {
  const { B, T, H, D, N } = o;
  const qkv = (b: number, t: number, part: number, h: number) => o.a + (b * T + t) * 3 * N + part * N + h * D;
  const dO = (b: number, t: number, h: number) => o.e + (b * T + t) * N + h * D;
  const out = (b: number, t: number, h: number) => o.b + (b * T + t) * N + h * D;
  const P = (b: number, h: number, i: number, j: number) => work[o.d + ((b * H + h) * T + i) * T + j];
  const dot = (x: number, y: number) => {
    let s = 0;
    for (let c = 0; c < D; c++) s += work[x + c] * work[y + c];
    return s;
  };
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      // Queries.
      for (let i = 0; i < T; i++) {
        const Di = dot(dO(b, i, h), out(b, i, h));
        const dq = new Float64Array(D);
        for (let j = 0; j <= i; j++) {
          const ds = P(b, h, i, j) * (dot(dO(b, i, h), qkv(b, j, 2, h)) - Di);
          for (let c = 0; c < D; c++) dq[c] += ds * work[qkv(b, j, 1, h) + c];
        }
        for (let c = 0; c < D; c++) work[o.c + (b * T + i) * 3 * N + h * D + c] = dq[c] * o.s0;
      }
      // Keys and values.
      for (let j = 0; j < T; j++) {
        const dk = new Float64Array(D);
        const dv = new Float64Array(D);
        for (let i = j; i < T; i++) {
          const Di = dot(dO(b, i, h), out(b, i, h));
          const p = P(b, h, i, j);
          const ds = p * (dot(dO(b, i, h), qkv(b, j, 2, h)) - Di);
          for (let c = 0; c < D; c++) {
            dk[c] += ds * work[qkv(b, i, 0, h) + c];
            dv[c] += p * work[dO(b, i, h) + c];
          }
        }
        for (let c = 0; c < D; c++) {
          work[o.c + (b * T + j) * 3 * N + N + h * D + c] = dk[c] * o.s0;
          work[o.c + (b * T + j) * 3 * N + 2 * N + h * D + c] = dv[c];
        }
      }
    }
  }
}

function crossEntropy(o: Op, { work, tokens }: Buffers) {
  const { V } = o;
  for (let b = 0; b < o.B; b++) {
    const s = windowStart(o, tokens, b);
    for (let t = 0; t < o.T; t++) {
      const r = b * o.T + t;
      const row = o.a + r * V;
      const target = tokens[s + t + 1];
      let mx = -Infinity;
      for (let v = 0; v < V; v++) mx = Math.max(mx, work[row + v]);
      let sum = 0;
      for (let v = 0; v < V; v++) sum += Math.exp(work[row + v] - mx);
      const lse = mx + Math.log(sum);
      work[o.c + r] = lse - work[row + target];
      if (o.flags & 1) {
        for (let v = 0; v < V; v++) work[o.e + r * V + v] = (Math.exp(work[row + v] - lse) - (v === target ? 1 : 0)) * o.s0;
      }
    }
  }
}

function normPartial(o: Op, { opt }: Buffers) {
  const chunk = Math.ceil(o.K / NORM_PARTS);
  for (let i = 0; i < NORM_PARTS; i++) {
    let s = 0;
    for (let p = i * chunk; p < Math.min(o.K, (i + 1) * chunk); p++) s += opt[o.a + p] * opt[o.a + p];
    opt[o.c + i] = s;
  }
}

function normFinish(o: Op, { opt }: Buffers) {
  let s = 0;
  for (let i = 0; i < NORM_PARTS; i++) s += opt[o.c + i];
  const norm = Math.sqrt(s);
  opt[o.c + NORM_PARTS] = norm;
  opt[o.c + NORM_PARTS + 1] = Math.min(1, o.s0 / (norm + 1e-6));
}

function adam(o: Op, { params, opt }: Buffers) {
  const scale = opt[o.d];
  const [lr, bc1, bc2, wd] = [opt[o.e], opt[o.e + 1], opt[o.e + 2], opt[o.e + 3]];
  for (let p = 0; p < o.K; p++) {
    const g = opt[o.a + p] * scale;
    const m = o.s0 * opt[o.b + p] + (1 - o.s0) * g;
    const v = o.s1 * opt[o.c + p] + (1 - o.s1) * g * g;
    opt[o.b + p] = m;
    opt[o.c + p] = v;
    let w = params[p];
    if (p < o.M) w *= 1 - lr * wd;
    params[p] = w - (lr * (m / bc1)) / (Math.sqrt(v / bc2) + o.s2);
  }
}

/** The next letter from a row of scores: the most likely at temperature 0, else a draw from softmax(scores / T). */
export function pickLetter(scores: ArrayLike<number>, from: number, V: number, temperature: number, u: number): number {
  let mx = -Infinity;
  let best = 0;
  for (let v = 0; v < V; v++) {
    if (scores[from + v] > mx) {
      mx = scores[from + v];
      best = v;
    }
  }
  if (temperature <= 0) return best;
  let total = 0;
  for (let v = 0; v < V; v++) total += Math.exp((scores[from + v] - mx) / temperature);
  const target = u * total;
  let cum = 0;
  for (let v = 0; v < V; v++) {
    cum += Math.exp((scores[from + v] - mx) / temperature);
    if (cum > target) return v;
  }
  return V - 1;
}

function sample(o: Op, { work, tokens }: Buffers) {
  const n = o.gen > 0 ? tokens[o.gen] : 0;
  for (let b = 0; b < o.B; b++) {
    const s = windowStart(o, tokens, b);
    const row = o.a + (b * o.T + o.T - 1) * o.V;
    const pick = pickLetter(work, row, o.V, o.s0, sampleUniform(o.K, b, n));
    tokens[s + o.T] = pick;
    if (o.flags & 1 && n < o.h) {
      let mx = -Infinity;
      for (let v = 0; v < o.V; v++) mx = Math.max(mx, work[row + v]);
      let sum = 0;
      for (let v = 0; v < o.V; v++) sum += Math.exp(work[row + v] - mx);
      work[o.c + b * o.h + n] = mx + Math.log(sum) - work[row + pick];
    }
  }
}

const KERNEL_IMPL: Record<Op["kernel"], (o: Op, b: Buffers) => void> = {
  embed,
  embedBack,
  layernorm,
  layernormBack,
  matmul,
  colsum,
  act,
  attention,
  attentionBack,
  crossEntropy,
  normPartial,
  normFinish,
  adam,
  sample,
  advance: (o, { tokens }) => {
    tokens[o.gen] += 1;
  },
};

export function runOp(o: Op, buffers: Buffers) {
  KERNEL_IMPL[o.kernel](o, buffers);
}

export function runOps(ops: readonly Op[], buffers: Buffers) {
  for (const o of ops) runOp(o, buffers);
}
