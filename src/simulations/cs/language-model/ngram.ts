/**
 * Count tables ("n-gram models"): the probability of the next letter given the last k letters, read straight from
 * how often each continuation follows each context in the training text. Exact, on the CPU.
 *
 * - The training-set loss of the raw counts is the lowest loss *any* model that sees only the last k letters can
 *   reach on that text: the counts are the maximum-likelihood answer.
 * - On held-out text, unseen continuations need some probability: add-α smoothing,
 *   p(next | context) = (count(context, next) + α) / (count(context) + α V), with α picked from a grid as the one
 *   that does best (so the "best count table" line is the best this kind of model can do).
 */
import type { Corpus } from "./text";
import { rng } from "./text";

export const MAX_K = 6;
export const ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1] as const;

export interface TableResult {
  /** Letters of context. */
  k: number;
  /** Loss (nats per letter) of the raw counts on the text they were counted on. */
  trainLoss: number;
  /** Loss on the held-out text with the best α (on the training text itself when there is no held-out part). */
  heldOutLoss: number;
  alpha: number;
  /** Distinct contexts seen in the training text. */
  contexts: number;
  /** Share of held-out letters whose context never occurs in the training text. */
  unseen: number;
  /** Text written by the table (sampled from the raw counts). */
  sample: string;
  /** Loss with the best α on chosen letters (the simulation's evaluation letters), when asked for. */
  rowsLoss?: number;
}

interface Table {
  k: number;
  V: number;
  pairs: Map<number, number>;
  totals: Map<number, number>;
}

const contextAt = (tokens: Uint8Array, i: number, k: number, V: number) => {
  let ctx = 0;
  for (let j = i - k; j < i; j++) ctx = ctx * V + tokens[j];
  return ctx;
};

/** Counts every (context, next) pair with a target in [from, to) (targets need k letters before them). */
function count(tokens: Uint8Array, V: number, k: number, from: number, to: number): Table {
  const pairs = new Map<number, number>();
  const totals = new Map<number, number>();
  const start = Math.max(from, k);
  const top = Math.pow(V, Math.max(0, k - 1));
  let ctx = start >= k ? contextAt(tokens, start, k, V) : 0;
  for (let i = start; i < to; i++) {
    const key = ctx * V + tokens[i];
    pairs.set(key, (pairs.get(key) ?? 0) + 1);
    totals.set(ctx, (totals.get(ctx) ?? 0) + 1);
    if (k > 0) ctx = (ctx % top) * V + tokens[i];
  }
  return { k, V, pairs, totals };
}

/** Mean −ln p over targets in [from, to): raw counts when alpha = 0 (every pair must have been seen). */
function lossOf(t: Table, tokens: Uint8Array, from: number, to: number, alpha: number): { loss: number; unseen: number } {
  const { k, V } = t;
  const start = Math.max(from, k);
  const top = Math.pow(V, Math.max(0, k - 1));
  let ctx = contextAt(tokens, start, k, V);
  let sum = 0;
  let unseen = 0;
  for (let i = start; i < to; i++) {
    const c = t.pairs.get(ctx * V + tokens[i]) ?? 0;
    const total = t.totals.get(ctx) ?? 0;
    if (!total) unseen++;
    sum -= Math.log((c + alpha) / (total + alpha * V));
    if (k > 0) ctx = (ctx % top) * V + tokens[i];
  }
  const n = Math.max(1, to - start);
  return { loss: sum / n, unseen: unseen / n };
}

/** Samples `length` letters from the raw counts, starting after a line break when the text has one. */
function sampleFrom(t: Table, c: Corpus, length: number, seed: number): string {
  const { k, V } = t;
  const r = rng(seed);
  const nl = c.vocab.indexOf("\n");
  // Start from the k letters before some line start in the training text.
  let at = k;
  for (let i = Math.max(k, Math.floor(c.nTrain * 0.37)); i < c.nTrain; i++) {
    if (nl < 0 || c.tokens[i - 1] === nl) {
      at = i;
      break;
    }
  }
  const history = Array.from(c.tokens.subarray(at - k, at));
  let out = "";
  for (let n = 0; n < length; n++) {
    const ctx = history.slice(history.length - k).reduce((a, s) => a * V + s, 0);
    const total = t.totals.get(ctx) ?? 0;
    let next = Math.floor(r() * V);
    if (total) {
      let u = r() * total;
      for (let v = 0; v < V; v++) {
        u -= t.pairs.get(ctx * V + v) ?? 0;
        if (u < 0) {
          next = v;
          break;
        }
      }
    }
    history.push(next);
    out += c.vocab[next];
  }
  return out;
}

/** Mean −ln p (add-α) of the letters at the given positions (each needs k letters before it). */
function lossAt(t: Table, tokens: Uint8Array, targets: ArrayLike<number>, alpha: number): number {
  const { k, V } = t;
  let sum = 0;
  for (let n = 0; n < targets.length; n++) {
    const i = targets[n];
    const ctx = contextAt(tokens, i, k, V);
    const c = t.pairs.get(ctx * V + tokens[i]) ?? 0;
    sum -= Math.log((c + alpha) / ((t.totals.get(ctx) ?? 0) + alpha * V));
  }
  return targets.length ? sum / targets.length : 0;
}

/** The table for k letters of context: its losses, contexts and a sample (and its loss on `targets`, if given). */
export function tableResult(c: Corpus, k: number, sampleLength = 240, seed = 1, targets?: ArrayLike<number>): TableResult {
  const V = c.vocab.length;
  const t = count(c.tokens, V, k, 0, c.nTrain);
  const trainLoss = lossOf(t, c.tokens, 0, c.nTrain, 0).loss;
  const [from, to] = c.heldOut ? [c.nTrain, c.tokens.length] : [0, c.nTrain];
  let best = { loss: Infinity, unseen: 0, alpha: ALPHAS[0] as number };
  for (const alpha of ALPHAS) {
    const r = lossOf(t, c.tokens, from, to, alpha);
    if (r.loss < best.loss) best = { ...r, alpha };
  }
  const rowsLoss = targets ? lossAt(t, c.tokens, targets, best.alpha) : undefined;
  return { k, trainLoss, heldOutLoss: best.loss, alpha: best.alpha, contexts: t.totals.size, unseen: best.unseen, sample: sampleFrom(t, c, sampleLength, seed + k), rowsLoss };
}

/** All tables k = 0 … maxK, one per call of the generator's next() (each takes up to about a second on Shakespeare). */
export function* tableLadder(c: Corpus, maxK = MAX_K, sampleLength = 240, seed = 1): Generator<TableResult, TableResult[]> {
  const out: TableResult[] = [];
  for (let k = 0; k <= maxK; k++) {
    const r = tableResult(c, k, sampleLength, seed);
    out.push(r);
    yield r;
  }
  return out;
}

/** The reference lines of the loss chart: guessing, letter frequencies, the previous letter. */
export function referenceLosses(c: Corpus): { uniform: number; unigram: number; bigram: number } {
  const V = c.vocab.length;
  const [from, to] = c.heldOut ? [c.nTrain, c.tokens.length] : [0, c.nTrain];
  const best = (k: number) => {
    const t = count(c.tokens, V, k, 0, c.nTrain);
    return Math.min(...ALPHAS.map((a) => lossOf(t, c.tokens, from, to, a).loss));
  };
  return { uniform: Math.log(V), unigram: best(0), bigram: best(1) };
}

/** The best held-out loss among the tables (and its k). */
export function bestTable(results: readonly TableResult[]): TableResult {
  return results.reduce((a, b) => (b.heldOutLoss < a.heldOutLoss ? b : a));
}
