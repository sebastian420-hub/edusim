/**
 * Where a model puts each letter: the two main directions (principal components) of its letter vectors, so letters
 * it treats alike land close together. For a transformer and the MLP these are the learnt embeddings; for the neural
 * bigram, each letter's row of next-letter scores.
 */
import type { Layout, ModelSpec } from "./model";

/** The V letter vectors of a model, row by row. */
export function letterVectors(spec: ModelSpec, lay: Layout, params: ArrayLike<number>): { rows: number; cols: number; data: Float64Array } {
  const [at, cols] = spec.kind === "gpt" ? [lay.at.wte, spec.d] : spec.kind === "mlp" ? [lay.at.C, spec.e] : [lay.at.table, spec.V];
  const data = new Float64Array(spec.V * cols);
  for (let i = 0; i < data.length; i++) data[i] = params[at + i];
  return { rows: spec.V, cols, data };
}

/** The first two principal components of a set of row vectors (power iteration with deflation), and each row's coordinates. */
export function project2d(m: { rows: number; cols: number; data: ArrayLike<number> }, iterations = 60): { x: number[]; y: number[]; explained: number } {
  const { rows, cols, data } = m;
  const mean = new Float64Array(cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) mean[c] += data[r * cols + c] / rows;
  const X = new Float64Array(rows * cols);
  let total = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      X[r * cols + c] = data[r * cols + c] - mean[c];
      total += X[r * cols + c] ** 2;
    }
  }
  const components: Float64Array[] = [];
  const variances: number[] = [];
  for (let k = 0; k < 2; k++) {
    // Deterministic start, so the picture doesn't flip between updates.
    let v = Float64Array.from({ length: cols }, (_, c) => 1 + ((c * 7919 + k * 104729) % 13) / 13);
    let lambda = 0;
    for (let it = 0; it < iterations; it++) {
      const Xv = new Float64Array(rows);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) Xv[r] += X[r * cols + c] * v[c];
      const next = new Float64Array(cols);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) next[c] += X[r * cols + c] * Xv[r];
      for (const u of components) {
        let d = 0;
        for (let c = 0; c < cols; c++) d += next[c] * u[c];
        for (let c = 0; c < cols; c++) next[c] -= d * u[c];
      }
      const norm = Math.hypot(...next);
      if (norm === 0) break;
      lambda = norm;
      v = next.map((x) => x / norm);
    }
    // A sign convention: the largest coordinate is positive.
    let big = 0;
    for (let c = 0; c < cols; c++) if (Math.abs(v[c]) > Math.abs(big)) big = v[c];
    if (big < 0) v = v.map((x) => -x);
    components.push(v);
    variances.push(lambda);
  }
  const coord = (u: Float64Array) => Array.from({ length: rows }, (_, r) => {
    let s = 0;
    for (let c = 0; c < cols; c++) s += X[r * cols + c] * u[c];
    return s;
  });
  return { x: coord(components[0]), y: coord(components[1]), explained: total > 0 ? (variances[0] + variances[1]) / total : 0 };
}
