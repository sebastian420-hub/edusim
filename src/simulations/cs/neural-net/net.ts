/**
 * 64-bit CPU twin of the neural-network shaders (`mlp.wgsl`, `eval.wgsl`, `boundary.wgsl`), plus the datasets, the
 * batch order and every measurement the simulation shows. The GPU tests compare the shaders against these functions.
 *
 * The network is a fully connected multilayer perceptron: inputs → hidden layers (one activation for all of them) →
 * one logit (two classes: sigmoid + cross-entropy) or ten logits (digits: softmax + cross-entropy). Training is
 * mini-batch stochastic gradient descent with momentum and L2 weight decay:
 *   g ← ∇loss + λ·w (weights only),  v ← μ·v + g,  w ← w − η·v.
 * So the step just taken is exactly −η·v, which the step probe uses to look along it.
 *
 * Parameter layout (shared with the GPU): for each layer l = 1…L, its weights W[j][i] (row j = output neuron,
 * column i = input) then its biases, layers one after another.
 */

export type Activation = "tanh" | "relu" | "sigmoid" | "linear";
export type Init = "random" | "zeros";
/** Squared error exists only in the twin, to test the textbook step-size threshold on a pure quadratic bowl. */
export type Loss = "cross-entropy" | "squared";

export interface NetSpec {
  inputs: number;
  hidden: readonly number[];
  /** 1 = two classes (sigmoid), more = that many classes (softmax). */
  outputs: number;
  activation: Activation;
}

export interface Layer {
  /** Inputs and neurons of this layer. */
  nIn: number;
  nOut: number;
  /** Offsets into the packed parameters. */
  w: number;
  b: number;
}

export interface Layout {
  sizes: number[];
  layers: Layer[];
  /** Total number of parameters. */
  count: number;
  /** Widest layer (inputs excluded). */
  maxWidth: number;
}

export function layout(spec: NetSpec): Layout {
  const sizes = [spec.inputs, ...spec.hidden, spec.outputs];
  const layers: Layer[] = [];
  let at = 0;
  for (let l = 1; l < sizes.length; l++) {
    const nIn = sizes[l - 1];
    const nOut = sizes[l];
    layers.push({ nIn, nOut, w: at, b: at + nIn * nOut });
    at += nIn * nOut + nOut;
  }
  return { sizes, layers, count: at, maxWidth: Math.max(...sizes.slice(1)) };
}

// ---------------------------------------------------------------------------------------------------------------
// Random numbers: a small seeded generator so a link reproduces a whole training run.

/** mulberry32: 32-bit state, uniform in [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal from two uniforms (Box–Muller). */
export function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/** Fisher–Yates shuffle of 0…n−1. */
export function permutation(n: number, rand: () => number): Uint32Array {
  const p = new Uint32Array(n);
  for (let i = 0; i < n; i++) p[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = p[i];
    p[i] = p[j];
    p[j] = t;
  }
  return p;
}

/**
 * Glorot-uniform (tanh, sigmoid, linear) or He-uniform (ReLU) weights, zero biases; or everything zero (the classic
 * mistake: every hidden neuron then stays identical, so a layer is no better than one neuron).
 */
export function initParams(spec: NetSpec, seed: number, init: Init = "random"): Float64Array {
  const lay = layout(spec);
  const p = new Float64Array(lay.count);
  if (init === "zeros") return p;
  const rand = rng(seed);
  for (const L of lay.layers) {
    const limit = spec.activation === "relu" ? Math.sqrt(6 / L.nIn) : Math.sqrt(6 / (L.nIn + L.nOut));
    for (let k = 0; k < L.nIn * L.nOut; k++) p[L.w + k] = (2 * rand() - 1) * limit;
  }
  return p;
}

// ---------------------------------------------------------------------------------------------------------------
// Activations. Every derivative is written in terms of the activation's output, which is all the GPU keeps.

export function activate(a: Activation, z: number): number {
  switch (a) {
    case "tanh":
      return Math.tanh(z);
    case "relu":
      return z > 0 ? z : 0;
    case "sigmoid":
      return 1 / (1 + Math.exp(-z));
    case "linear":
      return z;
  }
}

/** d(activation)/dz, from the activation's output y. */
export function slope(a: Activation, y: number): number {
  switch (a) {
    case "tanh":
      return 1 - y * y;
    case "relu":
      return y > 0 ? 1 : 0;
    case "sigmoid":
      return y * (1 - y);
    case "linear":
      return 1;
  }
}

/** Numerically safe ln(1 + eᶻ). */
export const softplus = (z: number) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));

// ---------------------------------------------------------------------------------------------------------------
// Datasets

/** A dataset: `n` samples of `dim` inputs, samples 0…nTrain−1 for training and the rest held out for testing. */
export interface Dataset {
  n: number;
  dim: number;
  nTrain: number;
  x: Float64Array;
  y: Uint8Array;
  classes: number;
}

/** Turns of each spiral arm: enough to need depth, few enough that a small net untangles it. */
export const SPIRAL_TURNS = 1.25;

export const PLANE_SETS = ["blobs", "circle", "xor", "spiral"] as const;
export type PlaneSet = (typeof PLANE_SETS)[number];

/** Raw points on the plane, both coordinates in [−1, 1], labels 0 or 1, in random order. */
export function planePoints(kind: PlaneSet, count: number, noise: number, seed: number): { px: Float64Array; py: Float64Array; y: Uint8Array } {
  const rand = rng(seed);
  const px = new Float64Array(count);
  const py = new Float64Array(count);
  const y = new Uint8Array(count);
  const clamp = (v: number) => Math.min(1, Math.max(-1, v));
  for (let k = 0; k < count; k++) {
    const label = k % 2;
    let a = 0;
    let b = 0;
    if (kind === "blobs") {
      const c = label ? 0.4 : -0.4;
      a = c + 0.2 * gaussian(rand);
      b = c + 0.2 * gaussian(rand);
    } else if (kind === "circle") {
      const r = label ? 0.4 * Math.sqrt(rand()) : 0.6 + 0.3 * rand();
      const t = 2 * Math.PI * rand();
      a = r * Math.cos(t);
      b = r * Math.sin(t);
    } else if (kind === "xor") {
      const gap = 0.06;
      const sx = rand() < 0.5 ? -1 : 1;
      const sy = label ? sx : -sx; // label 1: same signs (x·y > 0)
      a = sx * (gap + (0.95 - gap) * rand());
      b = sy * (gap + (0.95 - gap) * rand());
    } else {
      const t = rand();
      const r = 0.08 + 0.82 * t;
      const angle = SPIRAL_TURNS * 2 * Math.PI * t + (label ? Math.PI : 0);
      a = r * Math.cos(angle) + 0.025 * gaussian(rand);
      b = r * Math.sin(angle) + 0.025 * gaussian(rand);
    }
    px[k] = clamp(a);
    py[k] = clamp(b);
    y[k] = rand() < noise ? 1 - label : label;
  }
  // Shuffle so the train/test split takes a random part of each class.
  const order = permutation(count, rand);
  return { px: Float64Array.from(order, (i) => px[i]), py: Float64Array.from(order, (i) => py[i]), y: Uint8Array.from(order, (i) => y[i]) };
}

export const FEATURES = ["x", "y", "x2", "y2", "xy", "sinx", "siny"] as const;
export type Feature = (typeof FEATURES)[number];

/** The input features of one point, in FEATURES order (only those in `mask`, a bit per feature). */
export function featureValue(f: number, x: number, y: number): number {
  switch (f) {
    case 0:
      return x;
    case 1:
      return y;
    case 2:
      return x * x;
    case 3:
      return y * y;
    case 4:
      return x * y;
    case 5:
      return Math.sin(Math.PI * x);
    default:
      return Math.sin(Math.PI * y);
  }
}

export const featureCount = (mask: number) => FEATURES.reduce((n, _, f) => n + ((mask >> f) & 1), 0);

export function features(mask: number, x: number, y: number, out: Float64Array | Float32Array = new Float64Array(featureCount(mask))) {
  let k = 0;
  for (let f = 0; f < FEATURES.length; f++) if ((mask >> f) & 1) out[k++] = featureValue(f, x, y);
  return out;
}

/** A plane dataset with the chosen features and train share. */
export function planeDataset(kind: PlaneSet, count: number, noise: number, trainShare: number, mask: number, seed: number): Dataset & { px: Float64Array; py: Float64Array } {
  const pts = planePoints(kind, count, noise, seed);
  const dim = featureCount(mask);
  const x = new Float64Array(count * dim);
  const row = new Float64Array(dim);
  for (let k = 0; k < count; k++) x.set(features(mask, pts.px[k], pts.py[k], row), k * dim);
  const nTrain = Math.min(count - 1, Math.max(1, Math.round(count * trainShare)));
  return { n: count, dim, nTrain, x, y: pts.y, classes: 2, px: pts.px, py: pts.py };
}

/**
 * The order samples are visited in: one shuffled pass over the training set after another, cut into batches of
 * `size`. A batch may span two passes. The GPU gets exactly these indices, so it trains on what the twin trains on.
 */
export function batchStream(nTrain: number, size: number, seed: number) {
  const rand = rng(seed ^ 0x9e3779b9);
  let order = permutation(nTrain, rand);
  let at = 0;
  let seen = 0;
  const batch = Math.max(1, Math.min(size, nTrain));
  return {
    size: batch,
    /** Training samples visited so far, in passes over the training set. */
    get epochs() {
      return seen / nTrain;
    },
    next(out: Uint32Array = new Uint32Array(batch)): Uint32Array {
      for (let k = 0; k < batch; k++) {
        if (at === nTrain) {
          order = permutation(nTrain, rand);
          at = 0;
        }
        out[k] = order[at++];
      }
      seen += batch;
      return out;
    },
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Forward, loss, backpropagation

/** Post-activation values of every layer (the last holds the logits) for one input. */
export function forward(spec: NetSpec, lay: Layout, p: ArrayLike<number>, input: ArrayLike<number>, inputOffset = 0): Float64Array[] {
  const acts: Float64Array[] = [];
  let prev: ArrayLike<number> = input;
  let prevOff = inputOffset;
  lay.layers.forEach((L, l) => {
    const out = new Float64Array(L.nOut);
    const last = l === lay.layers.length - 1;
    for (let j = 0; j < L.nOut; j++) {
      let z = p[L.b + j];
      const row = L.w + j * L.nIn;
      for (let i = 0; i < L.nIn; i++) z += p[row + i] * prev[prevOff + i];
      out[j] = last ? z : activate(spec.activation, z);
    }
    acts.push(out);
    prev = out;
    prevOff = 0;
  });
  return acts;
}

/** Loss of one sample from its logits, and the gradient of that loss with respect to the logits. */
export function head(logits: Float64Array, label: number, loss: Loss = "cross-entropy"): { loss: number; dz: Float64Array; correct: boolean; p: Float64Array } {
  const n = logits.length;
  const dz = new Float64Array(n);
  const p = new Float64Array(n);
  if (loss === "squared") {
    // Regression onto the label (one output): ½(z − y)².
    const e = logits[0] - label;
    dz[0] = e;
    p[0] = logits[0];
    return { loss: 0.5 * e * e, dz, correct: logits[0] > 0.5 === (label === 1), p };
  }
  if (n === 1) {
    const z = logits[0];
    p[0] = 1 / (1 + Math.exp(-z));
    dz[0] = p[0] - label;
    return { loss: softplus(z) - label * z, dz, correct: z > 0 === (label === 1), p };
  }
  let max = -Infinity;
  let best = 0;
  for (let k = 0; k < n; k++)
    if (logits[k] > max) {
      max = logits[k];
      best = k;
    }
  let sum = 0;
  for (let k = 0; k < n; k++) sum += (p[k] = Math.exp(logits[k] - max));
  for (let k = 0; k < n; k++) {
    p[k] /= sum;
    dz[k] = p[k] - (k === label ? 1 : 0);
  }
  return { loss: Math.log(sum) + max - logits[label], dz, correct: best === label, p };
}

/**
 * Mean loss over the given samples and its gradient with respect to every parameter (backpropagation: the
 * chain rule, layer by layer from the output back). L2 is not included here; it is part of the update.
 */
export function gradient(spec: NetSpec, lay: Layout, p: ArrayLike<number>, data: Dataset, samples: ArrayLike<number>, loss: Loss = "cross-entropy"): { loss: number; grad: Float64Array } {
  const grad = new Float64Array(lay.count);
  const L = lay.layers.length;
  let total = 0;
  const B = samples.length;
  for (let s = 0; s < B; s++) {
    const off = samples[s] * data.dim;
    const acts = forward(spec, lay, p, data.x, off);
    const h = head(acts[L - 1], data.y[samples[s]], loss);
    total += h.loss;
    let delta = h.dz;
    for (let l = L - 1; l >= 0; l--) {
      const layer = lay.layers[l];
      const prev = l > 0 ? acts[l - 1] : null;
      for (let j = 0; j < layer.nOut; j++) {
        const d = delta[j] / B;
        grad[layer.b + j] += d;
        const row = layer.w + j * layer.nIn;
        for (let i = 0; i < layer.nIn; i++) grad[row + i] += d * (prev ? prev[i] : data.x[off + i]);
      }
      if (l === 0) break;
      const back = new Float64Array(layer.nIn);
      for (let i = 0; i < layer.nIn; i++) {
        let sum = 0;
        for (let j = 0; j < layer.nOut; j++) sum += p[layer.w + j * layer.nIn + i] * delta[j];
        back[i] = sum * slope(spec.activation, prev![i]);
      }
      delta = back;
    }
  }
  return { loss: total / B, grad };
}

export interface Optimizer {
  lr: number;
  momentum: number;
  l2: number;
}

/** True for parameters that are weights (L2 applies to them), false for biases. */
export function weightMask(lay: Layout): Uint8Array {
  const m = new Uint8Array(lay.count);
  for (const L of lay.layers) m.fill(1, L.w, L.w + L.nIn * L.nOut);
  return m;
}

/** One SGD-with-momentum update, in place. Exactly `update` in mlp.wgsl. */
export function update(p: Float64Array, v: Float64Array, grad: ArrayLike<number>, opt: Optimizer, weights: Uint8Array) {
  for (let k = 0; k < p.length; k++) {
    const g = grad[k] + (weights[k] ? opt.l2 * p[k] : 0);
    v[k] = opt.momentum * v[k] + g;
    p[k] -= opt.lr * v[k];
  }
}

/** Mean loss and accuracy over samples [first, first + count). */
export function evaluate(spec: NetSpec, lay: Layout, p: ArrayLike<number>, data: Dataset, first: number, count: number, loss: Loss = "cross-entropy"): { loss: number; accuracy: number } {
  let total = 0;
  let right = 0;
  for (let s = first; s < first + count; s++) {
    const acts = forward(spec, lay, p, data.x, s * data.dim);
    const h = head(acts[acts.length - 1], data.y[s], loss);
    total += h.loss;
    if (h.correct) right++;
  }
  return { loss: count ? total / count : 0, accuracy: count ? right / count : 0 };
}

/** The probability of class 1 at a point of the plane (what each pixel of the decision map shows). */
export function probabilityAt(spec: NetSpec, lay: Layout, p: ArrayLike<number>, mask: number, x: number, y: number): number {
  const acts = forward(spec, lay, p, features(mask, x, y));
  return 1 / (1 + Math.exp(-acts[acts.length - 1][0]));
}

// ---------------------------------------------------------------------------------------------------------------
// Measurements

/** Centred finite-difference gradient: the standard check that backpropagation is right (h = 10⁻⁴ keeps rounding error below 10⁻⁶ in 64-bit even for tiny gradients). */
export function numericGradient(spec: NetSpec, lay: Layout, p: Float64Array, data: Dataset, samples: ArrayLike<number>, loss: Loss = "cross-entropy", h = 1e-4): Float64Array {
  const g = new Float64Array(p.length);
  const q = Float64Array.from(p);
  const at = () => gradient(spec, lay, q, data, samples, loss).loss;
  for (let k = 0; k < p.length; k++) {
    q[k] = p[k] + h;
    const up = at();
    q[k] = p[k] - h;
    const down = at();
    q[k] = p[k];
    g[k] = (up - down) / (2 * h);
  }
  return g;
}

/** Root-mean-square gradient of each layer's weights: how strongly the error reaches it. */
export function layerGradientSizes(lay: Layout, grad: ArrayLike<number>): number[] {
  return lay.layers.map((L) => {
    let s = 0;
    for (let k = L.w; k < L.w + L.nIn * L.nOut; k++) s += grad[k] * grad[k];
    return Math.sqrt(s / (L.nIn * L.nOut));
  });
}

/**
 * How different the hidden neurons of each layer are: the largest gap between any neuron's incoming weights and
 * bias and those of the layer's first neuron (0 = all identical clones).
 */
export function neuronSpread(lay: Layout, p: ArrayLike<number>): number {
  let spread = 0;
  for (const L of lay.layers.slice(0, -1)) {
    for (let j = 1; j < L.nOut; j++) {
      spread = Math.max(spread, Math.abs(p[L.b + j] - p[L.b]));
      for (let i = 0; i < L.nIn; i++) spread = Math.max(spread, Math.abs(p[L.w + j * L.nIn + i] - p[L.w + i]));
    }
  }
  return spread;
}

/** Loss along the step just taken: at w + (s − 1)·(−η·v) for each s, so s = 0 is before the step and s = 1 after. */
export function stepProfile(spec: NetSpec, lay: Layout, p: ArrayLike<number>, v: ArrayLike<number>, lr: number, data: Dataset, s: readonly number[], loss: Loss = "cross-entropy"): number[] {
  const q = new Float64Array(p.length);
  return s.map((t) => {
    for (let k = 0; k < q.length; k++) q[k] = p[k] - (t - 1) * lr * v[k];
    return evaluate(spec, lay, q, data, 0, data.nTrain, loss).loss;
  });
}

/** The s values the step probe samples: 0 … 2 in 16ths (s = 1 is where the step landed). */
export const PROBE_S = Array.from({ length: 17 }, (_, k) => k / 8);

/**
 * A training run on the twin: `steps` steps from `seed`, as the simulation trains. Returns the final parameters,
 * velocity and the training loss after each step (measured on the batch's data before the update).
 */
export function train(spec: NetSpec, data: Dataset, opt: Optimizer & { batch: number; seed: number; init?: Init; loss?: Loss }, steps: number, onStep?: (k: number, p: Float64Array, grad: Float64Array) => void) {
  const lay = layout(spec);
  const p = initParams(spec, opt.seed, opt.init ?? "random");
  const v = new Float64Array(lay.count);
  const weights = weightMask(lay);
  const stream = batchStream(data.nTrain, opt.batch, opt.seed);
  const batch = new Uint32Array(stream.size);
  for (let k = 0; k < steps; k++) {
    stream.next(batch);
    const { grad } = gradient(spec, lay, p, data, batch, opt.loss);
    update(p, v, grad, opt, weights);
    onStep?.(k, p, grad);
  }
  return { p, v, lay, stream };
}
