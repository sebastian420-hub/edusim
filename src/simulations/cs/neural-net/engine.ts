import { compute, storage } from "vgpu";
import type { Gpu, StorageBuffer } from "vgpu";
import boundaryShader from "./boundary.wgsl";
import evalShader from "./eval.wgsl";
import mlpShader from "./mlp.wgsl";
import { layout } from "./net";
import type { Activation, Layout, NetSpec, Optimizer } from "./net";

type Destroyable = { destroy?: () => void };

export const WORKGROUP = 64;
/** Layers with weights the shaders' tables hold (6 hidden + the output, with room to spare). */
export const MAX_LAYERS = 8;
/** Widest layer eval.wgsl can hold in private memory, and the widest boundary.wgsl can (the plane only). */
export const MAX_EVAL_WIDTH = 128;
export const MAX_MAP_WIDTH = 32;
/** Probe points along the last step (eval.wgsl's uniform holds up to 20). */
export const MAX_PROBES = 20;

export const ACTIVATION_CODE: Record<Activation, number> = { tanh: 0, relu: 1, sigmoid: 2, linear: 3 };

/** The data as the GPU stores it: f32 features, or bytes (digit pixels, 0 … 255, scaled to 0 … 1 on the GPU). */
export type EngineData =
  | { kind: "features"; n: number; dim: number; x: ArrayLike<number>; y: ArrayLike<number> }
  | { kind: "bytes"; n: number; dim: number; x: Uint8Array; y: ArrayLike<number> };

export interface EngineOptions {
  /** Largest batch a step may use. */
  maxBatch: number;
  /** Decision map: size × size cells for the output, thumb × thumb per hidden neuron (0 = no map). */
  mapSize?: number;
  thumb?: number;
}

/** Workgroup counts for n invocations, spilling into y past the 65 535 limit (shaders use flatId()). */
export function workgroupsFor(n: number): [number, number] {
  const groups = Math.max(1, Math.ceil(n / WORKGROUP));
  const x = Math.min(groups, 65535);
  return [x, Math.ceil(groups / x)];
}

/** Packs the data into one u32 buffer: inputs (f32 bits, or bytes four per word), then the labels. */
export function packData(d: EngineData): { words: Uint32Array; labelOff: number; inputKind: number } {
  if (d.kind === "bytes") {
    if (d.dim % 4 !== 0) throw new Error("byte inputs must be a multiple of 4 wide");
    const perSample = d.dim / 4;
    const words = new Uint32Array(d.n * perSample + d.n);
    const bytes = new Uint8Array(words.buffer, 0, d.n * d.dim);
    bytes.set(d.x.subarray(0, d.n * d.dim));
    for (let k = 0; k < d.n; k++) words[d.n * perSample + k] = d.y[k];
    return { words, labelOff: d.n * perSample, inputKind: 1 };
  }
  const words = new Uint32Array(d.n * d.dim + d.n);
  new Float32Array(words.buffer, 0, d.n * d.dim).set(Array.from({ length: d.n * d.dim }, (_, i) => d.x[i]));
  for (let k = 0; k < d.n; k++) words[d.n * d.dim + k] = d.y[k];
  return { words, labelOff: d.n * d.dim, inputKind: 0 };
}

const layerTable = (lay: Layout) => Array.from({ length: MAX_LAYERS }, (_, l) => (lay.layers[l] ? [lay.layers[l].nIn, lay.layers[l].nOut, lay.layers[l].w, lay.layers[l].b] : [0, 0, 0, 0]));

/**
 * The network on the GPU, without drawing: parameters, momentum, the data, one training step as a sequence of
 * dispatches, batched evaluation (with the step probe), and the decision map. Shared by the simulation and the
 * headless GPU tests. One engine per network shape and dataset: build a new one when either changes.
 */
export function createEngine(gpu: Gpu, spec: NetSpec, data: EngineData, opts: EngineOptions) {
  const lay = layout(spec);
  if (lay.layers.length > MAX_LAYERS) throw new Error(`at most ${MAX_LAYERS - 1} hidden layers`);
  if (lay.maxWidth > MAX_EVAL_WIDTH) throw new Error(`layers are at most ${MAX_EVAL_WIDTH} wide`);
  if (data.dim !== spec.inputs) throw new Error(`the network takes ${spec.inputs} inputs, the data has ${data.dim}`);
  const P = lay.count;
  const maxBatch = Math.max(1, opts.maxBatch);
  const widths = lay.layers.reduce((a, L) => a + L.nOut, 0);
  const packed = packData(data);
  const mapSize = opts.mapSize ?? 0;
  const thumb = opts.thumb ?? 0;
  const hiddenNeurons = spec.hidden.reduce((a, b) => a + b, 0);
  const mapCells = mapSize ? mapSize * mapSize + hiddenNeurons * thumb * thumb : 0;
  if (mapSize && (lay.maxWidth > MAX_MAP_WIDTH || spec.inputs > 7)) throw new Error("the decision map is for small plane networks");

  const params = storage(gpu, P * 4);
  const grads = storage(gpu, P * 8);
  const dataBuf = storage(gpu, packed.words.byteLength);
  const batch = storage(gpu, maxBatch * 4);
  const work = storage(gpu, Math.max(4, 2 * maxBatch * widths * 4));
  let outLen = data.n;
  let out = storage(gpu, outLen * 16);
  const map = storage(gpu, Math.max(4, mapCells * 4));
  dataBuf.write(packed.words as Uint32Array<ArrayBuffer>);

  const kernels = {
    forward: compute(gpu, mlpShader, { entry: "forward" }),
    head: compute(gpu, mlpShader, { entry: "head" }),
    backward: compute(gpu, mlpShader, { entry: "backward" }),
    gradients: compute(gpu, mlpShader, { entry: "gradients" }),
    update: compute(gpu, mlpShader, { entry: "update" }),
  };
  const evaluate = compute(gpu, evalShader);
  const boundary = compute(gpu, boundaryShader);

  const act = ACTIVATION_CODE[spec.activation];
  const layers = layerTable(lay);
  let steps = 0;

  const netUniform = (B: number, layer: number, opt: Optimizer) => {
    const offs: number[][] = [];
    let at = 0;
    for (let l = 0; l < MAX_LAYERS; l++) {
      const n = lay.layers[l]?.nOut ?? 0;
      offs.push([at, B * widths + at, 0, 0]);
      at += B * n;
    }
    return {
      L: lay.layers.length,
      B,
      layer,
      inputKind: packed.inputKind,
      act,
      P,
      labelOff: packed.labelOff,
      nInputs: spec.inputs,
      lr: opt.lr,
      momentum: opt.momentum,
      l2: opt.l2,
      pad: 0,
      layers,
      offs,
    };
  };

  const bind = { params, data: dataBuf, batch, work, grads };

  const engine = {
    spec,
    layout: lay,
    data,
    buffers: { params, grads, map, data: dataBuf } as { params: StorageBuffer; grads: StorageBuffer; map: StorageBuffer; data: StorageBuffer },
    /** u32 words per sample of byte data (the digits renderer reads pixels straight from the data buffer). */
    wordsPerSample: packed.inputKind === 1 ? data.dim / 4 : data.dim,
    mapSize,
    thumb,
    hiddenNeurons,
    /** Training steps taken since the parameters were loaded. */
    get steps() {
      return steps;
    },

    /** Loads parameters (in the packed layout) and clears the momentum. */
    load(p: ArrayLike<number>) {
      if (p.length !== P) throw new Error(`expected ${P} parameters, got ${p.length}`);
      params.write(Float32Array.from(p as ArrayLike<number>) as Float32Array<ArrayBuffer>);
      grads.write(new Float32Array(2 * P));
      steps = 0;
    },

    /** One step of SGD on the given samples (training sample indices). */
    step(samples: Uint32Array, opt: Optimizer) {
      const B = samples.length;
      if (B < 1 || B > maxBatch) throw new Error(`batch of ${B}, engine holds 1 … ${maxBatch}`);
      batch.write(samples as Uint32Array<ArrayBuffer>);
      const L = lay.layers.length;
      for (let l = 0; l < L - 1; l++) kernels.forward.set({ ...bind, net: netUniform(B, l, opt) }).dispatch(...workgroupsFor(B * lay.layers[l].nOut));
      kernels.head.set({ ...bind, net: netUniform(B, L - 1, opt) }).dispatch(...workgroupsFor(B));
      for (let l = L - 2; l >= 0; l--) kernels.backward.set({ ...bind, net: netUniform(B, l, opt) }).dispatch(...workgroupsFor(B * lay.layers[l].nOut));
      const all = netUniform(B, 0, opt);
      kernels.gradients.set({ ...bind, net: all }).dispatch(...workgroupsFor(P));
      kernels.update.set({ ...bind, net: all }).dispatch(...workgroupsFor(P));
      steps++;
    },

    /**
     * Evaluates samples [first, first + count) at each probe point s (default: just s = 1, the network now):
     * one vec4 (loss, correct, guess, its probability) per (probe point, sample), probe-major.
     */
    async evaluate(first: number, count: number, s: readonly number[] = [1], lr = 0): Promise<Float32Array> {
      if (s.length > MAX_PROBES) throw new Error(`at most ${MAX_PROBES} probe points`);
      const n = count * s.length;
      if (n > outLen) {
        (out as unknown as Destroyable).destroy?.();
        outLen = n;
        out = storage(gpu, outLen * 16);
      }
      const sv = Array.from({ length: 5 }, (_, k) => [0, 1, 2, 3].map((i) => s[4 * k + i] ?? 1));
      evaluate
        .set({
          params,
          grads,
          data: dataBuf,
          out,
          ev: { L: lay.layers.length, first, count, S: s.length, inputKind: packed.inputKind, nInputs: spec.inputs, labelOff: packed.labelOff, act, lr, P, pad0: 0, pad1: 0, layers, s: sv },
        })
        .dispatch(...workgroupsFor(n));
      return new Float32Array(await out.read(), 0, n * 4);
    },

    /** Recomputes the decision map (and the hidden neurons' small maps) from the current parameters. */
    drawMap(featureMask: number) {
      if (!mapCells) return;
      boundary.set({ params, map, m: { L: lay.layers.length, act, mask: featureMask, size: mapSize, thumb, neurons: hiddenNeurons, pad0: 0, pad1: 0, layers } }).dispatch(...workgroupsFor(mapCells));
    },

    async readParams(): Promise<Float32Array> {
      return new Float32Array(await params.read(), 0, P);
    },
    /** The last batch's gradient and the momentum velocity. */
    async readGrads(): Promise<{ grad: Float32Array; velocity: Float32Array }> {
      const all = new Float32Array(await grads.read(), 0, 2 * P);
      return { grad: all.subarray(0, P), velocity: all.subarray(P) };
    },
    async readMap(): Promise<Float32Array> {
      return new Float32Array(await map.read(), 0, mapCells);
    },

    dispose() {
      for (const b of [params, grads, dataBuf, batch, work, out, map]) (b as unknown as Destroyable).destroy?.();
    },
  };
  return engine;
}

export type Engine = ReturnType<typeof createEngine>;
