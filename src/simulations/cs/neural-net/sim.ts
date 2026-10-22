import type { FramePass } from "vgpu";
import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import { confusion, DIGIT_PIXELS, loadDigits } from "./digits";
import type { Digits } from "./digits";
import { createDigitsRenderer, digitsLayout, MAX_TILES } from "./digitsRenderer";
import type { DigitsRenderer } from "./digitsRenderer";
import { createEngine } from "./engine";
import type { Engine } from "./engine";
import { batchStream, forward, head, initParams, layerGradientSizes, neuronSpread, planeDataset, probabilityAt, PROBE_S } from "./net";
import type { Init, NetSpec, Optimizer, PlaneSet } from "./net";
import { createPlaygroundRenderer } from "./renderer";
import type { PlaygroundRenderer } from "./renderer";
import { nodeCenter, planeAt, sceneLayout } from "./scene";
import type { NetView } from "./settings";

/** Decision map: cells per side for the output, and for each hidden neuron's small map. */
export const MAP_SIZE = 160;
export const THUMB = 24;
const MAP_INTERVAL_MS = 33;
const MEASURE_INTERVAL_MS = 250;
const SLOW_MEASURE_INTERVAL_MS = 500;
const STATS_INTERVAL_MS = 100;
const HISTORY = 300;
/** Digits: how many training digits the training accuracy is measured on. */
const DIGITS_TRAIN_SAMPLE = 2000;
/** Templates shown for a network with hidden layers (its first neurons). */
const HIDDEN_TEMPLATES = 16;

export interface PlaneConfig {
  data: PlaneSet;
  points: number;
  noise: number;
  train: number;
  features: number;
  spec: NetSpec;
  init: Init;
  seed: number;
}

export interface NetConfig {
  view: NetView;
  plane: PlaneConfig;
  opt: Optimizer & { batch: number };
  digits: { spec: NetSpec; opt: Optimizer & { batch: number }; seed: number };
  /** Training steps per frame while playing. */
  speed: number;
  showTest: boolean;
}

export interface CurvePoint {
  step: number;
  epoch: number;
  trainLoss: number;
  trainAcc: number;
  testLoss: number;
  testAcc: number;
}

export interface PlaygroundStats {
  view: "playground";
  steps: number;
  epochs: number;
  nTrain: number;
  nTest: number;
  /** Hidden layers of the network being trained. */
  hidden: number;
  now: CurvePoint | null;
  /** The first measurement (before any step). */
  start: CurvePoint | null;
  history: CurvePoint[];
  /** Training loss along the last step (PROBE_S: 0 = before it, 1 = where it landed, 2 = twice as far). */
  probe: number[] | null;
  /** Root-mean-square gradient of each layer's weights (input side first). */
  gradients: number[] | null;
  /** How different the hidden neurons are (0 = clones). */
  spread: number | null;
  diverged: boolean;
}

export interface DigitsStats {
  view: "digits";
  status: "loading" | "ready" | "error";
  steps: number;
  epochs: number;
  hidden: number;
  now: CurvePoint | null;
  history: CurvePoint[];
  /** Test set: counts[truth][guess]. */
  confusion: number[][] | null;
  mistakes: number;
  nTest: number;
  diverged: boolean;
}

export type NetStats = PlaygroundStats | DigitsStats;

export interface NetOptions {
  onStats?: (stats: NetStats | null) => void;
}

export interface Label {
  text: string;
  /** CSS pixels from the canvas's top-left; the label is centred on x. */
  x: number;
  y: number;
}

export type TileMode = "all" | "mistakes";

export interface NetHandle extends SimHandle {
  configure(config: NetConfig): void;
  play(): void;
  pause(): void;
  step(): void;
  reset(): void;
  setTileMode(mode: TileMode): void;
  /** Labels drawn over the canvas (column names, captions), in CSS pixels. */
  labels(): Label[];
  /** Playground: the network's probability of class 1 at a canvas point, or null outside the plot. */
  probabilityAt(xCss: number, yCss: number): { x: number; y: number; p: number } | null;
  /** Digits: the ten probabilities for a 28 × 28 image (ink 0 … 1), from the latest weights. */
  classify(image: ArrayLike<number>): number[] | null;
}

const sameJson = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** Mean loss and accuracy of a block of eval results. */
function summarise(out: Float32Array, from: number, count: number) {
  let loss = 0;
  let right = 0;
  for (let k = from; k < from + count; k++) {
    loss += out[4 * k];
    right += out[4 * k + 1];
  }
  return { loss: count ? loss / count : 0, acc: count ? right / count : 0 };
}

const pushHistory = (history: CurvePoint[], p: CurvePoint) => {
  history.push(p);
  if (history.length > HISTORY) {
    // Keep the shape of the whole run: drop every other point.
    const kept = history.filter((_, k) => k % 2 === 0 || k === history.length - 1);
    history.length = 0;
    history.push(...kept);
  }
};

export function createNeuralNet({ gpu, canvas, surface, loop: frameLoop }: SimContext, { onStats }: NetOptions = {}): NetHandle {
  let config: NetConfig | null = null;
  let playing = false;
  let stepsRequested = 0;
  let disposed = false;
  let epoch = 0; // bumps on every reload, so late read-backs of an old network are dropped

  // ── Playground ──
  let plane: { engine: Engine; renderer: PlaygroundRenderer; nTrain: number; n: number; stream: ReturnType<typeof batchStream>; batch: number } | null = null;
  let planeKey = "";
  let mapDirty = true;
  let lastMap = 0;
  let planeParams: Float32Array | null = null;

  // ── Digits ──
  let digits: Digits | null = null;
  let digitsStatus: DigitsStats["status"] = "loading";
  let digitsEngine: { engine: Engine; renderer: DigitsRenderer; stream: ReturnType<typeof batchStream>; batch: number } | null = null;
  let digitsKey = "";
  let digitsParams: Float32Array | null = null;
  let tileMode: TileMode = "all";
  let lastGuesses: Float32Array | null = null;
  let templateScale = 1;
  let digitsConfusion: number[][] | null = null;
  let mistakes = 0;

  // ── Measurements (shared shape) ──
  let history: CurvePoint[] = [];
  let now: CurvePoint | null = null;
  let start: CurvePoint | null = null;
  let probe: number[] | null = null;
  let gradients: number[] | null = null;
  let spread: number | null = null;
  let measuring = false;
  let lastMeasure = -Infinity;
  let lastSlow = -Infinity;
  let measuredAt = -1;
  let slowAt = -1;
  let lastEmit = 0;
  let lastLr = 0;

  const engineNow = () => (config?.view === "digits" ? digitsEngine?.engine : plane?.engine) ?? null;
  const cssScale = () => {
    const rect = canvas.getBoundingClientRect();
    return rect.width > 0 ? surface.size[0] / rect.width : 1;
  };

  const clearMeasurements = () => {
    history = [];
    now = null;
    start = null;
    probe = null;
    gradients = null;
    spread = null;
    measuredAt = -1;
    slowAt = -1;
    lastMeasure = -Infinity;
    lastSlow = -Infinity;
    planeParams = null;
    digitsParams = null;
    lastGuesses = null;
    digitsConfusion = null;
    mistakes = 0;
  };

  const loadPlane = (c: PlaneConfig) => {
    epoch++;
    plane?.renderer.dispose();
    plane?.engine.dispose();
    const d = planeDataset(c.data, c.points, c.noise, c.train, c.features, c.seed);
    const engine = createEngine(gpu, c.spec, { kind: "features", n: d.n, dim: d.dim, x: d.x, y: d.y }, { maxBatch: d.nTrain, mapSize: MAP_SIZE, thumb: THUMB });
    engine.load(initParams(c.spec, c.seed, c.init));
    const renderer = createPlaygroundRenderer(gpu, engine, c.features);
    const pts = new Float32Array(d.n * 4);
    for (let k = 0; k < d.n; k++) pts.set([d.px[k], d.py[k], d.y[k], k >= d.nTrain ? 1 : 0], k * 4);
    renderer.setPoints(pts);
    const batch = Math.min(config!.opt.batch, d.nTrain);
    plane = { engine, renderer, nTrain: d.nTrain, n: d.n, stream: batchStream(d.nTrain, batch, c.seed), batch };
    mapDirty = true;
    clearMeasurements();
    onStats?.(null);
  };

  const loadDigitsNet = () => {
    if (!digits || !config) return;
    epoch++;
    digitsEngine?.renderer.dispose();
    digitsEngine?.engine.dispose();
    const c = config.digits;
    const engine = createEngine(gpu, c.spec, { kind: "bytes", n: digits.count, dim: DIGIT_PIXELS, x: digits.pixels, y: digits.labels }, { maxBatch: 128 });
    engine.load(initParams(c.spec, c.seed));
    const renderer = createDigitsRenderer(gpu, engine);
    const batch = Math.min(c.opt.batch, digits.nTrain);
    digitsEngine = { engine, renderer, stream: batchStream(digits.nTrain, batch, c.seed), batch };
    clearMeasurements();
    onStats?.(null);
  };

  const reload = () => {
    if (!config) return;
    if (config.view === "playground") loadPlane(config.plane);
    else if (digits) loadDigitsNet();
  };

  // The digits are fetched once, the first time the Digits view opens.
  let digitsRequested = false;
  const requestDigits = () => {
    if (digitsRequested) return;
    digitsRequested = true;
    digitsStatus = "loading";
    loadDigits().then(
      (d) => {
        if (disposed) return;
        digits = d;
        digitsStatus = "ready";
        if (config?.view === "digits") {
          digitsKey = JSON.stringify(config.digits.spec) + config.digits.seed;
          loadDigitsNet();
        }
      },
      () => {
        digitsStatus = "error";
        emit(performance.now(), true);
      },
    );
  };

  /** Applies a (possibly changed) batch size without restarting: the batch order continues from here. */
  const retune = () => {
    if (!config) return;
    if (plane) {
      const b = Math.min(config.opt.batch, plane.nTrain);
      if (b !== plane.batch) {
        plane.stream = batchStream(plane.nTrain, b, config.plane.seed + plane.engine.steps);
        plane.batch = b;
      }
    }
    if (digitsEngine && digits) {
      const b = Math.min(config.digits.opt.batch, digits.nTrain);
      if (b !== digitsEngine.batch) {
        digitsEngine.stream = batchStream(digits.nTrain, b, config.digits.seed + digitsEngine.engine.steps);
        digitsEngine.batch = b;
      }
    }
  };

  const train = (steps: number) => {
    if (!config || steps <= 0) return;
    if (config.view === "playground" && plane) {
      for (let k = 0; k < steps; k++) plane.engine.step(plane.stream.next(), config.opt);
      lastLr = config.opt.lr;
      mapDirty = true;
    } else if (config.view === "digits" && digitsEngine) {
      for (let k = 0; k < steps; k++) digitsEngine.engine.step(digitsEngine.stream.next(), config.digits.opt);
    }
  };

  const diverged = () => !!now && !Number.isFinite(now.trainLoss);

  const emit = (t: number, force = false) => {
    if (!config || (!force && t - lastEmit < STATS_INTERVAL_MS)) return;
    lastEmit = t;
    if (config.view === "playground") {
      if (!plane) return;
      onStats?.({
        view: "playground",
        steps: plane.engine.steps,
        epochs: plane.stream.epochs,
        nTrain: plane.nTrain,
        nTest: plane.n - plane.nTrain,
        hidden: config.plane.spec.hidden.length,
        now,
        start,
        history: history.slice(),
        probe,
        gradients,
        spread,
        diverged: diverged(),
      });
    } else {
      onStats?.({
        view: "digits",
        status: digitsStatus,
        steps: digitsEngine?.engine.steps ?? 0,
        epochs: digitsEngine?.stream.epochs ?? 0,
        hidden: config.digits.spec.hidden.length,
        now,
        history: history.slice(),
        confusion: digitsConfusion,
        mistakes,
        nTest: digits ? digits.count - digits.nTrain : 0,
        diverged: diverged(),
      });
    }
  };

  const updateTiles = () => {
    if (!digits || !digitsEngine || !lastGuesses) return;
    const nTest = digits.count - digits.nTrain;
    const t = new Float32Array(MAX_TILES * 4);
    let k = 0;
    for (let i = 0; i < nTest && k < MAX_TILES; i++) {
      const right = lastGuesses[4 * i + 1] > 0.5;
      if (tileMode === "mistakes" && right) continue;
      t.set([digits.nTrain + i, lastGuesses[4 * i + 2], right ? 1 : 0, digits.labels[digits.nTrain + i]], 4 * k++);
    }
    digitsEngine.renderer.setTiles(t.subarray(0, 4 * k));
  };

  /** Read-backs: loss and accuracy (often), then the step probe, gradients and weights (less often). One at a time. */
  const measure = (t: number) => {
    const engine = engineNow();
    if (!config || !engine || measuring) return;
    const steps = engine.steps;
    const fast = steps !== measuredAt && t - lastMeasure >= MEASURE_INTERVAL_MS;
    const slow = steps !== slowAt && t - lastSlow >= SLOW_MEASURE_INTERVAL_MS;
    if (!fast && !slow) return;
    measuring = true;
    const myEpoch = epoch;
    const view = config.view;
    const run = async () => {
      if (fast) {
        lastMeasure = t;
        measuredAt = steps;
        let point: CurvePoint;
        if (view === "playground" && plane) {
          const out = await engine.evaluate(0, plane.n);
          const tr = summarise(out, 0, plane.nTrain);
          const te = summarise(out, plane.nTrain, plane.n - plane.nTrain);
          point = { step: steps, epoch: plane.stream.epochs, trainLoss: tr.loss, trainAcc: tr.acc, testLoss: te.loss, testAcc: te.acc };
        } else if (digits && digitsEngine) {
          const trainOut = await engine.evaluate(0, Math.min(DIGITS_TRAIN_SAMPLE, digits.nTrain));
          const tr = summarise(trainOut, 0, Math.min(DIGITS_TRAIN_SAMPLE, digits.nTrain));
          const nTest = digits.count - digits.nTrain;
          const testOut = (await engine.evaluate(digits.nTrain, nTest)).slice();
          const te = summarise(testOut, 0, nTest);
          if (myEpoch !== epoch) return;
          lastGuesses = testOut;
          const truth = digits.labels.subarray(digits.nTrain);
          const guess = new Uint8Array(nTest);
          for (let i = 0; i < nTest; i++) guess[i] = testOut[4 * i + 2];
          digitsConfusion = confusion(truth, guess);
          mistakes = nTest - Math.round(te.acc * nTest);
          updateTiles();
          point = { step: steps, epoch: digitsEngine.stream.epochs, trainLoss: tr.loss, trainAcc: tr.acc, testLoss: te.loss, testAcc: te.acc };
        } else return;
        if (myEpoch !== epoch) return;
        now = point;
        start ??= point;
        pushHistory(history, point);
      }
      if (slow) {
        lastSlow = t;
        slowAt = steps;
        const params = (await engine.readParams()).slice();
        const { grad } = await engine.readGrads();
        const g = layerGradientSizes(engine.layout, grad);
        if (myEpoch !== epoch) return;
        if (view === "playground" && plane) {
          planeParams = params;
          spread = neuronSpread(engine.layout, params);
          gradients = steps > 0 ? g : null;
          if (steps > 0) {
            const out = await engine.evaluate(0, plane.nTrain, PROBE_S, lastLr);
            if (myEpoch !== epoch) return;
            probe = PROBE_S.map((_, k) => summarise(out, k * plane!.nTrain, plane!.nTrain).loss);
          }
        } else {
          digitsParams = params;
          const L0 = engine.layout.layers[0];
          const rows = engine.spec.hidden.length ? Math.min(HIDDEN_TEMPLATES, L0.nOut) : L0.nOut;
          let max = 0;
          for (let k = L0.w; k < L0.w + rows * L0.nIn; k++) max = Math.max(max, Math.abs(params[k]));
          templateScale = max;
          gradients = steps > 0 ? g : null;
        }
      }
    };
    run().then(
      () => {
        measuring = false;
        if (myEpoch === epoch) emit(performance.now(), true);
      },
      () => {
        measuring = false;
      },
    );
  };

  const loop = frameLoop((frame) => {
    if (!config) return;
    const c = config;
    let steps = stepsRequested;
    stepsRequested = 0;
    if (playing && !diverged()) steps += c.speed;
    train(steps);
    const t = performance.now();
    const dpr = cssScale();
    const [w, h] = surface.size;
    let body: ((pass: FramePass) => void) | null = null;
    if (c.view === "playground" && plane) {
      if (mapDirty && t - lastMap >= MAP_INTERVAL_MS) {
        plane.engine.drawMap(c.plane.features);
        mapDirty = false;
        lastMap = t;
      }
      const columns = [c.plane.spec.inputs, ...c.plane.spec.hidden, 1];
      body = plane.renderer.prepare(surface, sceneLayout(w, h, columns, dpr), { showTest: c.showTest, hasMap: true, dpr });
    } else if (c.view === "digits" && digitsEngine) {
      const templates = c.digits.spec.hidden.length ? Math.min(HIDDEN_TEMPLATES, c.digits.spec.hidden[0]) : 10;
      body = digitsEngine.renderer.prepare(surface, digitsLayout(w, h, templates, dpr), { templates, firstWeight: 0, scale: templateScale, hasParams: !!digitsParams });
    }
    measure(t);
    emit(t);
    frame.pass({ target: surface, clear: [0.035, 0.043, 0.06, 1] }, (pass) => body?.(pass));
  });

  const handle: NetHandle = {
    configure(next) {
      const prev = config;
      config = next;
      if (next.view === "playground") {
        const key = JSON.stringify(next.plane);
        if (prev?.view !== "playground" || key !== planeKey || !plane) {
          planeKey = key;
          digitsEngine?.renderer.dispose();
          digitsEngine?.engine.dispose();
          digitsEngine = null;
          digitsKey = "";
          loadPlane(next.plane);
        }
      } else {
        requestDigits();
        const key = JSON.stringify(next.digits.spec) + next.digits.seed;
        if (prev?.view !== "digits" || key !== digitsKey) {
          plane?.renderer.dispose();
          plane?.engine.dispose();
          plane = null;
          planeKey = "";
          digitsKey = key;
          if (digits) loadDigitsNet();
          else {
            clearMeasurements();
            onStats?.(null);
          }
        }
      }
      if (prev && sameJson(prev.opt, next.opt) && sameJson(prev.digits.opt, next.digits.opt)) return;
      retune();
    },
    play: () => {
      playing = true;
    },
    pause: () => {
      playing = false;
    },
    step: () => {
      stepsRequested += 1;
    },
    reset: () => reload(),
    setTileMode(mode) {
      tileMode = mode;
      updateTiles();
    },
    labels() {
      if (!config) return [];
      const r = cssScale();
      const [w, h] = surface.size;
      if (config.view === "playground") {
        const columns = [config.plane.spec.inputs, ...config.plane.spec.hidden, 1];
        const s = sceneLayout(w, h, columns, r);
        const top = (column: number) => nodeCenter(s, column, 0)[1] - s.node / 2;
        const names = columns.map((_, k) => (k === 0 ? "Inputs" : k === columns.length - 1 ? "Output" : columns.length > 4 ? `${k}` : `Layer ${k}`));
        return names.map((text, k) => ({ text, x: nodeCenter(s, k, 0)[0] / r, y: (top(k) - 16 * r) / r }));
      }
      const templates = config.digits.spec.hidden.length ? Math.min(HIDDEN_TEMPLATES, config.digits.spec.hidden[0]) : 10;
      const L = digitsLayout(w, h, templates, r);
      const narrow = w / r < 640;
      const hidden = config.digits.spec.hidden.length > 0;
      const strip = hidden ? (narrow ? "What the first neurons look for" : "What the first hidden neurons look for (blue: ink here counts for, orange: against)") : narrow ? "What each digit 0–9 looks for" : "What each digit 0–9 looks for (blue: ink here counts for it, orange: against)";
      const grid = tileMode === "mistakes" ? (narrow ? "Its mistakes: guess, then truth" : "Test digits it gets wrong (its guess, then the truth)") : narrow ? "Unseen test digits, with its guess" : "Test digits it has never seen, with its guess";
      return [
        { text: strip, x: (L.strip.x + L.strip.w / 2) / r, y: (L.strip.y - 16 * r) / r },
        { text: grid, x: (L.grid.x + L.grid.w / 2) / r, y: (L.grid.y - 16 * r) / r },
      ];
    },
    probabilityAt(xCss, yCss) {
      if (!config || config.view !== "playground" || !planeParams) return null;
      const r = cssScale();
      const [w, h] = surface.size;
      const s = sceneLayout(w, h, [config.plane.spec.inputs, ...config.plane.spec.hidden, 1], r);
      const at = planeAt(s, xCss * r, yCss * r);
      if (!at) return null;
      const spec = config.plane.spec;
      const lay = plane!.engine.layout;
      return { x: at[0], y: at[1], p: probabilityAt(spec, lay, planeParams, config.plane.features, at[0], at[1]) };
    },
    classify(image) {
      if (!config || config.view !== "digits" || !digitsParams || !digitsEngine) return null;
      const engine = digitsEngine.engine;
      const logits = forward(engine.spec, engine.layout, digitsParams, image).at(-1)!;
      return Array.from(head(logits, 0).p);
    },
    dispose() {
      disposed = true;
      loop.stop();
      plane?.renderer.dispose();
      plane?.engine.dispose();
      digitsEngine?.renderer.dispose();
      digitsEngine?.engine.dispose();
    },
  };
  return handle;
}
