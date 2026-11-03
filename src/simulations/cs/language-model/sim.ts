import type { FramePass } from "vgpu";
import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import { pickLetter, sampleUniform } from "./cpu";
import { letterVectors, project2d } from "./embedding";
import { createEngine } from "./engine";
import type { Engine } from "./engine";
import { initParams } from "./model";
import type { ModelSpec } from "./model";
import type { TableResult } from "./ngram";
import { COPY_SOURCE_MAX, longestCopy, newNames, repetition } from "./sample";
import { buildAtlas, cellSize, COLOUR, createScreen, displaySymbol, EXTRA_GLYPHS, TINT } from "./screen";
import type { Screen, TextItem } from "./screen";
import { modelGuesses } from "./shannon";
import { countTables } from "./tables";
import { batchStream, CONTEXT_MARGIN, copyingCorpus, decode, encode, loadText, namesCorpus, ownCorpus, passageStart, seedContext, shakespeareCorpus, sonnetCorpus, windowStarts } from "./text";
import type { Corpus, TextId } from "./text";
import { fontFor, predictScene, shannonScene, surpriseItems, tablesScene, trainScene } from "./views";
import type { Canvas, Glyphs, Label, Scene } from "./views";

export type LmView = "train" | "predict" | "tables" | "shannon";

/** Evaluation windows: a sample of training windows, of held-out windows, then a held-out passage to display. */
export const EVAL_TRAIN = 64;
export const EVAL_HELD = 128;
export const DISPLAY_MAX = 1024;
/** Writing: lines at once and letters per line. */
export const LINES = 4;
export const LINE_LETTERS = 240;
const EVAL_INTERVAL_MS = 700;
const READ_INTERVAL_MS = 120;
const STATS_INTERVAL_MS = 150;
const PREDICT_INTERVAL_MS = 600;
const HOLD_MS = 1800;
const HISTORY = 300;
/** Auto speed: keep the GPU work of a frame near this. */
const FRAME_BUDGET_MS = 14;
const MAX_STEPS_PER_FRAME = 64;

export interface LmConfig {
  view: LmView;
  text: TextId;
  /** Your own text (text = "own"); kept in the browser, never in a link. */
  ownText: string;
  spec: Omit<ModelSpec, "V">;
  batch: number;
  lr: number;
  seed: number;
  /** Training steps per frame; 0 = as many as fit in a frame. */
  speed: number;
  temperature: number;
  /** Count tables: the context shown. */
  tableK: number;
}

export interface CurvePoint {
  step: number;
  train: number;
  held: number;
  /**
   * Copying game: held-out loss on the letters a model could copy (their earlier copy is inside its context), and on
   * the fresh ones (first copies).
   */
  repeated?: number;
  fresh?: number;
}

export interface Refs {
  uniform: number;
  unigram: number | null;
  bigram: number | null;
  /** The best count table on the same held-out letters. */
  best: { k: number; loss: number } | null;
}

export interface WritingStats {
  temperature: number;
  /** The passages as written so far. */
  lines: string[];
  letters: number;
  /** Share of repeated 8-letter runs. */
  repetition: number;
  /** The model's own surprise at what it wrote (nats per letter). */
  surprise: number | null;
  /** Longest run copied verbatim from the training text (short texts only). */
  copy: number | null;
  /** Names: complete names written, and how many are not in the training list. */
  names: { total: number; fresh: number } | null;
}

export interface LmStats {
  status: "loading" | "ready" | "error";
  error?: string;
  text: TextId;
  vocab: number;
  letters: number;
  heldOut: boolean;
  params: number;
  steps: number;
  /** Letters the model has trained on (steps × batch × window). */
  seen: number;
  stepsPerSecond: number;
  now: CurvePoint | null;
  history: CurvePoint[];
  refs: Refs;
  writing: WritingStats | null;
  tables: TableResult[];
  diverged: boolean;
}

export interface Prediction {
  prompt: number[];
  T: number;
  /** Scores for the next letter (temperature 1). */
  logits: number[];
  /** Per layer, [H][T][T] (transformer only). */
  attention: Float32Array[];
}

export interface ShannonState {
  /** Game number (each game is GAME_LENGTH letters further into the held-out passage). */
  game: number;
  revealed: number;
  /** Guesses each revealed letter took. */
  guesses: number[];
  /** Wrong guesses at the current letter. */
  wrong: number[];
}

export interface ShannonModel {
  game: number;
  /** The model's guesses (rank of the true letter) and its surprise in bits, per letter. */
  ranks: number[];
  bits: number[];
}

export interface LmOptions {
  onStats?: (stats: LmStats | null) => void;
  onPrediction?: (p: Prediction | null) => void;
}

export interface LmHandle extends SimHandle {
  configure(config: LmConfig): void;
  play(): void;
  pause(): void;
  step(): void;
  reset(): void;
  labels(): Label[];
  corpus(): Corpus | null;
  /** Predict view. */
  setPrompt(text: string): void;
  setFocus(focus: { layer: number; head: number }): void;
  /** Writes n letters after the prompt (one prediction at a time); resolves with the text written. */
  writeOn(n: number, temperature: number, seed: number, onLetter: (text: string) => void): Promise<string>;
  /** Shannon game. */
  shannonLetters(game: number, length: number): { context: number[]; letters: number[] } | null;
  shannonModel(game: number, length: number): Promise<ShannonModel | null>;
  setShannon(state: ShannonState | null): void;
  /** Where the model puts each letter (the main two directions of its letter vectors). */
  letterMap(): Promise<{ symbols: string[]; x: number[]; y: number[]; explained: number } | null>;
}

const corpusCache = new Map<string, Promise<Corpus>>();

function corpusFor(id: TextId, own: string): Promise<Corpus> | Corpus | null {
  if (id === "sonnet") return sonnetCorpus();
  if (id === "copying") return copyingCorpus();
  if (id === "own") return ownCorpus(own);
  const key = id;
  if (!corpusCache.has(key)) {
    const p = loadText(id).then((t) => (id === "names" ? namesCorpus(t) : shakespeareCorpus(t)));
    p.catch(() => corpusCache.delete(key));
    corpusCache.set(key, p);
  }
  return corpusCache.get(key)!;
}

/** The held-out passage shown in the Train view (and played in Shannon's game): its windows and letters. */
export function displayPlan(c: Corpus, T: number) {
  const half = T / 2;
  const room = c.tokens.length - CONTEXT_MARGIN - T - 2;
  const length = Math.max(half, Math.min(DISPLAY_MAX, Math.floor((room - half) / half) * half));
  const p0 = passageStart(c, length + T);
  const windows = length / half;
  const starts = Uint32Array.from({ length: windows }, (_, w) => p0 + w * half);
  // Letter x (a target) is read from window ⌊(x − 1 − p0) / half⌋ − 1, where it sits at position ≥ T/2.
  const first = p0 + half + 1;
  return { starts, first, length };
}

/** The windows every measurement uses: training windows, held-out windows, then the displayed passage's. */
export function evaluationWindows(c: Corpus, T: number) {
  const display = displayPlan(c, T);
  const starts = new Uint32Array(EVAL_TRAIN + EVAL_HELD + display.starts.length);
  starts.set(windowStarts(c, T, "train", EVAL_TRAIN, 11));
  starts.set(windowStarts(c, T, "heldOut", EVAL_HELD, 12), EVAL_TRAIN);
  starts.set(display.starts, EVAL_TRAIN + EVAL_HELD);
  // The held-out letters measured (the count tables' reference lines use exactly these).
  const heldTargets = new Uint32Array(EVAL_HELD * T);
  for (let w = 0; w < EVAL_HELD; w++) for (let t = 0; t < T; t++) heldTargets[w * T + t] = starts[EVAL_TRAIN + w] + t + 1;
  return { starts, display, heldTargets };
}

/** Training and held-out loss (and, for the copying game, the copyable and fresh letters) from evaluation losses. */
export function curvePoint(c: Corpus, T: number, starts: Uint32Array, losses: ArrayLike<number>, step: number): CurvePoint {
  const mean = (from: number, n: number) => {
    let s = 0;
    for (let i = from; i < from + n; i++) s += losses[i];
    return s / n;
  };
  const point: CurvePoint = { step, train: mean(0, EVAL_TRAIN * T), held: mean(EVAL_TRAIN * T, EVAL_HELD * T) };
  if (c.repeated) {
    let rs = 0;
    let rn = 0;
    let fs = 0;
    let fn = 0;
    const nl = c.vocab.indexOf("\n");
    for (let w = 0; w < EVAL_HELD; w++) {
      for (let t = 0; t < T; t++) {
        const target = starts[EVAL_TRAIN + w] + t + 1;
        const x = losses[(EVAL_TRAIN + w) * T + t];
        if (c.repeated[target]) {
          // Copyable only if the letter's earlier copy is inside the window the model sees.
          if (t < (c.copyFrom?.[target] ?? 0)) continue;
          rs += x;
          rn++;
        } else if (c.tokens[target] !== nl) {
          fs += x;
          fn++;
        }
      }
    }
    point.repeated = rs / Math.max(1, rn);
    point.fresh = fs / Math.max(1, fn);
  }
  return point;
}

const sameJson = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

export function createLanguageModel({ gpu, canvas, surface, loop: frameLoop }: SimContext, { onStats, onPrediction }: LmOptions = {}): LmHandle {
  let config: LmConfig | null = null;
  let playing = false;
  let stepsRequested = 0;
  let disposed = false;
  let epoch = 0;

  // Text and model.
  let corpus: Corpus | null = null;
  let corpusKey = "";
  let status: LmStats["status"] = "loading";
  let error: string | undefined;
  let engine: Engine | null = null;
  let engineKey = "";
  let stream: ReturnType<typeof batchStream> | null = null;
  let evalStarts: Uint32Array | null = null;
  let heldTargets: Uint32Array | null = null;
  let evalKey = "";
  let display: ReturnType<typeof displayPlan> | null = null;
  let trainingNames: Set<string> | null = null;
  let sourceText: string | null = null;

  // Measurements.
  let history: CurvePoint[] = [];
  let now: CurvePoint | null = null;
  let displaySurprise: Float32Array | null = null;
  let measuring = false;
  let measuredAt = -1;
  let lastMeasure = -Infinity;
  /** Grows on a slow GPU: an evaluation never takes more than a fifth of the time. */
  let evalInterval = EVAL_INTERVAL_MS;
  /** An evaluation in progress (on a slow GPU it spans several frames). */
  let evalJob: { engine: Engine; next: number; steps: number; t0: number } | null = null;
  /** Measured cost of one evaluation chunk (ms), and the time between frames (ms, smoothed). */
  let chunkMs = 0;
  let frameMs = 16.7;
  let lastFrameAt = 0;
  let lastEmit = 0;
  let refs: Refs = { uniform: 0, unigram: null, bigram: null, best: null };
  let tables: TableResult[] = [];
  let cancelTables: (() => void) | null = null;

  // Speed.
  let stepsPerFrame = 2;
  /** On a slow GPU, no new work before this time (performance.now()). */
  let workResumesAt = 0;
  let rate = { steps: 0, since: performance.now(), value: 0 };

  // Writing.
  let writingSeed = 1;
  let writingTemp = 1;
  let writing: WritingStats | null = null;
  let writtenTokens: Uint32Array[] = [];
  let reading = false;
  let lastRead = 0;
  let fullSince = -1;
  let writingStarted = false;

  // Predict.
  let prompt: number[] = [];
  let prediction: Prediction | null = null;
  let predicting = false;
  let predictDirty = false;
  let lastPredict = 0;
  let predictedAt = -1;
  let focus = { layer: 0, head: 0 };
  let writingOn = 0; // bumps to cancel a writeOn in progress

  // Shannon.
  let shannon: ShannonState | null = null;

  // Drawing.
  let screen: Screen | null = createScreen(gpu);
  let atlasKey = "";
  let glyphs: Glyphs | null = null;
  let scene: Scene = { panels: [], labels: [] };
  let sceneKey = "";
  let sceneVersion = 0;

  // The atlas is drawn with the page's monospace font: draw it again once web fonts have loaded.
  if (typeof document !== "undefined") document.fonts?.ready.then(() => {
    atlasKey = "";
  });

  const cssScale = () => {
    const rect = canvas.getBoundingClientRect();
    return rect.width > 0 ? surface.size[0] / rect.width : 1;
  };

  const emit = (t: number, force = false) => {
    if (!config || (!force && t - lastEmit < STATS_INTERVAL_MS)) return;
    lastEmit = t;
    if (!corpus || !engine) {
      onStats?.({ status, error, text: config.text, vocab: 0, letters: 0, heldOut: false, params: 0, steps: 0, seen: 0, stepsPerSecond: 0, now: null, history: [], refs, writing: null, tables, diverged: false });
      return;
    }
    onStats?.({
      status,
      error,
      text: corpus.id,
      vocab: corpus.vocab.length,
      letters: corpus.tokens.length,
      heldOut: corpus.heldOut,
      params: engine.plan.lay.P,
      steps: engine.steps,
      seen: engine.steps * engine.plan.B * engine.spec.T,
      stepsPerSecond: rate.value,
      now,
      history: history.slice(),
      refs,
      writing,
      tables: tables.slice(),
      diverged: diverged(),
    });
  };

  const diverged = () => !!now && !Number.isFinite(now.train);

  const clearMeasurements = () => {
    evalJob = null;
    measuring = false;
    history = [];
    now = null;
    displaySurprise = null;
    measuredAt = -1;
    lastMeasure = -Infinity;
    writing = null;
    writtenTokens = [];
    prediction = null;
    predictedAt = -1;
    predictDirty = true;
    writingStarted = false;
    fullSince = -1;
    rate = { steps: 0, since: performance.now(), value: 0 };
    sceneVersion++;
  };

  const specOf = (c: LmConfig, corp: Corpus): ModelSpec => ({ ...c.spec, V: corp.vocab.length });

  /** The glyph atlas for the corpus's symbols at the current view's font size. */
  const ensureAtlas = () => {
    if (!corpus || !config || !screen) return;
    const dpr = cssScale();
    const cssW = surface.size[0] / dpr;
    const font = fontFor(config.view, cssW);
    const { w, h } = cellSize(font, dpr);
    const key = `${corpusKey}|${w}x${h}`;
    if (key === atlasKey) return;
    atlasKey = key;
    const family = (typeof document !== "undefined" && getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim()) || "ui-monospace, monospace";
    const symbols = [...corpus.vocab.map(displaySymbol), ...EXTRA_GLYPHS];
    try {
      screen.setAtlas(buildAtlas(symbols, w, h, family));
    } catch {
      // No 2D canvas (headless): text is not drawn.
    }
    const V = corpus.vocab.length;
    glyphs = {
      newline: corpus.vocab.indexOf("\n"),
      space: corpus.vocab.indexOf(" "),
      mark: V,
      hidden: V + 1,
      newlineMode: corpus.id === "names" || corpus.id === "copying" ? "mark" : "break",
    };
    sceneVersion++;
  };

  const startTables = () => {
    cancelTables?.();
    tables = [];
    refs = { uniform: corpus ? Math.log(corpus.vocab.length) : 0, unigram: null, bigram: null, best: null };
    if (!corpus || !heldTargets) return;
    const targets = heldTargets;
    const myEpoch = epoch;
    const c = corpus;
    cancelTables = countTables(c, targets, 360, (r) => {
      if (myEpoch !== epoch || c !== corpus) return;
      tables.push(r);
      tables.sort((a, b) => a.k - b.k);
      const at = (k: number) => tables.find((x) => x.k === k)?.rowsLoss ?? null;
      const ready = tables.filter((x) => x.rowsLoss !== undefined);
      const best = ready.length ? ready.reduce((a, b) => (b.heldOutLoss < a.heldOutLoss ? b : a)) : null;
      refs = { uniform: Math.log(c.vocab.length), unigram: at(0), bigram: at(1), best: tables.length === 7 && best ? { k: best.k, loss: best.rowsLoss! } : null };
      sceneVersion++;
      emit(performance.now(), true);
    });
  };

  /** A new model (and, if the text changed, new evaluation windows and tables). */
  const reload = () => {
    if (!config || !corpus) return;
    epoch++;
    writingOn++;
    engine?.dispose();
    engine = null;
    const c = config;
    const spec = specOf(c, corpus);
    engineKey = JSON.stringify([c.spec, c.batch, c.seed]);
    const textChanged = !display || evalStarts === null || evalKey !== `${corpusKey}|${spec.T}`;
    if (textChanged) {
      const w = evaluationWindows(corpus, spec.T);
      display = w.display;
      evalStarts = w.starts;
      heldTargets = w.heldTargets;
      evalKey = `${corpusKey}|${spec.T}`;
    }
    try {
      engine = createEngine(gpu, spec, corpus.tokens, { B: c.batch, G: LINES, W: LINE_LETTERS, E: evalStarts!.length });
      engine.load(initParams(spec, c.seed));
      stream = batchStream(corpus, spec.T, c.batch, c.seed);
      status = "ready";
      error = undefined;
    } catch (e) {
      status = "error";
      error = e instanceof Error ? e.message : String(e);
    }
    clearMeasurements();
    if (textChanged) startTables();
    onStats?.(null);
    onPrediction?.(null);
    emit(performance.now(), true);
  };

  const loadCorpus = (c: LmConfig) => {
    const key = c.text === "own" ? `own:${c.ownText.length}:${c.ownText.slice(0, 64)}:${c.ownText.slice(-64)}` : c.text;
    if (key === corpusKey && corpus) return false;
    corpusKey = key;
    epoch++;
    cancelTables?.();
    engine?.dispose();
    engine = null;
    corpus = null;
    display = null;
    evalStarts = null;
    evalKey = "";
    trainingNames = null;
    sourceText = null;
    atlasKey = "";
    status = "loading";
    clearMeasurements();
    onStats?.(null);
    onPrediction?.(null);
    const got = corpusFor(c.text, c.ownText);
    const accept = (corp: Corpus) => {
      if (disposed || key !== corpusKey) return;
      corpus = corp;
      if (corp.id === "names") trainingNames = new Set(decode(corp, corp.tokens.subarray(0, corp.nTrain)).split("\n"));
      if (corp.tokens.length <= COPY_SOURCE_MAX) sourceText = decode(corp, corp.tokens.subarray(0, corp.nTrain));
      reload();
    };
    if (got === null) {
      status = "error";
      error = "Paste at least 400 characters of your own text.";
      emit(performance.now(), true);
    } else if (got instanceof Promise) {
      got.then(accept, (e) => {
        if (key !== corpusKey) return;
        status = "error";
        error = `The text could not be loaded (${e instanceof Error ? e.message : String(e)}).`;
        emit(performance.now(), true);
      });
    } else accept(got);
    return true;
  };

  // ── Training ──

  const train = (n: number) => {
    if (!engine || !stream || !config || n <= 0) return 0;
    for (let k = 0; k < n; k++) engine.step(stream.next(), config.lr);
    return n;
  };

  /**
   * GPU time a frame's computing may take: 14 ms on a normal display, or half the time between frames on a device so
   * slow that drawing alone takes longer (computing more than that would only slow the page down further).
   */
  const computeBudget = () => Math.max(FRAME_BUDGET_MS, 0.5 * frameMs);

  /**
   * Pacing. After a frame's GPU work is submitted, time it: on a fast GPU, move the steps per frame towards the
   * budget; on a slow one (a frame of work longer than the budget), leave the GPU idle for half as long again before
   * the next work, so the page keeps drawing and answering while it trains.
   */
  const queue = () => (gpu.gpu as unknown as { queue: { onSubmittedWorkDone(): Promise<void> } }).queue;
  /** When the GPU finishes the work submitted so far (performance.now() at that moment). */
  const whenIdle = () => queue().onSubmittedWorkDone().then(() => performance.now());

  /**
   * Times the computing submitted this frame: from when it was submitted (or, if the GPU was still drawing the last
   * frame, from when that finished) until it is done. `earlier` resolves when the work before it finished.
   */
  const probeWork = (trainingSteps: number, chunks: number, submitted: number, earlier: Promise<number>) => {
    const myEpoch = epoch;
    Promise.all([earlier, whenIdle()]).then(
      ([before, done]) => {
        if (myEpoch !== epoch) return;
        const ms = done - Math.max(before, submitted);
        const budget = computeBudget();
        if (chunks > 0 && trainingSteps === 0) chunkMs = ms / chunks;
        if (ms > budget * 2) workResumesAt = done + ms * 0.5;
        if (trainingSteps <= 0 || config?.speed !== 0) return;
        if (ms > budget * 1.3) stepsPerFrame = Math.max(1, Math.floor(stepsPerFrame * 0.75));
        else if (ms < budget * 0.7) stepsPerFrame = Math.min(MAX_STEPS_PER_FRAME, Math.ceil(stepsPerFrame * 1.25));
      },
      () => {},
    );
  };

  // ── Measuring ──

  /**
   * Evaluation. On a fast GPU every chunk goes in one submit; on a slow one, as many chunks a frame as fit the
   * computing budget (timed by probeWork), with training held until the last chunk so all see the same weights.
   */
  /** Runs the evaluation's next chunks (starting one if due); returns how many chunks it submitted. */
  const measure = (t: number): number => {
    if (!engine || !config || !corpus || !evalStarts) return 0;
    if (evalJob) {
      if (evalJob.engine !== engine) {
        evalJob = null;
        measuring = false;
        return 0;
      }
      const from = evalJob.next;
      const perFrame = chunkMs > 0 ? Math.max(1, Math.floor(computeBudget() / chunkMs)) : engine.evalChunks;
      const to = Math.min(engine.evalChunks, from + perFrame);
      engine.evaluateChunks(evalStarts, from, to);
      evalJob.next = to;
      if (to >= engine.evalChunks) finishMeasure(evalJob.t0, evalJob.steps);
      return to - from;
    }
    if (measuring) return 0;
    const steps = engine.steps;
    if (steps === measuredAt || t - lastMeasure < evalInterval) return 0;
    measuring = true;
    lastMeasure = t;
    measuredAt = steps;
    evalJob = { engine, next: 0, steps, t0: t };
    return measure(t);
  };

  const finishMeasure = (t: number, steps: number) => {
    evalJob = null;
    const myEpoch = epoch;
    const e = engine!;
    const T = e.spec.T;
    const c = corpus!;
    const ev = evalStarts!;
    e.readLosses().then(
      (losses) => {
        measuring = false;
        const took = performance.now() - t;
        evalInterval = Math.max(EVAL_INTERVAL_MS, Math.min(10_000, 5 * took));
        if (myEpoch !== epoch) return;
        const point = curvePoint(c, T, ev, losses, steps);
        // The displayed passage: each letter from the window where it has at least T/2 letters of context.
        if (display) {
          const half = T / 2;
          const p0 = display.first - half - 1;
          const out = new Float32Array(display.length);
          for (let i = 0; i < display.length; i++) {
            const x = display.first + i;
            const w = Math.floor((x - 1 - p0) / half) - 1;
            const tt = x - 1 - p0 - w * half;
            out[i] = losses[(EVAL_TRAIN + EVAL_HELD + w) * T + tt];
          }
          displaySurprise = out;
        }
        now = point;
        history.push(point);
        if (history.length > HISTORY) history = history.filter((_, k) => k % 2 === 0 || k === history.length - 1);
        sceneVersion++;
        emit(performance.now(), true);
      },
      () => {
        measuring = false;
      },
    );
    return true;
  };

  // ── Writing ──

  const startWriting = () => {
    if (!engine || !corpus || !config) return;
    writingSeed++;
    writingTemp = config.temperature;
    const T = engine.spec.T;
    const contexts = Array.from({ length: LINES }, (_, g) => seedContext(corpus!, CONTEXT_MARGIN + T, writingSeed * 31 + g));
    engine.startWriting(contexts, writingTemp, writingSeed);
    writtenTokens = [];
    fullSince = -1;
    writingStarted = true;
    sceneVersion++;
  };

  const readWriting = (t: number) => {
    if (!engine || !corpus || reading || t - lastRead < READ_INTERVAL_MS || !writingStarted) return;
    reading = true;
    lastRead = t;
    const myEpoch = epoch;
    const e = engine;
    const c = corpus;
    const temp = writingTemp;
    Promise.all([e.written(), e.surprise()]).then(
      ([lines, sur]) => {
        reading = false;
        if (myEpoch !== epoch || e !== engine) return;
        writtenTokens = lines;
        const texts = lines.map((l) => decode(c, l));
        const n = lines[0]?.length ?? 0;
        let surprise = 0;
        let count = 0;
        sur.forEach((s, g) => {
          for (let i = 0; i < Math.min(n, lines[g].length); i++) {
            surprise += s[i];
            count++;
          }
        });
        let names: WritingStats["names"] = null;
        if (trainingNames) {
          const all = texts.map((tx) => newNames(tx, trainingNames!));
          names = { total: all.reduce((a, x) => a + x.total, 0), fresh: all.reduce((a, x) => a + x.fresh, 0) };
        }
        writing = {
          temperature: temp,
          lines: texts,
          letters: n,
          repetition: texts.length ? texts.reduce((a, tx) => a + repetition(tx), 0) / texts.length : 0,
          surprise: count ? surprise / count : null,
          copy: sourceText ? Math.max(0, ...texts.map((tx) => longestCopy(tx, sourceText!).length)) : null,
          names,
        };
        if (n >= LINE_LETTERS && fullSince < 0) fullSince = performance.now();
        sceneVersion++;
      },
      () => {
        reading = false;
      },
    );
  };

  // ── Predicting ──

  const padToken = () => Math.max(0, corpus?.vocab.indexOf("\n") ?? 0);

  const predict = (t: number): boolean => {
    if (!engine || !config || predicting || config.view !== "predict") return false;
    const due = predictDirty || (engine.steps !== predictedAt && t - lastPredict >= PREDICT_INTERVAL_MS);
    if (!due) return false;
    predicting = true;
    predictDirty = false;
    lastPredict = t;
    predictedAt = engine.steps;
    const myEpoch = epoch;
    const p = prompt.slice();
    engine.predict(p, padToken()).then(
      (r) => {
        predicting = false;
        if (myEpoch !== epoch) return;
        prediction = { prompt: p, T: r.T, logits: Array.from(r.logits), attention: r.attention };
        onPrediction?.(prediction);
        sceneVersion++;
      },
      () => {
        predicting = false;
      },
    );
    return true;
  };

  // ── Drawing ──

  const buildScene = (): Scene => {
    if (!config || !corpus || !glyphs || !screen?.atlas) return { panels: [], labels: [] };
    const dpr = cssScale();
    const [w, h] = surface.size;
    const a = screen.atlas;
    const cv: Canvas = { w, h, dpr, cw: a.w, ch: a.h };
    const narrow = w / dpr < 640;
    if (config.view === "train") {
      const lines = Array.from({ length: LINES }, (_, g) => {
        const toks = writtenTokens[g] ?? new Uint32Array(0);
        const items = surpriseItems(toks, null);
        if (toks.length < LINE_LETTERS) items.push({ glyph: glyphs!.hidden, colour: COLOUR.dim, tint: TINT.cursor, strength: 0.2 });
        return items;
      });
      const passage = display ? surpriseItems(corpus.tokens.subarray(display.first, display.first + display.length), displaySurprise) : [];
      const writingCaption = narrow ? `What it writes now (temperature ${writingTemp.toFixed(1)})` : `What it writes now, letter by letter (temperature ${writingTemp.toFixed(1)})`;
      const passageCaption = narrow ? "Its surprise at unseen text (brighter = more)" : "How surprised it is by text it has never seen: the brighter a letter, the less it expected it";
      return trainScene(cv, glyphs, lines, passage, { writing: writingCaption, passage: passageCaption });
    }
    if (config.view === "predict") {
      const T = prediction?.T ?? 0;
      const window = prediction ? prediction.prompt.slice(-T) : prompt.slice(-config.spec.T);
      const kind = config.spec.kind;
      // What the model reads: for a transformer, the focused head's attention from the last letter.
      let weights: number[] | null = null;
      if (prediction && kind === "gpt" && prediction.attention[focus.layer]) {
        const att = prediction.attention[focus.layer];
        const row = (focus.head * T + (T - 1)) * T;
        weights = Array.from({ length: T }, (_, j) => att[row + j]);
        const mx = Math.max(...weights, 1e-9);
        weights = weights.map((x) => x / mx);
      }
      const seen = kind === "bigram" ? 1 : kind === "mlp" ? config.spec.k : window.length;
      const items: TextItem[] = window.map((tok, j) => {
        const strength = weights ? weights[j] : j >= window.length - seen ? 1 : 0;
        return { glyph: tok, colour: j >= window.length - seen ? COLOUR.bright : COLOUR.dim, tint: TINT.attention, strength };
      });
      items.push({ glyph: glyphs.hidden, colour: COLOUR.sky, tint: TINT.cursor, strength: 0.6 });
      const caption = kind === "gpt" ? (narrow ? `Prompt, tinted by layer ${focus.layer + 1} head ${focus.head + 1}` : `The prompt it reads, tinted by where layer ${focus.layer + 1}, head ${focus.head + 1} looks from the last letter`) : kind === "mlp" ? `The prompt: it sees only the last ${config.spec.k} letters (bright)` : "The prompt: it sees only the last letter (bright)";
      const maps = kind === "gpt" && prediction?.attention.length ? { maps: prediction.attention, H: config.spec.H, T } : null;
      return predictScene(cv, glyphs, items, maps, focus, { prompt: caption, maps: narrow ? "Attention: rows read, columns are read" : "Every head's attention: each row is a letter, bright cells are the earlier letters it looks at" });
    }
    if (config.view === "tables") {
      const samples = Array.from({ length: 7 }, (_, k) => {
        const r = tables.find((x) => x.k === k);
        return r ? encode(corpus!, r.sample).map((glyph) => ({ glyph, colour: k === config!.tableK ? COLOUR.bright : COLOUR.text })) : null;
      });
      const captions = Array.from({ length: 7 }, (_, k) => {
        const r = tables.find((x) => x.k === k);
        const what = k === 0 ? "letter frequencies only" : k === 1 ? "looks at the previous letter" : `looks at the last ${k} letters`;
        return r ? `${what} · unseen text ${r.heldOutLoss.toFixed(2)} nats/letter` : `${what} · counting…`;
      });
      return tablesScene(cv, glyphs, samples, config.tableK, captions);
    }
    // Shannon.
    const items: TextItem[] = [];
    if (shannon && display) {
      const got = handle.shannonLetters(shannon.game, 50);
      if (got) {
        for (const tok of got.context) items.push({ glyph: tok, colour: COLOUR.dim });
        got.letters.slice(0, shannon.revealed).forEach((tok, i) => {
          const g = shannon!.guesses[i] ?? 1;
          items.push({ glyph: tok, colour: g === 1 ? COLOUR.emerald : COLOUR.text, tint: g === 1 ? TINT.right : TINT.guesses, strength: g === 1 ? 1 : Math.min(1, Math.log2(g) / 4) });
        });
        if (shannon.revealed < got.letters.length) items.push({ glyph: glyphs.hidden, colour: COLOUR.sky, tint: TINT.cursor, strength: 1 });
      }
    }
    return shannonScene(cv, glyphs, items, narrow ? "Guess the next letter" : "Guess the next letter of text the model has never seen (green: first guess; warmer: more guesses)");
  };

  const loop = frameLoop((frame) => {
    if (!config) return;
    const t = performance.now();
    if (lastFrameAt > 0) frameMs = 0.8 * frameMs + 0.2 * Math.min(2000, t - lastFrameAt);
    lastFrameAt = t;
    if (t >= workResumesAt) {
      const earlier = whenIdle();
      // No training while an evaluation spans frames (a requested step waits for it).
      let steps = 0;
      if (!evalJob) {
        steps = stepsRequested;
        stepsRequested = 0;
        if (playing && engine && !diverged() && status === "ready") steps += config.speed > 0 ? config.speed : stepsPerFrame;
      }
      const done = train(steps);
      if (done > 0) {
        rate.steps += done;
        if (t - rate.since >= 1000) rate = { steps: 0, since: t, value: (rate.steps * 1000) / (t - rate.since) };
      }
      // Writing: keep the lines going (while paused, finish them but start no new ones).
      let wrote = false;
      if (engine && config.view === "train" && status === "ready") {
        if (!writingStarted) startWriting();
        else if (fullSince >= 0 && playing && t - fullSince >= HOLD_MS) startWriting();
        else if (engine.writtenCount < LINE_LETTERS) {
          engine.write(2);
          wrote = true;
        }
      }
      const chunks = measure(t);
      const predicted = predict(t);
      if (done > 0 || wrote || chunks > 0 || predicted) probeWork(done, chunks, t, earlier);
    }
    if (engine && config.view === "train" && status === "ready") readWriting(t);
    if (t - rate.since >= 2000 && rate.steps === 0) rate = { steps: 0, since: t, value: 0 };
    emit(t);
    ensureAtlas();
    const key = `${config.view}|${sceneVersion}|${focus.layer}.${focus.head}|${config.tableK}`;
    if (key !== sceneKey) {
      sceneKey = key;
      scene = buildScene();
    }
    const body = screen ? screen.prepare(surface, scene.panels, key) : null;
    frame.pass({ target: surface, clear: [0.035, 0.043, 0.06, 1] }, (pass: FramePass) => body?.(pass));
  });

  const handle: LmHandle = {
    configure(next) {
      const prev = config;
      config = next;
      if (loadCorpus(next)) return;
      if (!corpus) return;
      const key = JSON.stringify([next.spec, next.batch, next.seed]);
      if (key !== engineKey || !engine) reload();
      else if (prev && prev.temperature !== next.temperature && next.view === "train") {
        startWriting();
      }
      if (prev?.view !== next.view) {
        atlasKey = "";
        sceneVersion++;
        predictDirty = true;
        if (next.view === "train" && engine && engine.writtenCount >= LINE_LETTERS) fullSince = performance.now() - HOLD_MS;
      }
      if (prev && !sameJson(prev.tableK, next.tableK)) sceneVersion++;
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
    labels: () => scene.labels,
    corpus: () => corpus,
    setPrompt(text) {
      if (!corpus) return;
      const p = encode(corpus, text);
      if (sameJson(p, prompt)) return;
      prompt = p;
      predictDirty = true;
      sceneVersion++;
    },
    setFocus(f) {
      focus = f;
      sceneVersion++;
    },
    async writeOn(n, temperature, seed, onLetter) {
      const myRun = ++writingOn;
      const myEpoch = epoch;
      let text = "";
      for (let i = 0; i < n; i++) {
        if (!engine || !corpus || myRun !== writingOn || myEpoch !== epoch || disposed) break;
        const r = await engine.predict(prompt, padToken());
        if (myRun !== writingOn || myEpoch !== epoch) break;
        const next = pickLetter(r.logits, 0, r.logits.length, temperature, sampleUniform(seed, 0, i));
        prompt = [...prompt, next];
        text += corpus.vocab[next];
        prediction = { prompt: prompt.slice(), T: r.T, logits: Array.from(r.logits), attention: r.attention };
        predictDirty = true;
        sceneVersion++;
        onLetter(decode(corpus, prompt));
      }
      return text;
    },
    shannonLetters(game, length) {
      if (!corpus || !display) return null;
      const from = display.first + game * length;
      if (from + length > display.first + display.length) return null;
      const before = Math.min(120, from);
      return { context: Array.from(corpus.tokens.subarray(from - before, from)), letters: Array.from(corpus.tokens.subarray(from, from + length)) };
    },
    async shannonModel(game, length) {
      const got = handle.shannonLetters(game, length);
      if (!got || !engine || !corpus) return null;
      const e = engine;
      const from = display!.first + game * length;
      const ranks: number[] = [];
      const bits: number[] = [];
      for (let i = 0; i < length; i++) {
        const ctx = Array.from(corpus.tokens.subarray(Math.max(0, from + i - e.spec.T), from + i));
        const r = await e.predict(ctx, padToken());
        if (e !== engine) return null;
        const truth = got.letters[i];
        const mx = Math.max(...r.logits);
        let sum = 0;
        for (const l of r.logits) sum += Math.exp(l - mx);
        bits.push((mx + Math.log(sum) - r.logits[truth]) / Math.LN2);
        ranks.push(modelGuesses(r.logits, truth));
      }
      return { game, ranks, bits };
    },
    setShannon(state) {
      shannon = state;
      sceneVersion++;
    },
    async letterMap() {
      if (!engine || !corpus) return null;
      const e = engine;
      const c = corpus;
      const params = await e.readParams();
      if (e !== engine) return null;
      const p = project2d(letterVectors(e.spec, e.plan.lay, params));
      return { symbols: c.vocab, ...p };
    },
    dispose() {
      disposed = true;
      epoch++;
      loop.stop();
      cancelTables?.();
      engine?.dispose();
      screen?.dispose();
      screen = null;
    },
  };
  return handle;
}
