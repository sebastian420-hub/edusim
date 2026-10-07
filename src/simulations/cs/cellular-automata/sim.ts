import { compute, effect, pingPongStorage, storage } from "vgpu";
import type { PingPongStorage, StorageBuffer } from "vgpu";
import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import computeShader from "./compute.wgsl";
import countShader from "./count.wgsl";
import renderShader from "./render.wgsl";
import { DEFAULT_PATTERN, patterns, rasterizePattern } from "./patterns";
import { parseRule } from "./rules";
import { STATS_SLOTS } from "./sim-constants";
import { PopulationTracker } from "./tracker";
import type { PatternStatus } from "./tracker";
import { workgroupsFor } from "./workgroup";

export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 40;
/** Cap on generations computed in one frame, so a fast speed on a slow GPU can't stall the page. */
const MAX_STEPS_PER_FRAME = 8;

export interface CellularAutomataParams {
  birth: string;
  survive: string;
  /** Generations per second. */
  speed: number;
  theme: number;
}

/** Generations of history shown in the population graph. */
const GRAPH_GENERATIONS = 240;
/** Minimum time between read-backs / stats updates (ms). */
const STATS_INTERVAL_MS = 100;

/** What the population graph and readouts show, measured on the GPU. */
export interface CellularAutomataStats {
  generation: number;
  /** Live cells now. */
  population: number;
  /** Total cells in the grid. */
  cells: number;
  /** Recent generations and their populations (parallel arrays, oldest first). */
  gens: number[];
  counts: number[];
  status: PatternStatus;
}

export interface CellularAutomataOptions {
  gridSize?: number;
  onGeneration?: (generation: number) => void;
  onZoom?: (zoom: number) => void;
  /**
   * Receives measurements of the running automaton (at most every 100 ms), or `null` the moment the history
   * is discarded (new pattern, edit) until the next measurement arrives. Omit it to skip measuring entirely.
   */
  onStats?: (stats: CellularAutomataStats | null) => void;
}

export interface CellularAutomataHandle extends SimHandle {
  setParams(params: Partial<CellularAutomataParams>): void;
  setGridSize(size: number): void;
  play(): void;
  pause(): void;
  step(): void;
  /** Re-applies the last initialisation (pattern, random fill or clear). */
  reset(): void;
  clear(): void;
  randomize(): void;
  loadPattern(name: string): void;
  /** Frames the whole grid. */
  fitGrid(): void;
  /** Frames the live cells (reads the grid back from the GPU). */
  fitPattern(): Promise<void>;
  panBy(dxCss: number, dyCss: number): void;
  zoomAt(factor: number, xCss: number, yCss: number): void;
  /** Writes cells along a stroke; call `endStroke` when the pointer is released. */
  paint(xCss: number, yCss: number, alive: boolean): void;
  endStroke(): void;
}

type InitialState = { kind: "pattern"; name: string } | { kind: "random" } | { kind: "clear" };

/** The public StorageBuffer type hides the byte-offset overload that the runtime supports. */
type OffsetWritable = StorageBuffer & { write(data: BufferSource, offset?: number): void };
type Destroyable = { destroy?: () => void };

export function createCellularAutomata(
  { gpu, canvas, surface, loop: frameLoop }: SimContext,
  { gridSize = 256, onGeneration, onZoom, onStats }: CellularAutomataOptions = {},
): CellularAutomataHandle {
  const computePass = compute(gpu, computeShader);
  const renderPass = effect(gpu, renderShader);
  // Measuring costs a GPU pass per generation, so only hosts that asked for stats pay for it.
  const makeMeter = () => ({
    pass: compute(gpu, countShader),
    ring: storage(gpu, STATS_SLOTS * 8) as OffsetWritable,
    tracker: new PopulationTracker(600),
  });
  const meter = onStats ? makeMeter() : undefined;

  let size = gridSize;
  let cells: PingPongStorage = pingPongStorage(gpu, size * size * 4);
  let params: CellularAutomataParams = { birth: "3", survive: "23", speed: 10, theme: 0 };
  let ruleBirth = parseRule(params.birth);
  let ruleSurvive = parseRule(params.survive);

  let playing = false;
  let generation = 0;
  let generationDirty = true;
  let stepDebt = 0; // seconds accumulated towards the next generation
  let initial: InitialState = { kind: "pattern", name: DEFAULT_PATTERN };

  // View: `center` is the grid coordinate at the middle of the canvas; `zoom` multiplies the
  // fit-to-canvas scale.
  let centerX = size / 2;
  let centerY = size / 2;
  let zoom = 1;
  let lastCell: [number, number] | null = null;

  const fitScale = () => Math.min(surface.size[0] / size, surface.size[1] / size);
  const cellPx = () => fitScale() * zoom;
  const resetView = () => {
    centerX = size / 2;
    centerY = size / 2;
    zoom = 1;
    onZoom?.(zoom);
  };

  /** Frames the cell rectangle [x0, x1) × [y0, y1) with some margin; small patterns get at least 24 cells across. */
  const fitBounds = (x0: number, y0: number, x1: number, y1: number) => {
    const w = Math.max((x1 - x0) * 1.6, 24);
    const h = Math.max((y1 - y0) * 1.6, 24);
    zoom = Math.min(MAX_ZOOM, Math.max(1, Math.min(size / w, size / h)));
    centerX = (x0 + x1) / 2;
    centerY = (y0 + y1) / 2;
    onZoom?.(zoom);
  };

  /** Bounding box of the live cells, or null for an empty grid. */
  const boundsOf = (data: Uint32Array): [number, number, number, number] | null => {
    let x0 = size;
    let y0 = size;
    let x1 = -1;
    let y1 = -1;
    for (let y = 0; y < size; y++) {
      const row = y * size;
      for (let x = 0; x < size; x++) {
        if (data[row + x] === 0) continue;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        y1 = y;
      }
    }
    return x1 < 0 ? null : [x0, y0, x1 + 1, y1 + 1];
  };

  /** Canvas CSS pixels -> device pixels. */
  const toDevice = (xCss: number, yCss: number): [number, number] => {
    const rect = canvas.getBoundingClientRect();
    return [(xCss * surface.size[0]) / rect.width, (yCss * surface.size[1]) / rect.height];
  };

  const toGrid = (xCss: number, yCss: number): [number, number] => {
    const [dx, dy] = toDevice(xCss, yCss);
    const scale = cellPx();
    return [centerX + (dx - surface.size[0] / 2) / scale, centerY + (dy - surface.size[1] / 2) / scale];
  };

  const setGeneration = (value: number) => {
    generation = value;
    generationDirty = true;
  };

  // ── Measurement: per-generation live-cell count and fingerprint, reduced on the GPU ──
  let epoch = 0; // bumped whenever the history becomes invalid (new pattern, edit): late read-backs are dropped
  let ingested = -1; // last generation whose stats are in the tracker
  let readInFlight = false;
  let lastRead = 0;
  let statsDirty = false; // cells were edited by hand: restart the history from the current generation
  let disposed = false;

  /** Queues the count/fingerprint of the current generation into its ring-buffer slot. */
  const recordCount = () => {
    if (!meter) return;
    const slot = generation % STATS_SLOTS;
    meter.ring.write(new Uint32Array([0, 0]), slot * 8); // clear the slot, then accumulate into it
    meter.pass
      .set({ cells: cells.read, stats: meter.ring, params: { width: size, height: size, slot, pad: 0 } })
      .dispatch(workgroupsFor(size), workgroupsFor(size), 1);
  };

  /** The history no longer describes the grid: start over from the current generation. */
  const restartStats = () => {
    if (!meter) return;
    epoch++;
    meter.tracker.reset();
    ingested = generation - 1;
    lastRead = 0; // read back promptly
    recordCount();
    onStats?.(null);
  };

  const emitStats = () => {
    if (!meter) return;
    const { tracker } = meter;
    const from = Math.max(0, tracker.counts.length - GRAPH_GENERATIONS);
    onStats?.({
      generation,
      population: tracker.counts[tracker.counts.length - 1] ?? 0,
      cells: size * size,
      gens: tracker.gens.slice(from),
      counts: tracker.counts.slice(from),
      status: tracker.status(),
    });
  };

  /** Reads the ring buffer back (at most every STATS_INTERVAL_MS) and ingests the generations it holds. */
  const pumpStats = () => {
    if (!meter) return;
    const now = performance.now();
    if (readInFlight || ingested >= generation || now - lastRead < STATS_INTERVAL_MS) return;
    readInFlight = true;
    lastRead = now;
    const myEpoch = epoch;
    const upTo = generation; // every count up to here was queued before this read
    void meter.ring.read().then(
      (buffer) => {
        readInFlight = false;
        if (disposed || myEpoch !== epoch) return;
        const ring = new Uint32Array(buffer);
        let from = ingested + 1;
        if (upTo - from + 1 > STATS_SLOTS) from = upTo - STATS_SLOTS + 1; // fell too far behind: keep the newest
        for (let g = from; g <= upTo; g++) meter.tracker.push(g, ring[(g % STATS_SLOTS) * 2], ring[(g % STATS_SLOTS) * 2 + 1]);
        ingested = upTo;
        emitStats();
      },
      () => {
        readInFlight = false; // the device went away mid-read (page closing); the GPU error handler reports real faults
      },
    );
  };

  const upload = (data: Uint32Array<ArrayBuffer>) => {
    cells.read.write(data);
    setGeneration(0);
    stepDebt = 0;
    restartStats();
  };

  const applyInitial = () => {
    switch (initial.kind) {
      case "pattern": {
        const name = initial.name;
        const pattern = patterns.find((p) => p.name === name);
        const data = pattern ? rasterizePattern(pattern, size, size) : new Uint32Array(size * size);
        upload(data);
        // Open readable: a library pattern fills the view instead of being a speck in a big grid.
        const b = boundsOf(data);
        if (b) fitBounds(...b);
        else resetView();
        break;
      }
      case "random": {
        const data = new Uint32Array(size * size);
        for (let i = 0; i < data.length; i++) data[i] = Math.random() > 0.8 ? 1 : 0;
        upload(data);
        resetView();
        break;
      }
      case "clear":
        upload(new Uint32Array(size * size));
        resetView();
        break;
    }
  };

  const runStep = () => {
    computePass.set({
      cellsIn: cells.read,
      cellsOut: cells.write,
      params: { width: size, height: size, rule_birth: ruleBirth, rule_survive: ruleSurvive },
    });
    computePass.dispatch(workgroupsFor(size), workgroupsFor(size), 1);
    cells.swap();
    setGeneration(generation + 1);
    recordCount();
  };

  const writeCell = (x: number, y: number, alive: boolean) => {
    if (x < 0 || y < 0 || x >= size || y >= size) return;
    (cells.read as OffsetWritable).write(new Uint32Array([alive ? 1 : 0]), (y * size + x) * 4);
    statsDirty = true;
  };

  applyInitial();

  const loop = frameLoop((frame, dt) => {
    if (statsDirty) {
      statsDirty = false;
      restartStats(); // the grid was edited: the old history no longer describes it
    }
    if (playing) {
      stepDebt += dt;
      const interval = 1 / params.speed;
      let steps = 0;
      while (stepDebt >= interval && steps < MAX_STEPS_PER_FRAME) {
        runStep();
        stepDebt -= interval;
        steps++;
      }
      if (steps === MAX_STEPS_PER_FRAME) stepDebt = 0;
    }

    if (generationDirty) {
      generationDirty = false;
      onGeneration?.(generation);
    }
    pumpStats();

    renderPass.set({
      cells: cells.read,
      params: {
        gridWidth: size,
        gridHeight: size,
        resolution: surface.size,
        center: [centerX, centerY],
        cellPx: cellPx(),
        theme: params.theme,
      },
    });
    frame.pass(surface, renderPass);
  });

  return {
    setParams(next) {
      params = { ...params, ...next };
      const birth = parseRule(params.birth);
      const survive = parseRule(params.survive);
      const rulesChanged = birth !== ruleBirth || survive !== ruleSurvive;
      ruleBirth = birth;
      ruleSurvive = survive;
      if (rulesChanged) restartStats(); // the recorded generations followed other rules
    },
    setGridSize(next) {
      if (next === size) return;
      const old = cells;
      size = next;
      cells = pingPongStorage(gpu, size * size * 4);
      // vgpu frees buffers on gpu.dispose(); release the old pair early so resizing doesn't accumulate.
      (old.read as Destroyable).destroy?.();
      (old.write as Destroyable).destroy?.();
      resetView();
      applyInitial();
    },
    play: () => {
      playing = true;
    },
    pause: () => {
      playing = false;
    },
    step: runStep,
    reset: applyInitial,
    clear() {
      initial = { kind: "clear" };
      applyInitial();
    },
    randomize() {
      initial = { kind: "random" };
      applyInitial();
    },
    loadPattern(name) {
      if (!patterns.some((p) => p.name === name)) return;
      initial = { kind: "pattern", name };
      applyInitial();
    },
    fitGrid: resetView,
    async fitPattern() {
      const data = new Uint32Array(await cells.read.read());
      const b = boundsOf(data);
      if (b) fitBounds(...b);
      else resetView();
    },
    panBy(dxCss, dyCss) {
      const [dx, dy] = toDevice(dxCss, dyCss);
      const scale = cellPx();
      centerX -= dx / scale;
      centerY -= dy / scale;
    },
    zoomAt(factor, xCss, yCss) {
      const [gx, gy] = toGrid(xCss, yCss);
      zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom * factor));
      // Keep the grid point under the cursor fixed on screen.
      const [dx, dy] = toDevice(xCss, yCss);
      const scale = cellPx();
      centerX = gx - (dx - surface.size[0] / 2) / scale;
      centerY = gy - (dy - surface.size[1] / 2) / scale;
      onZoom?.(zoom);
    },
    paint(xCss, yCss, alive) {
      const [gx, gy] = toGrid(xCss, yCss);
      const x = Math.floor(gx);
      const y = Math.floor(gy);
      // Walk a straight line from the previous cell so fast strokes leave no gaps.
      let [x0, y0] = lastCell ?? [x, y];
      const dx = Math.abs(x - x0);
      const dy = -Math.abs(y - y0);
      const sx = x0 < x ? 1 : -1;
      const sy = y0 < y ? 1 : -1;
      let err = dx + dy;
      for (;;) {
        writeCell(x0, y0, alive);
        if (x0 === x && y0 === y) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
          err += dy;
          x0 += sx;
        }
        if (e2 <= dx) {
          err += dx;
          y0 += sy;
        }
      }
      lastCell = [x, y];
    },
    endStroke() {
      lastCell = null;
    },
    dispose() {
      disposed = true;
      loop.stop();
    },
  };
}
