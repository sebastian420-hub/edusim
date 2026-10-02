import { compute, effect, frameLoop, pingPongStorage } from "vgpu";
import type { PingPongStorage, StorageBuffer } from "vgpu";
import { frameDelta } from "@/lib/gpu/runtime";
import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import computeShader from "./compute.wgsl";
import renderShader from "./render.wgsl";
import { DEFAULT_PATTERN, patterns, rasterizePattern } from "./patterns";
import { parseRule } from "./rules";
import { workgroupsFor } from "./workgroup";

export const GRID_SIZES = [256, 512, 1024, 2048] as const;
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

export interface CellularAutomataOptions {
  gridSize?: number;
  onGeneration?: (generation: number) => void;
  onZoom?: (zoom: number) => void;
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
  { gpu, canvas, surface, clock }: SimContext,
  { gridSize = 256, onGeneration, onZoom }: CellularAutomataOptions = {},
): CellularAutomataHandle {
  const computePass = compute(gpu, computeShader);
  const renderPass = effect(gpu, renderShader);

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

  const upload = (data: Uint32Array<ArrayBuffer>) => {
    cells.read.write(data);
    setGeneration(0);
    stepDebt = 0;
  };

  const applyInitial = () => {
    switch (initial.kind) {
      case "pattern": {
        const name = initial.name;
        const pattern = patterns.find((p) => p.name === name);
        upload(pattern ? rasterizePattern(pattern, size, size) : new Uint32Array(size * size));
        break;
      }
      case "random": {
        const data = new Uint32Array(size * size);
        for (let i = 0; i < data.length; i++) data[i] = Math.random() > 0.8 ? 1 : 0;
        upload(data);
        break;
      }
      case "clear":
        upload(new Uint32Array(size * size));
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
  };

  const writeCell = (x: number, y: number, alive: boolean) => {
    if (x < 0 || y < 0 || x >= size || y >= size) return;
    (cells.read as OffsetWritable).write(new Uint32Array([alive ? 1 : 0]), (y * size + x) * 4);
  };

  applyInitial();

  const loop = frameLoop(gpu, (frame) => {
    const dt = frameDelta(clock);
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
      ruleBirth = parseRule(params.birth);
      ruleSurvive = parseRule(params.survive);
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
    dispose: () => loop.stop(),
  };
}
