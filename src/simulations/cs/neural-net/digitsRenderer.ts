import { effect, storage } from "vgpu";
import type { FramePass, Gpu } from "vgpu";
import type { Engine } from "./engine";
import shader from "./digits.wgsl";

type Destroyable = { destroy?: () => void };

export const MAX_TILES = 512;

export interface DigitsLayout {
  strip: { x: number; y: number; w: number; h: number };
  templateSize: number;
  grid: { x: number; y: number; w: number; h: number };
  tile: number;
  cols: number;
  rows: number;
}

/** Where the templates and the tiles go (device pixels). */
export function digitsLayout(width: number, height: number, templates: number, dpr = 1): DigitsLayout {
  const m = Math.round(16 * dpr);
  const label = 18 * dpr;
  const templateSize = Math.max(8, Math.min((width - 2 * m) / (templates * 1.08), height * 0.16, 96 * dpr));
  const strip = { x: m, y: m + label, w: width - 2 * m, h: templateSize };
  const top = strip.y + templateSize + m + label;
  const gridW = width - 2 * m;
  const cols = Math.max(4, Math.min(24, Math.floor(gridW / (48 * dpr))));
  const tile = gridW / cols;
  const rows = Math.max(1, Math.floor((height - top - m) / tile));
  return { strip, templateSize, grid: { x: m, y: top, w: gridW, h: rows * tile }, tile, cols, rows };
}

/** Draws the Digits view from the engine's data and parameters and a list of tiles (digits.wgsl). */
export function createDigitsRenderer(gpu: Gpu, engine: Engine) {
  const tiles = storage(gpu, MAX_TILES * 16);
  const fx = effect(gpu, shader);
  let tileCount = 0;
  return {
    /** Tiles: (sample, guess, 1 if correct, true label) each. */
    setTiles(t: Float32Array) {
      tileCount = Math.min(MAX_TILES, t.length / 4);
      if (tileCount) tiles.write(t.subarray(0, tileCount * 4) as Float32Array<ArrayBuffer>);
    },
    prepare(target: { readonly size: readonly [number, number] }, layout: DigitsLayout, opts: { templates: number; firstWeight: number; scale: number; hasParams: boolean }) {
      const resolution = [target.size[0], target.size[1]];
      const r = (q: { x: number; y: number; w: number; h: number }) => [q.x, q.y, q.w, q.h];
      fx.set({
        data: engine.buffers.data,
        tiles,
        params: engine.buffers.params,
        dg: {
          resolution,
          tile: layout.tile,
          templateSize: layout.templateSize,
          grid: r(layout.grid),
          strip: r(layout.strip),
          cols: layout.cols,
          tiles: Math.min(tileCount, layout.cols * layout.rows),
          templates: opts.templates,
          rowStride: engine.spec.inputs,
          firstWeight: opts.firstWeight,
          wordsPerSample: engine.wordsPerSample,
          hasParams: opts.hasParams ? 1 : 0,
          pad0: 0,
          scale: Math.max(1e-6, opts.scale),
          pad1: 0,
          pad2: 0,
          pad3: 0,
        },
      });
      return (pass: FramePass) => pass.draw(fx);
    },
    dispose() {
      (tiles as unknown as Destroyable).destroy?.();
    },
  };
}

export type DigitsRenderer = ReturnType<typeof createDigitsRenderer>;
