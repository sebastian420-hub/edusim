/**
 * Drawing for "How AI Writes": text panels (a glyph atlas rendered by the browser at exactly the cell size, so text
 * is crisp at any device pixel ratio), heat maps and boxes, composed on the CPU and drawn by screen.wgsl in one
 * fullscreen pass. Also the text layout (word wrapping, line breaks, following the end of live text).
 */
import { effect, storage } from "vgpu";
import type { FramePass, Gpu, StorageBuffer } from "vgpu";
import screenShader from "./screen.wgsl";

export const NO_GLYPH = 255;
export const COLOUR = { text: 0, dim: 1, bright: 2, sky: 3, amber: 4, rose: 5, emerald: 6, label: 7 } as const;
export const TINT = { none: 0, surprise: 1, attention: 2, cursor: 3, guesses: 4, right: 5 } as const;
export const MAX_PANELS = 24;

/** One cell word: glyph, text colour, tint kind and its strength (0 … 1). */
export function cellWord(glyph: number, colour = 0, tint = 0, strength = 0): number {
  const s = Math.round(Math.min(1, Math.max(0, strength)) * 65535);
  return ((glyph & 255) | ((colour & 15) << 8) | ((tint & 15) << 12) | (s << 16)) >>> 0;
}

export type Panel =
  | { kind: "text"; x: number; y: number; cols: number; rows: number; cells: Uint32Array }
  | { kind: "heat"; x: number; y: number; cols: number; rows: number; cell: number; gap: number; values: ArrayLike<number> }
  | { kind: "box"; x: number; y: number; w: number; h: number; highlight?: boolean };

/** The glyph shown for each symbol: line breaks as ↵ (drawn dim), everything else as itself. */
export const displaySymbol = (s: string) => (s === "\n" ? "↵" : s === "\t" ? "→" : s);

/** Extra glyphs after the vocabulary's: a separator dot and the placeholder of a hidden letter. */
export const EXTRA_GLYPHS = ["·", "_"] as const;

export interface Atlas {
  glyphs: number;
  w: number;
  h: number;
  bytes: Uint8Array;
}

/** Text cell size in device pixels for a font size in CSS pixels. */
export function cellSize(fontCss: number, dpr: number): { w: number; h: number } {
  return { w: Math.max(4, Math.round(fontCss * 0.6 * dpr)), h: Math.max(8, Math.round(fontCss * 1.45 * dpr)) };
}

/** Renders the glyphs with the browser's text engine (a monospace font), one cell each. */
export function buildAtlas(symbols: readonly string[], w: number, h: number, family: string): Atlas {
  const n = symbols.length;
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w * n, h) : Object.assign(document.createElement("canvas"), { width: w * n, height: h });
  const ctx = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  const bytes = new Uint8Array(Math.ceil((n * w * h) / 4) * 4);
  if (!ctx) return { glyphs: 0, w, h, bytes };
  ctx.clearRect(0, 0, w * n, h);
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  const px = Math.max(6, Math.round(h / 1.45));
  ctx.font = `${px}px ${family}`;
  symbols.forEach((s, k) => ctx.fillText(s, k * w + w / 2, Math.round(h * 0.72)));
  const img = ctx.getImageData(0, 0, w * n, h).data;
  for (let k = 0; k < n; k++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) bytes[(k * h + y) * w + x] = img[4 * (y * w * n + k * w + x) + 3];
    }
  }
  return { glyphs: n, w, h, bytes };
}

export interface TextItem {
  glyph: number;
  colour?: number;
  tint?: number;
  strength?: number;
}

export interface LayoutOptions {
  /** Glyph of a line break, and whether it starts a new row ("break") or shows inline as a separator ("mark"). */
  newline: number;
  newlineMode: "break" | "mark";
  /** Glyph of a space (rows never start with a wrapped space). */
  space: number;
  /** The separator glyph drawn for line breaks in "mark" mode. */
  mark: number;
  /** Keep the last rows when the text is longer than the panel (live writing), else the first. */
  tail?: boolean;
}

/**
 * Lays out text into a cols × rows grid of cell words: words wrap whole when they fit on a row, line breaks end the
 * row (drawn as a dim ↵ at its end) or show as a dim dot. Returns the cells and the (row, column) of each item
 * (−1 when it fell outside the panel).
 */
export function layoutText(items: readonly TextItem[], cols: number, rows: number, o: LayoutOptions): { cells: Uint32Array; at: Int32Array; rowsUsed: number } {
  const lines: number[][] = [[]];
  const placeRow: number[] = [];
  const placeCol: number[] = [];
  const isBreak = (g: number) => g === o.space || g === o.newline;
  for (let i = 0; i < items.length; i++) {
    const g = items[i].glyph;
    let line = lines[lines.length - 1];
    if (g === o.newline && o.newlineMode === "break") {
      if (line.length >= cols) {
        lines.push([]);
        line = lines[lines.length - 1];
      }
      placeRow.push(lines.length - 1);
      placeCol.push(line.length);
      line.push(i);
      lines.push([]);
      continue;
    }
    if (!isBreak(g) && line.length > 0 && (i === 0 || isBreak(items[i - 1].glyph))) {
      // Start of a word: wrap it whole if it fits on a fresh row.
      let end = i;
      while (end < items.length && !isBreak(items[end].glyph)) end++;
      const len = end - i;
      if (line.length + len > cols && len <= cols) {
        lines.push([]);
        line = lines[lines.length - 1];
      }
    }
    if (line.length >= cols) {
      lines.push([]);
      line = lines[lines.length - 1];
      if (g === o.space) {
        placeRow.push(-1);
        placeCol.push(-1);
        continue;
      }
    }
    placeRow.push(lines.length - 1);
    placeCol.push(line.length);
    line.push(i);
  }
  if (lines.length > 1 && lines[lines.length - 1].length === 0) lines.pop();
  const first = o.tail ? Math.max(0, lines.length - rows) : 0;
  const cells = new Uint32Array(cols * rows).fill(cellWord(NO_GLYPH));
  const at = new Int32Array(items.length * 2).fill(-1);
  for (let i = 0; i < items.length; i++) {
    const r = placeRow[i] - first;
    if (placeRow[i] < 0 || r < 0 || r >= rows) continue;
    const it = items[i];
    let glyph = it.glyph;
    let colour = it.colour ?? COLOUR.text;
    if (glyph === o.newline) {
      colour = COLOUR.dim;
      if (o.newlineMode === "mark") glyph = o.mark;
    }
    cells[r * cols + placeCol[i]] = cellWord(glyph, colour, it.tint ?? 0, it.strength ?? 0);
    at[2 * i] = r;
    at[2 * i + 1] = placeCol[i];
  }
  return { cells, at, rowsUsed: Math.min(rows, lines.length - first) };
}

/** The pass that draws a list of panels. */
export function createScreen(gpu: Gpu) {
  const fx = effect(gpu, screenShader);
  let cellsLen = 4096;
  let heatLen = 4096;
  let atlasLen = 4096;
  let cells: StorageBuffer = storage(gpu, cellsLen * 4, "read");
  let heat: StorageBuffer = storage(gpu, heatLen * 4, "read");
  let atlasBuf: StorageBuffer = storage(gpu, atlasLen, "read");
  let atlas: Atlas | null = null;
  const destroy = (b: StorageBuffer) => (b as unknown as { destroy?: () => void }).destroy?.();
  let lastKey = "";

  const emptyPanel = { rect: [0, 0, 0, 0], grid: [0, 0, 0, 0], cell: [0, 0, 0, 0] };

  return {
    setAtlas(a: Atlas) {
      atlas = a;
      if (a.bytes.byteLength > atlasLen) {
        destroy(atlasBuf);
        atlasLen = Math.ceil(a.bytes.byteLength / 4096) * 4096;
        atlasBuf = storage(gpu, atlasLen, "read");
      }
      atlasBuf.write(a.bytes as Uint8Array<ArrayBuffer>);
    },
    get atlas() {
      return atlas;
    },

    /** Uploads the panels (only when their content changed, by `key`) and returns the body of a frame pass. */
    prepare(target: { readonly size: readonly [number, number] }, panels: readonly Panel[], key: string) {
      const list = panels.slice(0, MAX_PANELS);
      const fullKey = `${key}|${target.size[0]}x${target.size[1]}`;
      if (fullKey !== lastKey) {
        lastKey = fullKey;
        const nCells = list.reduce((a, p) => a + (p.kind === "text" ? p.cells.length : 0), 0);
        const nHeat = list.reduce((a, p) => a + (p.kind === "heat" ? p.values.length : 0), 0);
        if (nCells > cellsLen) {
          destroy(cells);
          cellsLen = Math.ceil(nCells / 4096) * 4096;
          cells = storage(gpu, cellsLen * 4, "read");
        }
        if (nHeat > heatLen) {
          destroy(heat);
          heatLen = Math.ceil(nHeat / 4096) * 4096;
          heat = storage(gpu, heatLen * 4, "read");
        }
        const cellData = new Uint32Array(Math.max(1, nCells));
        const heatData = new Float32Array(Math.max(1, nHeat));
        let c = 0;
        let hOff = 0;
        const g = atlas ?? { w: 8, h: 16 };
        const uniformPanels = list.map((p) => {
          if (p.kind === "text") {
            cellData.set(p.cells, c);
            const entry = { rect: [p.x, p.y, p.cols * g.w, p.rows * g.h], grid: [p.cols, p.rows, 0, c], cell: [g.w, g.h, 0, 0] };
            c += p.cells.length;
            return entry;
          }
          if (p.kind === "heat") {
            heatData.set(p.values, hOff);
            const entry = { rect: [p.x, p.y, p.cols * p.cell, p.rows * p.cell], grid: [p.cols, p.rows, 1, hOff], cell: [p.cell, p.cell, p.gap, 0] };
            hOff += p.values.length;
            return entry;
          }
          return { rect: [p.x, p.y, p.w, p.h], grid: [p.highlight ? 1 : 0, 0, 2, 0], cell: [1, 1, 0, 0] };
        });
        if (nCells) cells.write(cellData as Uint32Array<ArrayBuffer>);
        if (nHeat) heat.write(heatData as Float32Array<ArrayBuffer>);
        fx.set({
          cells,
          heat,
          atlas: atlasBuf,
          screen: {
            resolution: [target.size[0], target.size[1]],
            count: uniformPanels.length,
            glyphs: atlas?.glyphs ?? 0,
            glyph: [g.w, g.h],
            pad0: 0,
            pad1: 0,
            panels: Array.from({ length: MAX_PANELS }, (_, k) => uniformPanels[k] ?? emptyPanel),
          },
        });
      }
      return (pass: FramePass) => pass.draw(fx);
    },

    /** Forces the next prepare() to upload. */
    invalidate() {
      lastKey = "";
    },

    dispose() {
      destroy(cells);
      destroy(heat);
      destroy(atlasBuf);
    },
  };
}

export type Screen = ReturnType<typeof createScreen>;
