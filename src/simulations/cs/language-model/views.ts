/**
 * Where everything goes on the canvas in each view of "How AI Writes": boxes, text panels and heat maps in device
 * pixels, and the captions (drawn as HTML over the canvas) in CSS pixels. Pure functions of the canvas size and
 * what there is to show.
 */
import { COLOUR, layoutText, TINT } from "./screen";
import type { Panel, TextItem } from "./screen";

export interface Label {
  text: string;
  /** CSS pixels from the canvas's top-left; the label's left edge (or centre, when centred). */
  x: number;
  y: number;
  centred?: boolean;
}

export interface Glyphs {
  newline: number;
  space: number;
  mark: number;
  hidden: number;
  /** Line breaks start a new row (verse, prose) or show as a dot (lists of names, the copying game). */
  newlineMode: "break" | "mark";
}

export interface Canvas {
  w: number;
  h: number;
  dpr: number;
  /** Text cell size in device pixels. */
  cw: number;
  ch: number;
}

export interface Scene {
  panels: Panel[];
  labels: Label[];
}

/** The font size (CSS pixels) each view uses on a canvas this wide. */
export function fontFor(view: "train" | "predict" | "tables" | "shannon", cssWidth: number): number {
  const base = view === "shannon" ? cssWidth / 52 : view === "predict" ? cssWidth / 60 : cssWidth / 80;
  return Math.round(Math.min(view === "shannon" ? 22 : 15, Math.max(11, base)));
}

const CAPTION = 22;
const PAD = 14;

function textBox(c: Canvas, x: number, y: number, w: number, h: number, items: readonly TextItem[], g: Glyphs, tail: boolean, highlight = false): { panels: Panel[]; cols: number; rows: number; at: Int32Array } {
  const inner = Math.round(8 * c.dpr);
  const cols = Math.max(1, Math.floor((w - 2 * inner) / c.cw));
  const rows = Math.max(1, Math.floor((h - 2 * inner) / c.ch));
  const { cells, at } = layoutText(items, cols, rows, { newline: g.newline, newlineMode: g.newlineMode, space: g.space, mark: g.mark, tail });
  return { panels: [{ kind: "box", x, y, w, h, highlight }, { kind: "text", x: x + inner, y: y + inner, cols, rows, cells }], cols, rows, at };
}

/** Train: the model's writing (up to four passages) above a held-out passage tinted by its surprise. */
export function trainScene(c: Canvas, g: Glyphs, lines: readonly TextItem[][], passage: readonly TextItem[], captions: { writing: string; passage: string }): Scene {
  const r = c.dpr;
  const pad = PAD * r;
  const cap = CAPTION * r;
  const wide = c.w / r >= 700;
  const shown = wide ? lines.slice(0, 4) : lines.slice(0, 2);
  const colsN = wide ? 2 : 1;
  const rowsN = Math.max(1, Math.ceil(shown.length / colsN));
  const writeTop = pad + cap;
  const writeH = Math.round((c.h - 2 * pad - 2 * cap) * 0.58);
  const gap = Math.round(10 * r);
  const boxW = Math.floor((c.w - 2 * pad - (colsN - 1) * gap) / colsN);
  const boxH = Math.floor((writeH - (rowsN - 1) * gap) / rowsN);
  const panels: Panel[] = [];
  shown.forEach((items, k) => {
    const x = pad + (k % colsN) * (boxW + gap);
    const y = writeTop + Math.floor(k / colsN) * (boxH + gap);
    panels.push(...textBox(c, x, y, boxW, boxH, items, g, true).panels);
  });
  const passTop = writeTop + writeH + cap + gap;
  panels.push(...textBox(c, pad, passTop, c.w - 2 * pad, c.h - pad - passTop, passage, g, false).panels);
  return {
    panels,
    labels: [
      { text: captions.writing, x: pad / r, y: (writeTop - cap + 3 * r) / r },
      { text: captions.passage, x: pad / r, y: (passTop - cap + 3 * r) / r },
    ],
  };
}

export interface AttentionMaps {
  /** Per layer, [H][T][T] probabilities. */
  maps: readonly ArrayLike<number>[];
  H: number;
  T: number;
}

/** Predict: the prompt the model reads (tinted by one head's attention from the last letter) and every head's map. */
export function predictScene(c: Canvas, g: Glyphs, prompt: readonly TextItem[], att: AttentionMaps | null, focus: { layer: number; head: number }, captions: { prompt: string; maps: string }): Scene {
  const r = c.dpr;
  const pad = PAD * r;
  const cap = CAPTION * r;
  const top = pad + cap;
  const promptH = Math.min(Math.round(c.h * 0.36), Math.round(6 * c.ch + 16 * r));
  const panels: Panel[] = [...textBox(c, pad, top, c.w - 2 * pad, promptH, prompt, g, true).panels];
  const labels: Label[] = [{ text: captions.prompt, x: pad / r, y: (top - cap + 3 * r) / r }];
  if (att && att.maps.length) {
    const L = att.maps.length;
    const { H, T } = att;
    const mapsTop = top + promptH + cap + Math.round(12 * r);
    labels.push({ text: captions.maps, x: pad / r, y: (mapsTop - cap + 3 * r) / r });
    const sub = Math.round(16 * r);
    const gap = Math.round(12 * r);
    const availW = c.w - 2 * pad - (H - 1) * gap;
    const availH = c.h - mapsTop - pad - L * sub - (L - 1) * gap;
    const cell = Math.max(1, Math.floor(Math.min(availW / H, availH / L) / T));
    const size = cell * T;
    const gridW = H * size + (H - 1) * gap;
    const left = Math.round((c.w - gridW) / 2);
    for (let l = 0; l < L; l++) {
      for (let h = 0; h < H; h++) {
        const x = left + h * (size + gap);
        const y = mapsTop + l * (size + sub + gap) + sub;
        const values = new Float32Array(T * T);
        const m = att.maps[l];
        for (let i = 0; i < T * T; i++) values[i] = Math.sqrt(m[h * T * T + i]);
        if (l === focus.layer && h === focus.head) panels.push({ kind: "box", x: x - 3 * r, y: y - 3 * r, w: size + 6 * r, h: size + 6 * r, highlight: true });
        panels.push({ kind: "heat", x, y, cols: T, rows: T, cell, gap: cell >= 6 ? Math.max(1, Math.round(r * 0.5)) : 0, values });
        labels.push({ text: `layer ${l + 1} · head ${h + 1}`, x: (x + size / 2) / r, y: (y - sub + 1 * r) / r, centred: true });
      }
    }
  }
  return { panels, labels };
}

/** Count tables: a sample of each table's writing, k = 0 … 6 letters of context, the chosen one highlighted. */
export function tablesScene(c: Canvas, g: Glyphs, samples: readonly (readonly TextItem[] | null)[], selected: number, captions: string[]): Scene {
  const r = c.dpr;
  const pad = PAD * r;
  const cap = Math.round(18 * r);
  const gap = Math.round(6 * r);
  const n = samples.length;
  const boxH = Math.floor((c.h - 2 * pad - n * cap - (n - 1) * gap) / n);
  const panels: Panel[] = [];
  const labels: Label[] = [];
  samples.forEach((items, k) => {
    const y = pad + k * (boxH + cap + gap) + cap;
    labels.push({ text: captions[k], x: pad / r, y: (y - cap + 2 * r) / r });
    panels.push(...textBox(c, pad, y, c.w - 2 * pad, boxH, items ?? [], { ...g, newlineMode: "mark" }, false, k === selected).panels);
  });
  return { panels, labels };
}

/** Shannon's game: the text so far, the letters you have guessed, a cursor on the hidden one. */
export function shannonScene(c: Canvas, g: Glyphs, items: readonly TextItem[], caption: string): Scene {
  const r = c.dpr;
  const pad = PAD * r;
  const cap = CAPTION * r;
  const top = pad + cap;
  const box = textBox(c, pad, top, c.w - 2 * pad, Math.min(c.h - top - pad, Math.round(10 * c.ch + 16 * r)), items, g, true);
  return { panels: box.panels, labels: [{ text: caption, x: pad / r, y: (top - cap + 3 * r) / r }] };
}

/** Text items for letters, optionally tinted (surprise in nats, scaled so 6 nats is full strength). */
export function surpriseItems(tokens: ArrayLike<number>, surprise: ArrayLike<number> | null, colour: number = COLOUR.text): TextItem[] {
  return Array.from(tokens, (t, i) => {
    const s = surprise ? surprise[i] : NaN;
    return Number.isFinite(s) ? { glyph: t, colour, tint: TINT.surprise, strength: Math.min(1, s / 6) } : { glyph: t, colour };
  });
}
