/**
 * Handwritten digits for "How AI Learns": the packed MNIST subset (`public/data/mnist-12k.dat`, built by
 * `scripts/build-digits.mjs`), the draw pad's conversion of a drawing into an MNIST-style image, and the confusion
 * matrix. MNIST: Yann LeCun, Corinna Cortes and Christopher J. C. Burges, CC BY-SA 3.0.
 */
import type { Dataset } from "./net";

export const DIGIT_SIZE = 28;
export const DIGIT_PIXELS = DIGIT_SIZE * DIGIT_SIZE;
export const DIGITS_URL = "data/mnist-12k.dat";

/** The digits as stored: one byte per pixel (0 = background, 255 = ink), train samples first. */
export interface Digits {
  count: number;
  nTrain: number;
  labels: Uint8Array;
  pixels: Uint8Array;
}

/** Parses the decompressed file (see build-digits.mjs for the format). */
export function parseDigits(bytes: Uint8Array): Digits {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== "EDUD") throw new Error("not a digits file");
  const count = view.getUint32(4, true);
  const nTrain = view.getUint32(8, true);
  const rows = view.getUint32(12, true);
  const cols = view.getUint32(16, true);
  if (rows !== DIGIT_SIZE || cols !== DIGIT_SIZE) throw new Error(`digits are ${rows}×${cols}, expected 28×28`);
  const labels = bytes.subarray(20, 20 + count);
  const pixels = bytes.subarray(20 + count, 20 + count + count * DIGIT_PIXELS);
  if (pixels.length !== count * DIGIT_PIXELS) throw new Error("digits file is truncated");
  return { count, nTrain, labels, pixels };
}

/** Fetches and unpacks the digits in the browser (gzip, decompressed unless the server already did). */
export async function loadDigits(base = "/"): Promise<Digits> {
  const res = await fetch(`${base}${DIGITS_URL}`);
  if (!res.ok) throw new Error(`digits: HTTP ${res.status}`);
  const raw = new Uint8Array(await res.arrayBuffer());
  if (raw[0] !== 0x1f || raw[1] !== 0x8b) return parseDigits(raw);
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
  return parseDigits(new Uint8Array(await new Response(stream).arrayBuffer()));
}

/** A subset as a twin dataset (pixels scaled to 0…1, as unpack4x8unorm gives the GPU). */
export function digitsDataset(d: Digits, nTrain = d.nTrain, nTest = d.count - d.nTrain): Dataset {
  const n = nTrain + nTest;
  const x = new Float64Array(n * DIGIT_PIXELS);
  const y = new Uint8Array(n);
  const take = (src: number, dst: number) => {
    for (let i = 0; i < DIGIT_PIXELS; i++) x[dst * DIGIT_PIXELS + i] = d.pixels[src * DIGIT_PIXELS + i] / 255;
    y[dst] = d.labels[src];
  };
  for (let k = 0; k < nTrain; k++) take(k, k);
  for (let k = 0; k < nTest; k++) take(d.nTrain + k, nTrain + k);
  return { n, dim: DIGIT_PIXELS, nTrain, x, y, classes: 10 };
}

/**
 * A drawing (any size, ink 0…1, row-major) as MNIST normalised its digits: crop to the ink, scale so the longer
 * side is 20 px (keeping the aspect ratio, area-averaged), and place it in 28 × 28 with its centre of mass at the
 * centre. Returns ink 0…1, or null for an empty drawing.
 */
export function toDigit(ink: ArrayLike<number>, width: number, height: number): Float64Array | null {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      if (ink[y * width + x] > 0.05) {
        x0 = Math.min(x0, x);
        y0 = Math.min(y0, y);
        x1 = Math.max(x1, x);
        y1 = Math.max(y1, y);
      }
  if (x1 < 0) return null;
  const w = x1 - x0 + 1;
  const h = y1 - y0 + 1;
  const scale = 20 / Math.max(w, h);
  const sw = Math.max(1, Math.round(w * scale));
  const sh = Math.max(1, Math.round(h * scale));
  // Area-average the crop into sw × sh.
  const small = new Float64Array(sw * sh);
  for (let sy = 0; sy < sh; sy++)
    for (let sx = 0; sx < sw; sx++) {
      const ax = x0 + (sx * w) / sw;
      const bx = x0 + ((sx + 1) * w) / sw;
      const ay = y0 + (sy * h) / sh;
      const by = y0 + ((sy + 1) * h) / sh;
      let sum = 0;
      let area = 0;
      for (let y = Math.floor(ay); y < Math.ceil(by); y++)
        for (let x = Math.floor(ax); x < Math.ceil(bx); x++) {
          const cover = (Math.min(bx, x + 1) - Math.max(ax, x)) * (Math.min(by, y + 1) - Math.max(ay, y));
          if (cover <= 0) continue;
          sum += cover * ink[y * width + x];
          area += cover;
        }
      small[sy * sw + sx] = area ? Math.min(1, sum / area) : 0;
    }
  // Centre of mass → the centre of the 28 × 28 image.
  let m = 0;
  let cx = 0;
  let cy = 0;
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < sw; x++) {
      const v = small[y * sw + x];
      m += v;
      cx += v * (x + 0.5);
      cy += v * (y + 0.5);
    }
  const ox = Math.round(DIGIT_SIZE / 2 - cx / m);
  const oy = Math.round(DIGIT_SIZE / 2 - cy / m);
  const out = new Float64Array(DIGIT_PIXELS);
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < sw; x++) {
      const tx = x + ox;
      const ty = y + oy;
      if (tx >= 0 && tx < DIGIT_SIZE && ty >= 0 && ty < DIGIT_SIZE) out[ty * DIGIT_SIZE + tx] = small[y * sw + x];
    }
  return out;
}

/** counts[truth][guess] over the given labels and guesses. */
export function confusion(truth: ArrayLike<number>, guess: ArrayLike<number>, classes = 10): number[][] {
  const m = Array.from({ length: classes }, () => new Array<number>(classes).fill(0));
  for (let k = 0; k < truth.length; k++) m[truth[k]][guess[k]]++;
  return m;
}
