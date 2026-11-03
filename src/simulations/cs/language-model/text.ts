/**
 * The texts of "How AI Writes": loading, the character vocabulary, the training / held-out split, the seeded windows
 * every model trains on, and the two built-in texts (Sonnet 18 and the copying game).
 *
 * Tiny Shakespeare (Andrej Karpathy's char-rnn collection, public domain) and the makemore names (the 32 000 most
 * common US baby names, from the Social Security Administration's data, CC0) ship gzipped in public/data
 * (scripts/build-texts.mjs).
 */

export const TEXT_IDS = ["shakespeare", "names", "sonnet", "copying", "own"] as const;
export type TextId = (typeof TEXT_IDS)[number];

/** Tokens before a window that a model may read as context (the MLP looks back up to this many letters). */
export const CONTEXT_MARGIN = 16;
/** Most distinct symbols a text may have (your own text: rarer symbols are dropped). */
export const MAX_VOCAB = 96;
/** Your own text: at least this many characters, at most this many kept. */
export const OWN_MIN = 400;
export const OWN_MAX = 200_000;
/** Below this many characters a text has no held-out part: it is all for training (and measured on itself). */
export const HELD_OUT_MIN = 4000;

export const TEXT_URLS: Partial<Record<TextId, string>> = {
  shakespeare: "data/shakespeare.dat",
  names: "data/names.dat",
};

export interface Corpus {
  id: TextId;
  /** The symbol of each token id, sorted (so "\n" and " " come first). */
  vocab: string[];
  tokens: Uint8Array;
  /** Tokens [0, nTrain) are for training; [nTrain, length) are held out (never trained on). */
  nTrain: number;
  /** False when the text is too short to hold part of it back: "held-out" measurements then use the training text. */
  heldOut: boolean;
  /** Copying game: 1 where the token is predictable by copying (the second copy, after its first letter). */
  repeated?: Uint8Array;
  /** Copying game: how far back the same letter stands in the first copy (the run's length), where repeated. */
  copyFrom?: Uint8Array;
}

/** mulberry32: a small, fast, seeded generator of uniform numbers in [0, 1). */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A corpus from a string: the vocabulary is its sorted set of characters. */
export function buildCorpus(id: TextId, text: string, split = 0.9, extra: Partial<Corpus> = {}): Corpus {
  const chars = Array.from(text);
  const vocab = Array.from(new Set(chars)).sort();
  if (vocab.length > 255) throw new Error("too many distinct symbols");
  const index = new Map(vocab.map((c, k) => [c, k]));
  const tokens = Uint8Array.from(chars, (c) => index.get(c)!);
  const heldOut = tokens.length >= HELD_OUT_MIN;
  return { id, vocab, tokens, nTrain: heldOut ? Math.floor(tokens.length * split) : tokens.length, heldOut, ...extra };
}

export const decode = (c: Corpus, tokens: ArrayLike<number>) => Array.from(tokens, (t) => c.vocab[t] ?? "").join("");
/** Encodes a string with the corpus's vocabulary, dropping symbols it does not have. */
export function encode(c: Corpus, text: string): number[] {
  const index = new Map(c.vocab.map((s, k) => [s, k]));
  return Array.from(text).flatMap((s) => (index.has(s) ? [index.get(s)!] : []));
}

/** The names, one per line, shuffled (the file is sorted by popularity) so the held-out names are typical ones. */
export function namesCorpus(file: string, seed = 2026): Corpus {
  const names = file.split("\n").map((s) => s.trim()).filter(Boolean);
  const r = rng(seed);
  for (let i = names.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [names[i], names[j]] = [names[j], names[i]];
  }
  const text = names.join("\n") + "\n";
  // Split at a name boundary near 90 %.
  const cut = text.indexOf("\n", Math.floor(text.length * 0.9)) + 1;
  return buildCorpus("names", text, cut / text.length);
}

export const shakespeareCorpus = (file: string) => buildCorpus("shakespeare", file);

/** Shakespeare's Sonnet 18 (1609, public domain): short enough for a model to learn by heart. */
export const SONNET_18 = `Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.
`;

export const sonnetCorpus = () => buildCorpus("sonnet", SONNET_18);

/** The copying game's segments: this many random letters, then the same letters again, then a line break. */
export const COPY_MIN = 8;
export const COPY_MAX = 24;
export const COPY_LENGTH = 120_000;

/**
 * The copying game: random letters a–z, each run followed by an exact copy of itself and a line break
 * ("qmzfkb" + "qmzfkb" + "\n"). The first copy is unpredictable (ln 26 per letter at best); the second is
 * predictable from the first, but only for a model that can find where the current letter appeared before and copy
 * what followed it. The runs have random lengths, so no fixed look-back distance works.
 */
export function copyingCorpus(seed = 7, length = COPY_LENGTH): Corpus {
  const r = rng(seed);
  let text = "";
  const marks: number[] = [];
  const from: number[] = [];
  while (text.length < length) {
    const n = COPY_MIN + Math.floor(r() * (COPY_MAX - COPY_MIN + 1));
    let run = "";
    for (let k = 0; k < n; k++) run += String.fromCharCode(97 + Math.floor(r() * 26));
    for (let k = 0; k < n; k++) {
      marks.push(0);
      from.push(0);
    }
    for (let k = 0; k < n; k++) {
      marks.push(k === 0 ? 0 : 1);
      from.push(k === 0 ? 0 : n);
    }
    marks.push(0);
    from.push(0);
    text += run + run + "\n";
  }
  const base = buildCorpus("copying", text);
  // Keep the split at a line boundary, so no run straddles it.
  const nTrain = text.lastIndexOf("\n", base.nTrain) + 1;
  return { ...base, nTrain, repeated: Uint8Array.from(marks), copyFrom: Uint8Array.from(from) };
}

/** Your own text: line endings normalised, trimmed to OWN_MAX, rare symbols dropped to fit MAX_VOCAB. */
export function ownCorpus(raw: string): Corpus | null {
  let text = raw.replace(/\r\n?/g, "\n").slice(0, OWN_MAX);
  const counts = new Map<string, number>();
  for (const c of text) counts.set(c, (counts.get(c) ?? 0) + 1);
  if (counts.size > MAX_VOCAB) {
    const keep = new Set([...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, MAX_VOCAB).map(([c]) => c));
    text = Array.from(text).filter((c) => keep.has(c)).join("");
  }
  if (Array.from(text).length < OWN_MIN) return null;
  return buildCorpus("own", text);
}

/** Range of window starts [lo, hi] for windows of `T` letters whose targets (the letter after each) lie in a part. */
export function windowRange(c: Corpus, T: number, part: "train" | "heldOut"): [number, number] {
  const n = c.tokens.length;
  if (part === "train" || !c.heldOut) return [CONTEXT_MARGIN, Math.max(CONTEXT_MARGIN, c.nTrain - T - 1)];
  return [c.nTrain, Math.max(c.nTrain, n - T - 1)];
}

/** `count` window starts drawn uniformly from a part, from a seeded generator. */
export function windowStarts(c: Corpus, T: number, part: "train" | "heldOut", count: number, seed: number): Uint32Array {
  const [lo, hi] = windowRange(c, T, part);
  const r = rng(seed);
  return Uint32Array.from({ length: count }, () => lo + Math.floor(r() * (hi - lo + 1)));
}

/** The training batches: B random windows per step, the same sequence for the GPU and the twin. */
export function batchStream(c: Corpus, T: number, B: number, seed: number) {
  const [lo, hi] = windowRange(c, T, "train");
  const r = rng(seed ^ 0x5bd1e995);
  let steps = 0;
  return {
    next(): Uint32Array {
      steps++;
      return Uint32Array.from({ length: B }, () => lo + Math.floor(r() * (hi - lo + 1)));
    },
    get steps() {
      return steps;
    },
  };
}

/**
 * Where a held-out passage of `length` letters starts: the first line start after a fixed point of the held-out
 * text (or the training text, when there is none), so the passage begins at the start of a line.
 */
export function passageStart(c: Corpus, length: number): number {
  const nl = c.vocab.indexOf("\n");
  const [lo, hi] = c.heldOut ? [c.nTrain, c.tokens.length - length - 1] : [CONTEXT_MARGIN, c.tokens.length - length - 1];
  let at = Math.max(lo, Math.min(hi, lo + Math.floor((hi - lo) * 0.25)));
  if (nl >= 0) {
    const limit = Math.min(hi, at + 400);
    for (let k = at; k < limit; k++) {
      if (c.tokens[k - 1] === nl) {
        at = k;
        break;
      }
    }
  }
  return Math.max(lo, Math.min(at, hi));
}

/** A starting context for writing: `T` letters of text ending at a line break (held-out text when there is some). */
export function seedContext(c: Corpus, T: number, seed: number): Uint8Array {
  const nl = c.vocab.indexOf("\n");
  const n = c.tokens.length;
  const [lo, hi] = c.heldOut ? [c.nTrain, n - 1] : [0, n - 1];
  const r = rng(seed);
  // Candidate ends (exclusive) run from lo + T to hi; start at a random one and move on to the next line break.
  const span = Math.max(1, hi - lo - T + 1);
  const first = Math.floor(r() * span);
  let end = Math.min(hi, lo + T + first);
  if (nl >= 0) {
    for (let k = 0; k < span; k++) {
      const at = lo + T + ((first + k) % span);
      if (c.tokens[(at - 1 + n) % n] === nl) {
        end = at;
        break;
      }
    }
  }
  const out = new Uint8Array(T);
  for (let k = 0; k < T; k++) {
    const i = end - T + k;
    // A text shorter than the context wraps around (the sonnet with a long context).
    out[k] = c.tokens[((i % n) + n) % n];
  }
  return out;
}

/** Fetches a shipped text (gzip, decompressed unless the server already did). */
export async function loadText(id: "shakespeare" | "names", base = "/"): Promise<string> {
  const res = await fetch(`${base}${TEXT_URLS[id]}`);
  if (!res.ok) throw new Error(`${id}: HTTP ${res.status}`);
  const raw = new Uint8Array(await res.arrayBuffer());
  const bytes = raw[0] === 0x1f && raw[1] === 0x8b ? new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()) : raw;
  return new TextDecoder().decode(bytes);
}
