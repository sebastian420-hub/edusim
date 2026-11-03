/**
 * Measurements of written text: how repetitive it is, how much of it is copied verbatim from the training text, how
 * many of its names are new, and the entropy of a next-letter distribution.
 */

/** The probabilities softmax(scores / T) (T = 0: all on the most likely letter). */
export function distribution(scores: ArrayLike<number>, temperature: number): number[] {
  const s = Array.from(scores);
  const mx = Math.max(...s);
  if (temperature <= 0) {
    const best = s.indexOf(mx);
    return s.map((_, v) => (v === best ? 1 : 0));
  }
  const w = s.map((x) => Math.exp((x - mx) / temperature));
  const total = w.reduce((a, b) => a + b, 0);
  return w.map((x) => x / total);
}

/** Entropy in nats of a distribution. */
export function entropy(p: readonly number[]): number {
  const s = p.reduce((a, x) => a + (x > 0 ? x * Math.log(x) : 0), 0);
  return s === 0 ? 0 : -s;
}

/** Share of the text's n-letter sequences that already appeared earlier in it: 0 for fresh text, near 1 for loops. */
export function repetition(text: string, n = 8): number {
  const total = text.length - n + 1;
  if (total <= 1) return 0;
  const seen = new Set<string>();
  let repeats = 0;
  for (let i = 0; i < total; i++) {
    const g = text.slice(i, i + n);
    if (seen.has(g)) repeats++;
    else seen.add(g);
  }
  return repeats / total;
}

/**
 * The longest run of `written` that appears verbatim in `source` (length, and where it starts in `written`).
 * Dynamic programming over both texts: meant for short sources (a sonnet, a pasted text).
 */
export function longestCopy(written: string, source: string): { length: number; at: number } {
  const m = written.length;
  let prev = new Uint16Array(m + 1);
  let cur = new Uint16Array(m + 1);
  let best = { length: 0, at: 0 };
  for (let i = 1; i <= source.length; i++) {
    for (let j = 1; j <= m; j++) {
      cur[j] = source[i - 1] === written[j - 1] ? Math.min(65535, prev[j - 1] + 1) : 0;
      if (cur[j] > best.length) best = { length: cur[j], at: j - cur[j] };
    }
    [prev, cur] = [cur, prev];
  }
  return best;
}

/** Sources up to this long are searched for verbatim copies. */
export const COPY_SOURCE_MAX = 60_000;

/**
 * Of the complete names in written text (one per line; the writing starts at the start of a line, so only the last
 * piece may be unfinished), how many are not among the training names.
 */
export function newNames(written: string, training: ReadonlySet<string>): { total: number; fresh: number } {
  const names = written.split("\n").slice(0, -1).filter(Boolean);
  return { total: names.length, fresh: names.filter((n) => !training.has(n)).length };
}
