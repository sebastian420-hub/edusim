/**
 * Population history of a running automaton, and what it says about the pattern: extinct, still life,
 * oscillator (with its period), moving (population repeats but the pattern travels), or still evolving.
 */

export type PatternStatus =
  | { kind: "empty" }
  | { kind: "still" }
  | { kind: "oscillator"; period: number }
  | { kind: "moving"; period?: number }
  | { kind: "evolving" };

const MAX_PERIOD = 120;

/** Smallest p such that `values` repeats with period p over its last `window(p)` entries, if any. */
function smallestPeriod(values: number[], window: (p: number) => number): number | undefined {
  const n = values.length;
  for (let p = 1; p <= MAX_PERIOD; p++) {
    const w = window(p);
    if (n < w + p) break; // not enough history to verify this period (or any larger one)
    let ok = true;
    for (let i = n - w; i < n && ok; i++) ok = values[i] === values[i - p];
    if (ok) return p;
  }
  return undefined;
}

/**
 * Classifies a pattern from per-generation `counts` (live cells) and `hashes` (state fingerprints).
 * An exact repeat of the fingerprint over two full periods is a still life / oscillator; a repeating
 * population whose fingerprint never repeats means something is travelling (a glider or spaceship).
 */
export function analyze(counts: number[], hashes: number[]): PatternStatus {
  if (counts.length === 0) return { kind: "evolving" };
  if (counts[counts.length - 1] === 0) return { kind: "empty" };

  const exact = smallestPeriod(hashes, (p) => Math.max(2 * p, 6));
  if (exact !== undefined) return exact === 1 ? { kind: "still" } : { kind: "oscillator", period: exact };

  const byCount = smallestPeriod(counts, (p) => Math.max(3 * p, 12));
  if (byCount !== undefined) return { kind: "moving", period: byCount > 1 ? byCount : undefined };
  return { kind: "evolving" };
}

/** Rolling history of (generation, population, fingerprint). Resets itself if generations aren't consecutive. */
export class PopulationTracker {
  gens: number[] = [];
  counts: number[] = [];
  hashes: number[] = [];

  constructor(private readonly capacity = 600) {}

  reset(): void {
    this.gens = [];
    this.counts = [];
    this.hashes = [];
  }

  push(generation: number, count: number, hash: number): void {
    const last = this.gens[this.gens.length - 1];
    if (last !== undefined && generation !== last + 1) this.reset();
    this.gens.push(generation);
    this.counts.push(count);
    this.hashes.push(hash);
    if (this.gens.length > this.capacity) {
      this.gens.shift();
      this.counts.shift();
      this.hashes.shift();
    }
  }

  status(): PatternStatus {
    return analyze(this.counts, this.hashes);
  }
}

export function describeStatus(status: PatternStatus): string {
  switch (status.kind) {
    case "empty":
      return "Extinct";
    case "still":
      return "Still life";
    case "oscillator":
      return `Oscillator, period ${status.period}`;
    case "moving":
      return status.period ? `Moving pattern, population repeats every ${status.period}` : "Moving pattern (glider or spaceship)";
    case "evolving":
      return "Evolving…";
  }
}
