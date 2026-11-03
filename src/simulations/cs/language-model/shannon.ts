/**
 * Shannon's guessing game (1951): guess the next letter of a hidden text until you get it right. The numbers of
 * guesses you needed bound the entropy of the text *as you predict it*, in bits per letter:
 *
 *   Σ_r r (q_r − q_{r+1}) log₂ r  ≤  H  ≤  −Σ_r q_r log₂ q_r
 *
 * where q_r is the share of letters that took exactly r guesses. The upper bound is the entropy of the guess counts
 * (a perfect compressor could store your counts instead of the letters); the lower bound is Shannon's.
 */

/** q_r for r = 1 … maxGuesses (index r − 1). */
export function guessShares(guesses: readonly number[], maxGuesses: number): number[] {
  const q = new Array(maxGuesses).fill(0);
  for (const g of guesses) q[Math.min(maxGuesses, Math.max(1, g)) - 1]++;
  return q.map((c) => (guesses.length ? c / guesses.length : 0));
}

export function shannonBounds(guesses: readonly number[], maxGuesses: number): { lower: number; upper: number } {
  const q = guessShares(guesses, maxGuesses);
  let upper = 0;
  let lower = 0;
  for (let r = 1; r <= maxGuesses; r++) {
    const qr = q[r - 1];
    const next = r < maxGuesses ? q[r] : 0;
    if (qr > 0) upper -= qr * Math.log2(qr);
    lower += r * (qr - next) * Math.log2(r);
  }
  return { lower, upper };
}

/** How many guesses a model needs for the true letter: its rank in the model's ordering (1 = its top choice). */
export function modelGuesses(probabilities: ArrayLike<number>, truth: number): number {
  const p = probabilities[truth];
  let rank = 1;
  for (let v = 0; v < probabilities.length; v++) if (probabilities[v] > p || (probabilities[v] === p && v < truth)) rank++;
  return rank;
}

/** Letters in one game. */
export const GAME_LENGTH = 50;
