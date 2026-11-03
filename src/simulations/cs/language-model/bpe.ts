/**
 * Byte-pair encoding, the way chatbots cut text into tokens: start from single letters and repeatedly merge the most
 * frequent adjacent pair into a new symbol (Sennrich et al. 2016; GPT-2's tokenizer works the same way, on bytes).
 * Words are tokenised separately, a leading space belonging to the word that follows it, as in GPT-2.
 */

/** Words and the spaces before them: " the", " cat", ",", "\n". */
export function preTokenize(text: string): string[] {
  return text.match(/ ?[A-Za-z']+| ?[0-9]+| ?[^\sA-Za-z0-9']+|\s+/g) ?? [];
}

export interface Bpe {
  /** Merges in the order they were learnt (a pair's index is its rank). */
  merges: [string, string][];
}

/** Learns up to `count` merges from a text, yielding every `chunk` merges so a page can stay responsive. */
export function* learnBpe(text: string, count: number, chunk = 25): Generator<number, Bpe> {
  const freq = new Map<string, number>();
  for (const w of preTokenize(text)) freq.set(w, (freq.get(w) ?? 0) + 1);
  const words = [...freq.entries()].map(([w, n]) => ({ symbols: Array.from(w), n }));
  const merges: [string, string][] = [];
  for (let m = 0; m < count; m++) {
    const pairs = new Map<string, number>();
    for (const { symbols, n } of words) {
      for (let i = 0; i + 1 < symbols.length; i++) {
        const key = symbols[i] + "\u0000" + symbols[i + 1];
        pairs.set(key, (pairs.get(key) ?? 0) + n);
      }
    }
    let best = "";
    let bestN = 1; // a pair seen once is not worth a symbol
    for (const [k, n] of pairs) {
      if (n > bestN || (n === bestN && best !== "" && k < best)) {
        best = k;
        bestN = n;
      }
    }
    if (!best) break;
    const [a, b] = best.split("\u0000");
    merges.push([a, b]);
    for (const w of words) {
      const s = w.symbols;
      for (let i = 0; i + 1 < s.length; i++) {
        if (s[i] === a && s[i + 1] === b) s.splice(i, 2, a + b);
      }
    }
    if ((m + 1) % chunk === 0) yield m + 1;
  }
  return { merges };
}

/** Splits text into tokens with learnt merges (lowest rank first, as GPT-2 does). */
export function tokenize(bpe: Bpe, text: string): string[] {
  const rank = new Map(bpe.merges.map(([a, b], k) => [a + "\u0000" + b, k]));
  const out: string[] = [];
  for (const word of preTokenize(text)) {
    const s = Array.from(word);
    for (;;) {
      let best = -1;
      let at = -1;
      for (let i = 0; i + 1 < s.length; i++) {
        const r = rank.get(s[i] + "\u0000" + s[i + 1]);
        if (r !== undefined && (best < 0 || r < best)) {
          best = r;
          at = i;
        }
      }
      if (at < 0) break;
      s.splice(at, 2, s[at] + s[at + 1]);
    }
    out.push(...s);
  }
  return out;
}

/** Merges learnt for the panel (a few hundred, like a very small tokenizer). */
export const BPE_MERGES = 400;
