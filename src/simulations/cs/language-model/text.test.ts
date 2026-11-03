import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { bestTable, referenceLosses, tableLadder, tableResult } from "./ngram";
import { longestCopy, newNames, repetition, distribution, entropy } from "./sample";
import { guessShares, modelGuesses, shannonBounds } from "./shannon";
import { batchStream, buildCorpus, CONTEXT_MARGIN, copyingCorpus, decode, encode, namesCorpus, ownCorpus, passageStart, seedContext, shakespeareCorpus, sonnetCorpus, windowRange, windowStarts } from "./text";

const read = (f: string) => gunzipSync(readFileSync(`public/data/${f}`)).toString("utf8");
const shakespeare = shakespeareCorpus(read("shakespeare.dat"));
const names = namesCorpus(read("names.dat"));

describe("texts", () => {
  it("ship tiny Shakespeare and the names unchanged, with their vocabularies", () => {
    expect(shakespeare.tokens.length).toBe(1_115_394);
    expect(shakespeare.vocab.length).toBe(65);
    expect(shakespeare.vocab.slice(0, 2)).toEqual(["\n", " "]);
    expect(shakespeare.nTrain).toBe(Math.floor(1_115_394 * 0.9));
    expect(names.vocab.join("")).toBe("\nabcdefghijklmnopqrstuvwxyz");
    expect(decode(names, names.tokens).split("\n").filter(Boolean).length).toBe(32_033);
    // The split falls between names.
    expect(names.vocab[names.tokens[names.nTrain - 1]]).toBe("\n");
  });

  it("encode and decode round-trip, dropping unknown symbols", () => {
    const c = sonnetCorpus();
    expect(decode(c, encode(c, "Shall I compare"))).toBe("Shall I compare");
    expect(decode(c, encode(c, "Zebra"))).toBe("ebra");
    expect(c.heldOut).toBe(false);
    expect(c.nTrain).toBe(c.tokens.length);
  });

  it("build the copying game: each run of random letters is followed by itself", () => {
    const c = copyingCorpus(3, 5000);
    const text = decode(c, c.tokens);
    for (const line of text.split("\n").filter(Boolean)) {
      expect(line.length % 2).toBe(0);
      expect(line.slice(0, line.length / 2)).toBe(line.slice(line.length / 2));
      expect(line.length / 2).toBeGreaterThanOrEqual(8);
      expect(line.length / 2).toBeLessThanOrEqual(24);
    }
    // Marked letters: the second copy after its first letter, and only those.
    let at = 0;
    for (const line of text.split("\n").slice(0, 20)) {
      const n = line.length / 2;
      for (let k = 0; k < line.length; k++) {
        expect(c.repeated![at + k]).toBe(k > n ? 1 : 0);
        // The same letter stands n back.
        if (k > n) expect(c.tokens[at + k - c.copyFrom![at + k]]).toBe(c.tokens[at + k]);
      }
      expect(c.repeated![at + line.length]).toBe(0);
      at += line.length + 1;
    }
    expect(c.vocab[c.tokens[c.nTrain - 1]]).toBe("\n");
  });

  it("accept your own text, normalising line ends and capping the vocabulary", () => {
    expect(ownCorpus("too short")).toBeNull();
    const own = ownCorpus("Hello there.\r\n".repeat(50))!;
    expect(decode(own, own.tokens).includes("\r")).toBe(false);
    const wild = ownCorpus(Array.from({ length: 3000 }, (_, i) => String.fromCharCode(0x4e00 + (i % 200))).join(""))!;
    expect(wild.vocab.length).toBeLessThanOrEqual(96);
  });

  it("draw training windows only from the training part and held-out windows only after it", () => {
    for (const T of [16, 64, 128]) {
      const [lo, hi] = windowRange(shakespeare, T, "train");
      expect(lo).toBe(CONTEXT_MARGIN);
      expect(hi + T).toBeLessThan(shakespeare.nTrain);
      const held = windowStarts(shakespeare, T, "heldOut", 200, 3);
      for (const s of held) {
        expect(s).toBeGreaterThanOrEqual(shakespeare.nTrain);
        expect(s + T + 1).toBeLessThanOrEqual(shakespeare.tokens.length);
      }
      const stream = batchStream(shakespeare, T, 16, 1);
      for (let k = 0; k < 20; k++) for (const s of stream.next()) expect(s + T).toBeLessThan(shakespeare.nTrain);
    }
    // Same seed, same windows.
    expect(Array.from(batchStream(names, 32, 8, 5).next())).toEqual(Array.from(batchStream(names, 32, 8, 5).next()));
  });

  it("start passages and writing contexts at the start of a line", () => {
    const nl = shakespeare.vocab.indexOf("\n");
    const p = passageStart(shakespeare, 1024);
    expect(p).toBeGreaterThanOrEqual(shakespeare.nTrain);
    expect(shakespeare.tokens[p - 1]).toBe(nl);
    for (const seed of [1, 2, 3]) {
      const ctx = seedContext(shakespeare, 80, seed);
      expect(ctx.length).toBe(80);
      expect(ctx[79]).toBe(nl);
    }
    const sonnet = sonnetCorpus();
    expect(seedContext(sonnet, 80, 4)[79]).toBe(sonnet.vocab.indexOf("\n"));
  });
});

describe("count tables (exact, on the shipped texts)", () => {
  it("reproduce the plan's Shakespeare ladder: best at 4 letters, worse beyond", () => {
    const g = tableLadder(shakespeare, 6, 50);
    const rows = [];
    for (let r = g.next(); !r.done; r = g.next()) rows.push(r.value);
    const train = [3.309, 2.452, 1.903, 1.491, 1.216, 0.983, 0.761];
    const held = [3.347, 2.482, 2.046, 1.779, 1.75, 1.979, 2.342];
    rows.forEach((r, k) => {
      expect(r.trainLoss).toBeCloseTo(train[k], 3);
      expect(r.heldOutLoss).toBeCloseTo(held[k], 3);
    });
    expect(bestTable(rows).k).toBe(4);
    expect(rows[6].contexts).toBe(264_623);
    // Training loss keeps falling while held-out rises: memorising.
    for (let k = 1; k < 7; k++) expect(rows[k].trainLoss).toBeLessThan(rows[k - 1].trainLoss);
    expect(rows[6].unseen).toBeGreaterThan(0.2);
  }, 60_000);

  it("give the names' bigram makemore's 2.454 and the best table at 3 letters", () => {
    const whole = buildCorpus("names", decode(names, names.tokens), 1);
    expect(tableResult(whole, 1).trainLoss).toBeCloseTo(2.454, 3);
    expect(tableResult(whole, 0).trainLoss).toBeCloseTo(2.823, 3);
    const best = bestTable([0, 1, 2, 3, 4].map((k) => tableResult(names, k)));
    expect(best.k).toBe(3);
    expect(best.heldOutLoss).toBeCloseTo(2.133, 3);
  });

  it("give guessing ln V and order the reference lines", () => {
    const ref = referenceLosses(shakespeare);
    expect(ref.uniform).toBeCloseTo(Math.log(65), 12);
    expect(ref.unigram).toBeCloseTo(3.347, 3);
    expect(ref.bigram).toBeCloseTo(2.482, 3);
  });

  it("write Shannon's approximations: more context, more English", () => {
    const k0 = tableResult(shakespeare, 0, 400).sample;
    const k4 = tableResult(shakespeare, 4, 400).sample;
    const words = (s: string) => s.split(/[^a-z]+/i).filter((w) => w.length >= 4).length;
    expect(words(k4)).toBeGreaterThan(words(k0));
  });
});

describe("measurements of written text", () => {
  it("score loops as repetitive and fresh text as not", () => {
    expect(repetition("the the the the the the the the the the the the ")).toBeGreaterThan(0.7);
    expect(repetition("Shall I compare thee to a summer's day? Thou art more lovely")).toBe(0);
  });

  it("find the longest verbatim copy", () => {
    expect(longestCopy("xx compare thee yy", "Shall I compare thee to")).toEqual({ length: 14, at: 2 });
    expect(longestCopy("zzz", "abc").length).toBe(0);
  });

  it("count invented names", () => {
    expect(newNames("emma\nzorbina\nolivia\nunfini", new Set(["emma", "olivia"]))).toEqual({ total: 3, fresh: 1 });
  });

  it("raise the entropy of the next-letter distribution with the temperature", () => {
    const scores = [2, 1, 0.5, -1];
    const h = [0, 0.5, 1, 2].map((t) => entropy(distribution(scores, t)));
    expect(h[0]).toBe(0);
    for (let k = 1; k < h.length; k++) expect(h[k]).toBeGreaterThan(h[k - 1]);
    expect(h[3]).toBeLessThan(Math.log(4));
  });
});

describe("Shannon's bounds", () => {
  it("bracket the entropy of sources guessed in order of probability", () => {
    for (const p of [
      [0.5, 0.25, 0.125, 0.125],
      [0.7, 0.1, 0.1, 0.05, 0.05],
      [0.25, 0.25, 0.25, 0.25],
    ]) {
      // A perfect guesser of an i.i.d. source needs r guesses with probability p_(r) (sorted).
      const sorted = [...p].sort((a, b) => b - a);
      const guesses: number[] = [];
      sorted.forEach((q, r) => {
        for (let n = 0; n < Math.round(q * 10000); n++) guesses.push(r + 1);
      });
      const H = -p.reduce((a, q) => a + q * Math.log2(q), 0);
      const b = shannonBounds(guesses, p.length);
      expect(b.lower).toBeLessThanOrEqual(H + 1e-9);
      expect(b.upper).toBeCloseTo(H, 6);
    }
    // Uniform over V: both bounds are exactly log2 V.
    const uniform = Array.from({ length: 2600 }, (_, i) => (i % 26) + 1);
    const b = shannonBounds(uniform, 26);
    expect(b.lower).toBeCloseTo(Math.log2(26), 9);
    expect(b.upper).toBeCloseTo(Math.log2(26), 9);
  });

  it("give zero for a perfect predictor and count a model's guesses by rank", () => {
    expect(shannonBounds([1, 1, 1, 1], 27)).toEqual({ lower: 0, upper: 0 });
    expect(guessShares([1, 2, 2, 5], 5)).toEqual([0.25, 0.5, 0, 0, 0.25]);
    expect(modelGuesses([0.1, 0.6, 0.3], 2)).toBe(2);
    expect(modelGuesses([0.1, 0.6, 0.3], 1)).toBe(1);
  });
});

describe("byte-pair tokens", () => {
  it("merge frequent pairs into tokens and split a word the way a chatbot would", async () => {
    const { learnBpe, tokenize, preTokenize, BPE_MERGES } = await import("./bpe");
    expect(preTokenize("Shall I compare thee?\n")).toEqual(["Shall", " I", " compare", " thee", "?", "\n"]);
    const text = decode(shakespeare, shakespeare.tokens.subarray(0, 300_000));
    const t0 = performance.now();
    const g = learnBpe(text, BPE_MERGES);
    let r = g.next();
    while (!r.done) r = g.next();
    const bpe = r.value;
    const ms = performance.now() - t0;
    expect(bpe.merges.length).toBe(BPE_MERGES);
    expect(ms).toBeLessThan(10_000);
    // The commonest words become single tokens; a rare one is cut into pieces.
    expect(tokenize(bpe, " the")).toEqual([" the"]);
    const straw = tokenize(bpe, " strawberry");
    expect(straw.length).toBeGreaterThan(1);
    expect(straw.join("")).toBe(" strawberry");
    // Tokens rejoin into the text exactly.
    const sample = "KING RICHARD III:\nNow is the winter of our discontent";
    expect(tokenize(bpe, sample).join("")).toBe(sample);
    expect(tokenize(bpe, sample).length).toBeLessThan(sample.length / 2);
    console.log(`BPE: ${ms.toFixed(0)} ms; strawberry = ${straw.join("·")}; ${tokenize(bpe, sample).join("·")}`);
  }, 30_000);
});
