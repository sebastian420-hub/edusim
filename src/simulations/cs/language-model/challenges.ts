import type { Challenge } from "@/lib/challenges";
import type { ModelKind } from "./model";
import type { TableResult } from "./ngram";
import type { LmSettings } from "./settings";
import type { CurvePoint, LmStats, Refs, WritingStats } from "./sim";
import { GAME_LENGTH } from "./shannon";
import type { TextId } from "./text";

/** Measured thresholds (each one checked on the GPU in challenges.test.ts and engine experiments). */
export const BIGRAM_TOLERANCE = 0.02;
export const LOOP_REPETITION = 0.3;
export const LOW_TEMPERATURE = 0.2;
export const VERBATIM = 80;
export const COPY_LOSS = 1;

export interface ShannonResult {
  played: number;
  you: { lower: number; upper: number };
  model: { lower: number; upper: number };
  modelBits: number;
}

/** What challenge goals can observe. */
export interface LmReadouts {
  ready: boolean;
  text: TextId | null;
  kind: ModelKind;
  layers: number;
  steps: number;
  now: CurvePoint | null;
  refs: Refs | null;
  writing: WritingStats | null;
  tables: TableResult[];
  /** Count tables the student has looked at since the challenge started. */
  tablesSeen: number[];
  shannon: ShannonResult | null;
}

export function readoutsFor(stats: LmStats | null, extra: { tablesSeen: number[]; shannon: ShannonResult | null; kind: ModelKind; layers: number }): LmReadouts {
  const ready = stats?.status === "ready";
  return {
    ready,
    text: ready ? stats!.text : null,
    kind: extra.kind,
    layers: extra.layers,
    steps: stats?.steps ?? 0,
    now: stats?.now ?? null,
    refs: stats?.refs ?? null,
    writing: stats?.writing ?? null,
    tables: stats?.tables ?? [],
    tablesSeen: extra.tablesSeen,
    shannon: extra.shannon,
  };
}

const bestOf = (tables: readonly TableResult[]) => (tables.length === 7 ? tables.reduce((a, b) => (b.heldOutLoss < a.heldOutLoss ? b : a)) : null);

/** The default transformer on Shakespeare, as every challenge starts from (where it trains one). */
const base: Partial<LmSettings> = { view: "train", text: "shakespeare", model: "gpt", ctx: 64, layers: 2, heads: 4, width: 64, batch: 16, lr: 0.003, seed: 1, speed: 0, temp: 1 };

/** Guided experiments. Every number in the texts is measured on the shipped texts (challenges.test.ts). */
export const LM_CHALLENGES: Challenge<LmSettings, LmReadouts>[] = [
  {
    id: "bigram",
    title: "How surprised?",
    prompt:
      "The simplest neural language model: a table of 65 × 65 scores, one for each letter after each letter, learnt by gradient descent. It only ever sees the previous letter. Press Play and watch its loss on unseen text.",
    setup: { ...base, model: "bigram", lr: 0.05 },
    prediction: {
      question: "Seeing only the previous letter, how low can its loss on unseen Shakespeare go?",
      options: [{ label: "about 4.2 nats: no better than guessing" }, { label: "about 2.5 nats", correct: true }, { label: "about 1.0 nat" }],
    },
    goal: {
      description: `Train the neural bigram until its unseen-text loss is within ${BIGRAM_TOLERANCE} of the count table that looks at the previous letter (the yellow line).`,
      check: ({ params, readouts: r }) => r.ready && r.text === "shakespeare" && params.model === "bigram" && r.kind === "bigram" && r.steps >= 200 && r.now !== null && r.refs?.bigram !== null && r.refs?.bigram !== undefined && Math.abs(r.now.held - r.refs.bigram) <= BIGRAM_TOLERANCE,
    },
    hint: "Just let it train: a few hundred steps. If it stalls above the line, the default learning rate (0.05) is a good one.",
    explanation:
      "Counting and learning meet: the loss of a bigram table is a bowl with one bottom, and that bottom is exactly the table of counts (how often each letter followed each letter). Gradient descent finds it. So 2.45 nats (3.5 bits) per letter is the limit of any model that sees one letter back, however it is built. To do better, a model has to look further back.",
  },
  {
    id: "tables",
    title: "More context, until it memorises",
    prompt:
      "Count tables that look further back: the next letter given the last k letters, counted in the training text. Their writing is on the canvas, from k = 0 (letter frequencies) down to k = 6. Look at the losses as k grows.",
    setup: { view: "tables", text: "shakespeare", tk: 1 },
    prediction: {
      question: "As the table looks further back, its loss on text it has never seen…",
      options: [{ label: "keeps falling" }, { label: "falls, then rises", correct: true }, { label: "stays flat" }],
    },
    goal: {
      description: "Find the table with the lowest loss on unseen text, after looking at one that looks further back than it.",
      check: ({ params, readouts: r }) => {
        const best = bestOf(r.tables);
        return r.text === "shakespeare" && params.view === "tables" && best !== null && params.tk === best.k && r.tablesSeen.some((k) => k > best.k);
      },
    },
    hint: "Try k = 5 and 6, then come back down. The chart under the buttons shows both losses.",
    explanation:
      "On its own training text a longer context always helps (the green line falls to 0.76 nats at k = 6). On unseen text it helps up to 4 letters (1.75 nats), then hurts: 6-letter contexts are so many (264 623) that a fifth of the unseen letters come after a context the table has never seen, and it can only guess. It has memorised instead of generalised. A neural network does not store contexts: it learns what similar contexts have in common, which is why it can use much more of the past.",
  },
  {
    id: "beat-table",
    title: "Beat the table",
    prompt:
      "Names are short, so a neural model learns them quickly. The blue dashed line is the best count table on the same unseen names. Train a transformer on the names.",
    setup: { ...base, text: "names" },
    prediction: {
      question: "Can a neural network beat the best count table on names it has never seen?",
      options: [{ label: "No: counting is exact, so it is the best possible" }, { label: "Yes, a little: it shares what it learns between similar names", correct: true }, { label: "Yes: it gets the loss to zero" }],
    },
    goal: {
      description: "Get the unseen-text loss on names below the best count table (the blue dashed line).",
      check: ({ readouts: r }) => r.ready && r.text === "names" && r.kind !== "bigram" && r.now !== null && r.refs?.best !== null && r.refs?.best !== undefined && r.now.held < r.refs.best.loss,
    },
    hint: "Keep it training; it should cross the line within a minute or two. Watch the “New names” readout: most of what it writes is invented, not copied.",
    explanation:
      "A count table treats “mar” and “mer” as unrelated contexts; a neural model learns letter embeddings in which similar letters sit close together, so what it learns after one context carries over to similar ones it has rarely seen. That is generalisation, and it is why the transformer beats the best table even though the table is exact about what it has seen. It still writes mostly new names: it learnt how names are built, not the list.",
  },
  {
    id: "temperature",
    title: "Turn down the temperature",
    prompt:
      "Temperature divides the model's scores before they become probabilities: below 1 the likely letters get likelier, above 1 everything evens out. Let the transformer train on Shakespeare for a while, then turn the temperature down.",
    setup: { ...base },
    prediction: {
      question: "At a temperature near 0 (always the likeliest letter), the model writes…",
      options: [{ label: "its best text" }, { label: "the same phrase over and over", correct: true }, { label: "nothing at all" }],
    },
    goal: {
      description: `After 500 steps of training or more, write a full passage at temperature ${LOW_TEMPERATURE} or less and see it repeat itself (${Math.round(LOOP_REPETITION * 100)} % or more).`,
      check: ({ readouts: r }) => r.ready && r.steps >= 500 && r.writing !== null && r.writing.temperature <= LOW_TEMPERATURE && r.writing.letters >= 200 && r.writing.repetition >= LOOP_REPETITION,
    },
    hint: "Under Writing, drag the temperature to 0 and wait for the passages to fill. Then try 2: noise, and its own surprise at what it wrote shoots up.",
    explanation:
      "The likeliest next letter is a safe bet, but always taking it leads into loops: once “the ” is the likeliest continuation of itself a few letters later, the text cycles for ever. Real chatbots sample (with a temperature around 0.7–1) for exactly this reason. Turn it up to 2 and the rare letters become as likely as the common ones: the model's own surprise at its writing climbs towards guessing. Temperature is a dial between safe and varied, not between dumb and clever.",
  },
  {
    id: "by-heart",
    title: "Learn by heart",
    prompt:
      "Sonnet 18 is 621 letters; the transformer has more than 100 000 numbers to learn. Train it on the sonnet and read what it writes.",
    setup: { ...base, text: "sonnet", temp: 0.5 },
    prediction: {
      question: "Trained on one short poem, what will the model write?",
      options: [{ label: "new poems in Shakespeare's style" }, { label: "the poem itself, word for word", correct: true }, { label: "random letters" }],
    },
    goal: {
      description: `Get the model to write a run of ${VERBATIM} letters or more copied verbatim from the sonnet.`,
      check: ({ readouts: r }) => r.ready && r.text === "sonnet" && r.writing !== null && r.writing.copy !== null && r.writing.copy >= VERBATIM,
    },
    hint: "Let it train for a few hundred steps; the “Longest verbatim copy” readout counts the letters it reproduces exactly.",
    explanation:
      "With far more numbers than the text has letters, the easiest way to make the loss small is to store the text: training rewards memorising. Big language models memorise too, especially text that appears many times in their data, which is why they can reproduce passages verbatim: a problem for privacy (personal data in the training set) and copyright. The fix is more, more varied data, and checks for copying.",
  },
  {
    id: "copy",
    title: "Copy what you just saw",
    prompt:
      "The copying game: runs of random letters, each followed by an exact copy (“qmzf…qmzf…”). The first half is unpredictable; the second half is easy, for a model that can find the current letter earlier in the line and copy what came after it. Here is a transformer with one layer.",
    setup: { ...base, text: "copying", layers: 1, ctx: 32, batch: 32 },
    prediction: {
      question: "A one-layer transformer learns to predict the copied half…",
      options: [{ label: "easily" }, { label: "slowly" }, { label: "never: it can't find where the copy started", correct: true }],
    },
    goal: {
      description: `With two or more layers, get the loss on the copyable letters below ${COPY_LOSS} nat.`,
      check: ({ readouts: r }) => r.ready && r.text === "copying" && r.layers >= 2 && r.now?.repeated !== undefined && r.now.repeated < COPY_LOSS,
    },
    hint: "With one layer, “Loss on copyable letters” creeps from ln 26 ≈ 3.26 down to about 2.5 (it learns which letters are in the current run) and stops there. Set Layers to 2 and keep it training: after a long, slow stretch the loss drops suddenly, around 2 000–3 000 steps in.",
    explanation:
      "Copying needs two steps: first each letter notes which letter came before it (one layer of attention), then the current letter looks for the place where its own letter came before, and copies the letter there (a second layer). This pair is called an induction head, and its sudden appearance during training is a well-known phase change. It is how language models learn from examples in a prompt: in-context learning.",
  },
  {
    id: "shannon",
    title: "You against the machine",
    prompt:
      "Claude Shannon measured how predictable English is by asking people to guess the next letter of a hidden text. Play his game on Shakespeare the model has never seen, then compare yourself with the model.",
    setup: { view: "shannon", text: "shakespeare" },
    prediction: {
      question: "Shannon found that people predict English at about how many bits per letter?",
      options: [{ label: "about 1", correct: true }, { label: "about 3" }, { label: "about 6 (guessing)" }],
    },
    goal: {
      description: `Play a whole passage of ${GAME_LENGTH} letters.`,
      check: ({ params, readouts: r }) => params.view === "shannon" && r.shannon !== null && r.shannon.played >= GAME_LENGTH,
    },
    hint: "Type into the guess box; Enter is a line break. “Show it” gives up on a letter.",
    explanation:
      "Each guess count tells how surprising the letter was to you, and Shannon proved the counts bound the true entropy from both sides. People know words, grammar and the story, so they need about one bit per letter of English; a model you can train in a tab needs two or three. Big models closed that gap with vastly more text and size, not with a different idea.",
  },
];
