"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChallengesPanel } from "@/components/ChallengesPanel";
import { MiniChart } from "@/components/MiniChart";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { usePersistedParams } from "@/lib/usePersistedParams";
import { prefersReducedMotion, usePlaying } from "@/lib/usePlaying";
import { BPE_MERGES, learnBpe, tokenize } from "./bpe";
import type { Bpe } from "./bpe";
import { LM_CHALLENGES, readoutsFor } from "./challenges";
import type { ShannonResult } from "./challenges";
import { parameterCount } from "./model";
import type { ModelKind } from "./model";
import { distribution, entropy } from "./sample";
import { BATCHES, CONTEXTS, fittedBatch, HEADS, HIDDEN, LM_DEFAULTS, LM_SCHEMA, lmConfigOf, LR_FOR, specOf, SPEEDS, VIEWS, WIDTHS } from "./settings";
import type { LmSettings } from "./settings";
import { GAME_LENGTH, shannonBounds } from "./shannon";
import { createLanguageModel } from "./sim";
import type { LmHandle, LmStats, LmView, Prediction, ShannonModel } from "./sim";
import { decode, encode, OWN_MIN } from "./text";
import type { Corpus } from "./text";
import type { TextId } from "./text";
import type { Label } from "./views";

const SIM_ID = "language-model";
const OWN_KEY = "edusim:language-model:own-text";

const VIEW_NAMES: Record<LmView, string> = { train: "Train", predict: "Predict", tables: "Count tables", shannon: "Shannon game" };
const TEXT_NAMES: Record<TextId, string> = { shakespeare: "Shakespeare (1.1 M letters)", names: "Names (32 000 of them)", sonnet: "Sonnet 18 (one poem)", copying: "The copying game", own: "Your own text" };
const MODEL_NAMES: Record<ModelKind, string> = { bigram: "Neural bigram (previous letter only)", mlp: "MLP (the last few letters)", gpt: "Transformer (GPT)" };
const DEFAULT_PROMPT: Record<TextId, string> = {
  shakespeare: "KING RICHARD III:\nNow is the winter of our ",
  names: "\nmar",
  sonnet: "Shall I compare thee ",
  copying: "\nkqzfbwlr",
  own: "",
};

const selectClass = "rounded border border-slate-700 bg-slate-800 p-2 text-sm text-slate-200";
const toggleClass = (on: boolean) => `rounded px-2 py-1.5 text-sm transition-colors ${on ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"}`;
const buttonClass = "rounded bg-slate-800 px-3 py-1.5 text-sm text-slate-300 transition-colors hover:bg-slate-700 disabled:opacity-40";

const nats = (v: number | null | undefined) => (v === null || v === undefined ? "—" : Number.isFinite(v) ? v.toFixed(3) : "∞ (diverged)");
const bits = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "" : ` (${(v / Math.LN2).toFixed(2)} bits)`);
const thousands = (n: number) => n.toLocaleString("en-US");
const symbolName = (s: string) => (s === "\n" ? "↵" : s === " " ? "␣" : s);

function readOwnText(): string {
  try {
    return window.localStorage.getItem(OWN_KEY) ?? "";
  } catch {
    return "";
  }
}

/** The next-letter probabilities as bars: the most likely first; click one to append it. */
function Bars({ prediction, vocab, temperature, onPick }: { prediction: Prediction; vocab: string[]; temperature: number; onPick: (s: string) => void }) {
  const p = distribution(prediction.logits, temperature);
  const order = p.map((x, v) => [x, v] as const).sort((a, b) => b[0] - a[0]).slice(0, 12);
  const h = entropy(p) / Math.LN2;
  return (
    <figure className="space-y-1">
      <figcaption className="text-xs font-semibold text-slate-300">
        Next letter: its {order.length} likeliest (uncertainty {h.toFixed(2)} bits)
      </figcaption>
      <div className="space-y-0.5" data-testid="lm-bars">
        {order.map(([x, v]) => (
          <button key={v} type="button" onClick={() => onPick(vocab[v])} className="flex w-full items-center gap-2 rounded px-1 text-left font-mono text-xs text-slate-300 hover:bg-slate-800" aria-label={`Append ${symbolName(vocab[v])}: ${(x * 100).toFixed(1)} percent`}>
            <span className="w-5 text-center text-sm text-slate-100">{symbolName(vocab[v])}</span>
            <span className="h-3 flex-1 rounded-sm bg-slate-800">
              <span className="block h-3 rounded-sm bg-sky-400" style={{ width: `${Math.max(0.5, x * 100)}%` }} />
            </span>
            <span className="w-12 text-right">{(x * 100).toFixed(1)} %</span>
          </button>
        ))}
      </div>
    </figure>
  );
}

const VOWELS = new Set(Array.from("aeiouAEIOU"));

/** Where the model puts each letter: its letter vectors squashed onto their two main directions. */
function LetterMap({ sim, steps }: { sim: LmHandle; steps: number }) {
  const [map, setMap] = useState<Awaited<ReturnType<LmHandle["letterMap"]>>>(null);
  const due = Math.floor(steps / 100);
  useEffect(() => {
    let live = true;
    sim.letterMap().then((m) => live && setMap(m));
    return () => {
      live = false;
    };
  }, [sim, due]);
  if (!map) return null;
  const pad = 12;
  const W = 280;
  const H = 200;
  const [x0, x1] = [Math.min(...map.x), Math.max(...map.x)];
  const [y0, y1] = [Math.min(...map.y), Math.max(...map.y)];
  const sx = (v: number) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
  const sy = (v: number) => H - pad - ((v - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
  return (
    <figure>
      <figcaption className="mb-1 text-xs font-semibold text-slate-300">Where it puts each letter (letters it treats alike sit close)</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded bg-slate-950/60" role="img" aria-label={`Map of the ${map.symbols.length} letters along the two main directions of the model's letter vectors (${Math.round(map.explained * 100)} % of their spread). Vowels are orange.`} data-testid="lm-letter-map">
        {map.symbols.map((sym, v) => (
          <text key={v} x={sx(map.x[v])} y={sy(map.y[v])} textAnchor="middle" dominantBaseline="middle" fontSize={10} fontFamily="var(--font-mono)" fill={VOWELS.has(sym) ? "#fbbf24" : /[A-Z]/.test(sym) ? "#7dd3fc" : /[a-z]/.test(sym) ? "#e2e8f0" : "#94a3b8"}>
            {symbolName(sym)}
          </text>
        ))}
      </svg>
      <p className="mt-1 text-[11px] text-slate-400">Orange: vowels; blue: capitals. Two directions keep {Math.round(map.explained * 100)} % of how the letters differ.</p>
    </figure>
  );
}

/** Learnt tokenizers, one per text (learning takes a second or two). */
const bpeCache = new Map<string, Bpe>();
const TOKEN_COLOURS = ["bg-sky-500/25", "bg-amber-500/25", "bg-emerald-500/25", "bg-rose-500/25"];

/** How a chatbot would cut the prompt into tokens: a byte-pair tokenizer learnt on the same text. */
function TokensPanel({ corpus, prompt }: { corpus: Corpus; prompt: string }) {
  const key = `${corpus.id}:${corpus.tokens.length}`;
  const [bpe, setBpe] = useState<Bpe | null>(() => bpeCache.get(key) ?? null);
  const [progress, setProgress] = useState<number | null>(null);
  const cancel = useRef<(() => void) | null>(null);
  useEffect(() => () => cancel.current?.(), []);

  const learn = () => {
    const text = decode(corpus, corpus.tokens.subarray(0, Math.min(corpus.nTrain, 200_000)));
    const g = learnBpe(text, BPE_MERGES);
    let stopped = false;
    cancel.current = () => {
      stopped = true;
    };
    const tick = () => {
      if (stopped) return;
      const r = g.next();
      if (r.done) {
        bpeCache.set(key, r.value);
        setBpe(r.value);
        setProgress(null);
      } else {
        setProgress(r.value);
        setTimeout(tick, 0);
      }
    };
    setProgress(0);
    setTimeout(tick, 0);
  };

  const shown = Array.from(prompt).slice(-90).join("");
  const tokens = bpe ? tokenize(bpe, shown) : [];
  const straw = bpe ? tokenize(bpe, " strawberry") : [];
  return (
    <fieldset className="space-y-2 border-t border-slate-800 pt-3">
      <legend className="text-sm font-semibold text-slate-200">Tokens: how a chatbot reads it</legend>
      {!bpe && (
        <>
          <p className="text-xs text-slate-400">This model reads letters. Chatbots read tokens: frequent pieces of words, found by repeatedly merging the commonest pair of symbols in their training text.</p>
          <button type="button" className={buttonClass} disabled={progress !== null} onClick={learn}>
            {progress === null ? `Learn ${BPE_MERGES} tokens from this text` : `Merging… ${progress} of ${BPE_MERGES}`}
          </button>
        </>
      )}
      {bpe && (
        <>
          <p className="font-mono text-xs leading-relaxed text-slate-200" data-testid="lm-tokens">
            {tokens.map((t, k) => (
              <span key={k} className={`rounded-sm px-px ${TOKEN_COLOURS[k % TOKEN_COLOURS.length]}`}>
                {t.replace(/\n/g, "↵").replace(/ /g, "␣")}
              </span>
            ))}
          </p>
          <p className="text-xs text-slate-400">
            {Array.from(shown).length} letters, {tokens.length} tokens. “strawberry” becomes <span className="font-mono text-slate-200">{straw.map((t) => t.trim() || "␣").join(" · ")}</span>: a model that only sees these pieces must have learnt how each one is spelt before it can count the r&apos;s, which is why chatbots famously get it wrong.
          </p>
        </>
      )}
    </fieldset>
  );
}

/** Shannon's guessing game: guess each hidden letter until right. */
function ShannonGame({ sim, stats, onResult }: { sim: LmHandle | null; stats: LmStats | null; onResult: (r: ShannonResult | null) => void }) {
  const [game, setGame] = useState(0);
  const [guesses, setGuesses] = useState<number[]>([]);
  const [wrong, setWrong] = useState<number[]>([]);
  const [model, setModel] = useState<ShannonModel | null>(null);
  const [note, setNote] = useState("");
  const corpus = sim?.corpus() ?? null;
  const ready = stats?.status === "ready" && !!corpus;
  const letters = useMemo(() => (ready && sim ? sim.shannonLetters(game, GAME_LENGTH) : null), [ready, sim, game, stats?.text]); // eslint-disable-line react-hooks/exhaustive-deps
  const revealed = guesses.length;
  const done = !!letters && revealed >= letters.letters.length;
  const V = corpus?.vocab.length ?? 1;

  useEffect(() => {
    sim?.setShannon(letters ? { game, revealed, guesses, wrong } : null);
  }, [sim, letters, game, revealed, guesses, wrong]);
  useEffect(() => () => sim?.setShannon(null), [sim]);

  const restart = (g: number) => {
    setGame(g);
    setGuesses([]);
    setWrong([]);
    setModel(null);
    setNote("");
    onResult(null);
  };

  useEffect(() => {
    if (!done || !sim) return;
    let live = true;
    sim.shannonModel(game, GAME_LENGTH).then((m) => {
      if (!live || !m) return;
      setModel(m);
      const you = shannonBounds(guesses, V);
      const them = shannonBounds(m.ranks, V);
      onResult({ played: guesses.length, you, model: them, modelBits: m.bits.reduce((a, b) => a + b, 0) / m.bits.length });
    });
    return () => {
      live = false;
    };
  }, [done, sim, game, guesses, V, onResult]);

  const guess = (s: string) => {
    if (!letters || !corpus || done) return;
    const tok = corpus.vocab.indexOf(s);
    if (tok < 0) {
      setNote(`“${symbolName(s)}” never occurs in this text.`);
      return;
    }
    if (wrong.includes(tok)) {
      setNote(`You already tried “${symbolName(s)}”.`);
      return;
    }
    setNote("");
    if (tok === letters.letters[revealed]) {
      setGuesses([...guesses, wrong.length + 1]);
      setWrong([]);
    } else setWrong([...wrong, tok]);
  };
  const giveUp = () => {
    if (!letters || done) return;
    setGuesses([...guesses, V]);
    setWrong([]);
  };

  const you = guesses.length ? shannonBounds(guesses, V) : null;
  const them = model ? shannonBounds(model.ranks, V) : null;
  const modelBits = model ? model.bits.reduce((a, b) => a + b, 0) / model.bits.length : null;
  const games = stats && corpus ? Math.max(1, Math.floor(1024 / GAME_LENGTH)) : 1;

  return (
    <fieldset className="space-y-3">
      <legend className="mb-1 text-sm font-semibold text-slate-200">Guess the next letter</legend>
      <p className="text-xs text-slate-400">Type a letter. Wrong guesses are listed; keep going until it is right. Enter is a line break, the space bar a space.</p>
      <input
        aria-label="Your guess for the next letter"
        data-testid="lm-guess"
        disabled={!letters || done}
        className="w-full rounded border border-slate-600 bg-slate-950 px-3 py-2 font-mono text-lg text-slate-100"
        placeholder={done ? "Done: see your score below" : "Type your guess here"}
        value=""
        onChange={(e) => {
          const ch = e.target.value.slice(-1);
          if (ch) guess(ch);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            guess("\n");
          }
        }}
      />
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
        <span data-testid="lm-shannon-progress">
          Letter {Math.min(revealed + 1, GAME_LENGTH)} of {GAME_LENGTH}
        </span>
        {wrong.length > 0 && <span className="font-mono text-rose-300">not: {wrong.map((t) => symbolName(corpus!.vocab[t])).join(" ")}</span>}
        <button type="button" className={buttonClass} disabled={!letters || done} onClick={giveUp}>
          Show it ({V} guesses)
        </button>
        <button type="button" className={buttonClass} onClick={() => restart((game + 1) % games)}>
          New passage
        </button>
      </div>
      {note && <p className="text-xs text-amber-300">{note}</p>}
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
        <dt className="text-slate-500">Guesses per letter</dt>
        <dd data-testid="lm-shannon-guesses">{guesses.length ? (guesses.reduce((a, b) => a + b, 0) / guesses.length).toFixed(2) : "—"}</dd>
        <dt className="text-slate-500">You: bits per letter</dt>
        <dd data-testid="lm-shannon-you">{you ? `${you.lower.toFixed(2)} – ${you.upper.toFixed(2)}` : "—"}</dd>
        <dt className="text-slate-500">The model: bits per letter</dt>
        <dd data-testid="lm-shannon-model">{them ? `${them.lower.toFixed(2)} – ${them.upper.toFixed(2)}` : done ? "measuring…" : "after the game"}</dd>
        <dt className="text-slate-500">Its measured surprise</dt>
        <dd>{modelBits !== null ? `${modelBits.toFixed(2)} bits` : "—"}</dd>
      </dl>
      {done && you && them && (
        <p className="text-xs text-slate-300" data-testid="lm-shannon-verdict">
          {you.upper < them.upper ? "You out-predicted the model on this passage." : "The model out-predicted you on this passage."} The ranges are Shannon&apos;s bounds from the numbers of guesses: the true figure lies in between (more letters would narrow them).
        </p>
      )}
    </fieldset>
  );
}

export default function LanguageModelControls() {
  const [stats, setStats] = useState<LmStats | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [factory] = useState(() => (ctx: Parameters<typeof createLanguageModel>[0]) => createLanguageModel(ctx, { onStats: setStats, onPrediction: setPrediction }));
  const { canvasRef, status, sim, quality } = useGpuSim<LmHandle>(factory);
  const [settings, setSettings] = usePersistedParams(SIM_ID, LM_SCHEMA, LM_DEFAULTS);
  const [playing, setPlaying] = usePlaying(sim, true);
  const [labels, setLabels] = useState<Label[]>([]);
  const [ownText, setOwnText] = useState(readOwnText);
  const [ownDraft, setOwnDraft] = useState(ownText);
  // One prompt per text, so switching texts shows a prompt that fits.
  const [prompts, setPrompts] = useState<Record<TextId, string>>(DEFAULT_PROMPT);
  const [writingOn, setWritingOn] = useState(false);
  const [focus, setFocus] = useState({ layer: 0, head: 0 });
  const [tablesSeen, setTablesSeen] = useState<number[]>([LM_DEFAULTS.tk]);
  const [shannonResult, setShannonResult] = useState<ShannonResult | null>(null);
  const writeSeed = useRef(1);

  const update = (patch: Partial<LmSettings>) => setSettings((prev) => ({ ...prev, ...patch }));
  const view = settings.view;
  const spec = specOf(settings);
  const promptText = prompts[settings.text];
  const setPromptText = useCallback((next: string | ((prev: string) => string)) => setPrompts((all) => ({ ...all, [settings.text]: typeof next === "function" ? next(all[settings.text]) : next })), [settings.text]);
  const shownFocus = { layer: Math.min(focus.layer, spec.L - 1), head: Math.min(focus.head, spec.H - 1) };

  useEffect(() => {
    sim?.configure(lmConfigOf(settings, ownText));
  }, [sim, settings, ownText]);

  useEffect(() => {
    sim?.setPrompt(promptText);
  }, [sim, promptText, stats?.status, stats?.text]);
  useEffect(() => {
    sim?.setFocus(shownFocus);
  }, [sim, shownFocus.layer, shownFocus.head]); // eslint-disable-line react-hooks/exhaustive-deps

  // Captions over the canvas follow the layout.
  useEffect(() => {
    if (!sim) return;
    let frame = 0;
    const refresh = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => setLabels(sim.labels()));
    };
    refresh();
    const timer = setInterval(refresh, 500);
    const canvas = canvasRef.current;
    const observer = typeof ResizeObserver !== "undefined" && canvas ? new ResizeObserver(refresh) : null;
    if (canvas) observer?.observe(canvas);
    return () => {
      cancelAnimationFrame(frame);
      clearInterval(timer);
      observer?.disconnect();
    };
  }, [sim, settings, stats?.tables.length, canvasRef]);

  const onShannonResult = useCallback((r: ShannonResult | null) => setShannonResult(r), []);
  const readouts = readoutsFor(stats, { tablesSeen, shannon: shannonResult, kind: settings.model, layers: spec.L });
  const corpus = sim?.corpus() ?? null;
  const vocab = corpus?.vocab ?? [];
  const params = stats?.status === "ready" && stats.vocab ? parameterCount({ ...spec, V: stats.vocab }) : null;
  const batchUsed = fittedBatch(settings);

  // ── Text ──
  const textControls = (
    <fieldset className="space-y-2">
      <legend className="mb-1 text-sm font-semibold text-slate-200">Text to learn</legend>
      <select aria-label="Text to learn" value={settings.text} onChange={(e) => update({ text: e.target.value as TextId })} className={`${selectClass} w-full`}>
        {(Object.keys(TEXT_NAMES) as TextId[]).map((t) => (
          <option key={t} value={t}>
            {TEXT_NAMES[t]}
          </option>
        ))}
      </select>
      {settings.text === "own" && (
        <div className="space-y-1.5">
          <textarea
            aria-label="Your own text"
            className="h-28 w-full rounded border border-slate-700 bg-slate-950 p-2 font-mono text-xs text-slate-200"
            placeholder={`Paste at least ${OWN_MIN} characters: song lyrics, code, a story…`}
            value={ownDraft}
            onChange={(e) => setOwnDraft(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <button
              type="button"
              className={buttonClass}
              disabled={ownDraft.length < OWN_MIN || ownDraft === ownText}
              onClick={() => {
                setOwnText(ownDraft);
                try {
                  window.localStorage.setItem(OWN_KEY, ownDraft);
                } catch {
                  // Not saved; still used for this visit.
                }
              }}
            >
              Learn this text
            </button>
            <span className="text-xs text-slate-500">{thousands(ownDraft.length)} characters · stays in this browser</span>
          </div>
        </div>
      )}
    </fieldset>
  );

  // ── Model ──
  const modelControls = (
    <fieldset className="space-y-3 rounded-lg border border-slate-700/60 p-3">
      <legend className="px-1 text-sm font-semibold text-slate-200">Model</legend>
      <div className="space-y-1.5 text-sm text-slate-300">
        {(Object.keys(MODEL_NAMES) as ModelKind[]).map((m) => (
          <label key={m} className="flex items-center gap-2">
            <input type="radio" name="lm-model" checked={settings.model === m} onChange={() => update({ model: m, lr: LR_FOR[m] })} />
            {MODEL_NAMES[m]}
          </label>
        ))}
      </div>
      {settings.model === "mlp" && (
        <>
          <ParameterSlider label="Letters of context" unit="" min={1} max={16} step={1} decimals={0} color="blue" value={settings.k} onChange={(v) => update({ k: v })} />
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium text-slate-200">Hidden neurons</span>
            <select value={settings.hidden} onChange={(e) => update({ hidden: Number(e.target.value) })} className={selectClass}>
              {HIDDEN.map((h) => (
                <option key={h} value={h}>
                  {h}
                </option>
              ))}
            </select>
          </label>
        </>
      )}
      {settings.model === "gpt" && (
        <>
          <ParameterSlider label="Layers" unit="" min={1} max={4} step={1} decimals={0} color="blue" value={settings.layers} onChange={(v) => update({ layers: v })} />
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1">
              <span className="text-sm font-medium text-slate-200">Width</span>
              <select value={settings.width} onChange={(e) => update({ width: Number(e.target.value) })} className={selectClass}>
                {WIDTHS.map((w) => (
                  <option key={w} value={w}>
                    {w}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-sm font-medium text-slate-200">Heads</span>
              <select value={settings.heads} onChange={(e) => update({ heads: Number(e.target.value) })} className={selectClass}>
                {HEADS.map((h) => (
                  <option key={h} value={h} disabled={settings.width / h > 64}>
                    {h}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" className={buttonClass} onClick={() => update({ layers: 2, heads: 4, width: 64 })}>
              Default (≈ 0.1 M)
            </button>
            <button type="button" className={buttonClass} onClick={() => update({ layers: 4, heads: 4, width: 128 })}>
              Bigger (≈ 0.8 M)
            </button>
          </div>
        </>
      )}
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">{settings.model === "gpt" ? "Context (letters it can look back over)" : "Window (letters per training example)"}</span>
        <select value={settings.ctx} onChange={(e) => update({ ctx: Number(e.target.value) })} className={selectClass}>
          {CONTEXTS.map((c) => (
            <option key={c} value={c}>
              {c} letters
            </option>
          ))}
        </select>
      </label>
      <p className="text-xs text-slate-400" data-testid="lm-params">
        {params !== null ? `${thousands(params)} numbers to learn (parameters)` : "—"}
      </p>
    </fieldset>
  );

  const trainingControls = (
    <fieldset className="space-y-3">
      <legend className="mb-1 text-sm font-semibold text-slate-200">Training</legend>
      <ParameterSlider label="Learning rate" unit="" min={0.0001} max={1} log decimals={4} color="amber" value={settings.lr} onChange={(v) => update({ lr: Number(v.toPrecision(2)) })} />
      <div className="grid grid-cols-2 gap-2">
        <label className="flex flex-col gap-1">
          <span className="text-sm font-medium text-slate-200">Batch</span>
          <select value={settings.batch} onChange={(e) => update({ batch: Number(e.target.value) })} className={selectClass}>
            {BATCHES.map((b) => (
              <option key={b} value={b}>
                {b} windows
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-medium text-slate-200">Speed</span>
          <select value={settings.speed} onChange={(e) => update({ speed: Number(e.target.value) })} className={selectClass}>
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s === 0 ? "As fast as it can" : `${s} step${s > 1 ? "s" : ""} a frame`}
              </option>
            ))}
          </select>
        </label>
      </div>
      {batchUsed < settings.batch && <p className="text-xs text-amber-300">This model is large: training uses {batchUsed} windows a step to fit in GPU memory.</p>}
      <button type="button" className={buttonClass} onClick={() => update({ seed: (settings.seed % 999999) + 1 })}>
        New random start
      </button>
    </fieldset>
  );

  const temperatureSlider = <ParameterSlider label="Temperature (0 = always the likeliest letter)" unit="" min={0} max={2} step={0.05} decimals={2} color="rose" value={settings.temp} onChange={(v) => update({ temp: v })} />;

  // ── Train view measurements ──
  const refs = stats?.refs;
  const h = stats?.history ?? [];
  const top = refs ? Math.max(refs.uniform + 0.2, ...h.map((p) => (Number.isFinite(p.train) ? p.train : 0))) : 5;
  const levels = refs
    ? [
        { y: refs.uniform, label: "guessing", color: "#94a3b8" },
        ...(refs.unigram !== null ? [{ y: refs.unigram, label: "letter frequencies", color: "#a78bfa" }] : []),
        ...(refs.bigram !== null ? [{ y: refs.bigram, label: "previous letter (count table)", color: "#fbbf24" }] : []),
        ...(refs.best ? [{ y: refs.best.loss, label: `best count table (${refs.best.k} letters)`, color: "#38bdf8" }] : []),
      ]
    : [];
  const w = stats?.writing;
  const trainMeasure = (
    <>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
        <dt className="text-slate-500">Steps · letters seen</dt>
        <dd data-testid="lm-steps">{stats?.status === "ready" ? `${thousands(stats.steps)} · ${thousands(stats.seen)}` : "—"}</dd>
        <dt className="text-slate-500">Steps per second</dt>
        <dd>{stats?.stepsPerSecond ? stats.stepsPerSecond.toFixed(0) : "—"}</dd>
        <dt className="text-slate-500">Training loss</dt>
        <dd data-testid="lm-train-loss">
          {nats(stats?.now?.train)}
          {bits(stats?.now?.train)}
        </dd>
        <dt className="text-slate-500">Unseen-text loss</dt>
        <dd data-testid="lm-held-loss">
          {nats(stats?.now?.held)}
          {bits(stats?.now?.held)}
        </dd>
        <dt className="text-slate-500">As unsure as choosing among</dt>
        <dd data-testid="lm-perplexity">{stats?.now && Number.isFinite(stats.now.held) ? `${Math.exp(stats.now.held).toFixed(1)} letters` : "—"}</dd>
        {stats?.now?.repeated !== undefined && (
          <>
            <dt className="text-slate-500">Loss on copyable letters</dt>
            <dd data-testid="lm-repeated">{nats(stats.now.repeated)}</dd>
            <dt className="text-slate-500">Loss on fresh letters</dt>
            <dd>{nats(stats.now.fresh)}</dd>
          </>
        )}
      </dl>
      {stats?.now?.repeated !== undefined && <p className="text-xs text-slate-400">Copyable: letters of a second copy whose first copy is within the {settings.ctx} letters the model sees. Fresh: letters of first copies, which nothing can predict (ln 26 ≈ 3.26).</p>}
      {!stats?.heldOut && stats?.status === "ready" && <p className="text-xs text-slate-400">This text is too short to hold any back: “unseen” is measured on the text itself.</p>}
      {stats?.diverged && <p className="rounded bg-rose-500/15 px-2 py-1 text-xs text-rose-300">The loss blew up: the learning rate is too high. Lower it and press Reset.</p>}
      {h.length > 1 && (
        <MiniChart
          title="Loss (surprise per letter, nats) as it trains"
          xLabel="steps"
          yLabel="loss"
          x={h.map((p) => p.step)}
          y={h.map((p) => Math.min(top, Number.isFinite(p.train) ? p.train : top))}
          y2={h.map((p) => Math.min(top, Number.isFinite(p.held) ? p.held : top))}
          labels={["training", "unseen"]}
          levels={levels}
          yMin={Math.max(0, Math.min(...h.map((p) => p.held), ...h.map((p) => p.train), refs?.best?.loss ?? 9) - 0.3)}
          yMax={top}
          format={(v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1))}
          summary={`Unseen-text loss ${nats(stats?.now?.held)} after ${stats?.steps ?? 0} steps; guessing is ${refs?.uniform.toFixed(2)}.`}
        />
      )}
      <fieldset className="space-y-2 border-t border-slate-800 pt-3">
        <legend className="text-sm font-semibold text-slate-200">Writing</legend>
        {temperatureSlider}
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
          <dt className="text-slate-500">Letters written</dt>
          <dd data-testid="lm-written">{w ? `${w.letters} per passage` : "—"}</dd>
          <dt className="text-slate-500">Repeats itself</dt>
          <dd data-testid="lm-repetition">{w ? `${Math.round(w.repetition * 100)} %` : "—"}</dd>
          <dt className="text-slate-500">Its own surprise</dt>
          <dd data-testid="lm-own-surprise">{w?.surprise !== null && w?.surprise !== undefined ? `${w.surprise.toFixed(2)} nats/letter` : "—"}</dd>
          {w?.copy !== null && w?.copy !== undefined && (
            <>
              <dt className="text-slate-500">Longest verbatim copy</dt>
              <dd data-testid="lm-copy">{w.copy} letters</dd>
            </>
          )}
          {w?.names && (
            <>
              <dt className="text-slate-500">New names</dt>
              <dd data-testid="lm-new-names">{w.names.total ? `${w.names.fresh} of ${w.names.total} not in the list` : "—"}</dd>
            </>
          )}
        </dl>
      </fieldset>
    </>
  );

  // ── Predict view ──
  const writeOn = async () => {
    if (!sim || writingOn) return;
    setWritingOn(true);
    writeSeed.current++;
    await sim.writeOn(200, settings.temp, writeSeed.current, (text) => setPromptText(text));
    setWritingOn(false);
  };
  const predictControls = (
    <fieldset className="space-y-3">
      <legend className="mb-1 text-sm font-semibold text-slate-200">Prompt</legend>
      <textarea
        aria-label="Prompt"
        data-testid="lm-prompt"
        className="h-24 w-full rounded border border-slate-700 bg-slate-950 p-2 font-mono text-xs text-slate-200"
        value={promptText}
        onChange={(e) => setPromptText(e.target.value)}
      />
      {corpus && promptText.length > 0 && encode(corpus, promptText).length < Array.from(promptText).length && <p className="text-xs text-amber-300">Some of these symbols never occur in the text, so the model skips them.</p>}
      {temperatureSlider}
      <div className="flex flex-wrap gap-2">
        <button type="button" className={buttonClass} disabled={!sim || writingOn || stats?.status !== "ready"} onClick={writeOn} data-testid="lm-write-on">
          Write 200 letters
        </button>
        {writingOn && (
          <button type="button" className={buttonClass} onClick={() => sim?.writeOn(0, 0, 0, () => {})}>
            Stop
          </button>
        )}
        <button type="button" className={buttonClass} onClick={() => setPromptText(DEFAULT_PROMPT[settings.text])}>
          Example prompt
        </button>
      </div>
      {prediction && <Bars prediction={prediction} vocab={vocab} temperature={settings.temp} onPick={(s) => setPromptText((p) => p + s)} />}
      {settings.model === "gpt" && (
        <div className="grid grid-cols-2 gap-2">
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium text-slate-200">Layer</span>
            <select value={shownFocus.layer} onChange={(e) => setFocus({ ...shownFocus, layer: Number(e.target.value) })} className={selectClass}>
              {Array.from({ length: spec.L }, (_, l) => (
                <option key={l} value={l}>
                  {l + 1}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-sm font-medium text-slate-200">Head</span>
            <select value={shownFocus.head} onChange={(e) => setFocus({ ...shownFocus, head: Number(e.target.value) })} className={selectClass}>
              {Array.from({ length: spec.H }, (_, k) => (
                <option key={k} value={k}>
                  {k + 1}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
      <p className="text-xs text-slate-400">The model keeps training while you type (pause it to hold it still).</p>
      {sim && stats?.status === "ready" && <LetterMap sim={sim} steps={stats.steps} />}
      {corpus && <TokensPanel corpus={corpus} prompt={promptText} />}
    </fieldset>
  );

  // ── Count tables view ──
  const tables = stats?.tables ?? [];
  const selected = tables.find((t) => t.k === settings.tk);
  const best = tables.length === 7 ? tables.reduce((a, b) => (b.heldOutLoss < a.heldOutLoss ? b : a)) : null;
  const tablesControls = (
    <fieldset className="space-y-3">
      <legend className="mb-1 text-sm font-semibold text-slate-200">How far back the table looks</legend>
      <div role="group" aria-label="Letters of context" className="grid grid-cols-7 gap-1">
        {Array.from({ length: 7 }, (_, k) => (
          <button
            key={k}
            type="button"
            aria-pressed={settings.tk === k}
            className={toggleClass(settings.tk === k)}
            onClick={() => {
              update({ tk: k });
              setTablesSeen((seen) => (seen.includes(k) ? seen : [...seen, k]));
            }}
          >
            {k}
          </button>
        ))}
      </div>
      <p className="text-xs text-slate-400">A count table predicts the next letter from how often each letter followed the same last k letters in the training text. No learning, just counting.</p>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
        <dt className="text-slate-500">Loss on its training text</dt>
        <dd data-testid="lm-table-train">{selected ? nats(selected.trainLoss) : "counting…"}</dd>
        <dt className="text-slate-500">Loss on unseen text</dt>
        <dd data-testid="lm-table-held">{selected ? nats(selected.heldOutLoss) : "counting…"}</dd>
        <dt className="text-slate-500">Contexts seen</dt>
        <dd>{selected ? thousands(selected.contexts) : "—"}</dd>
        <dt className="text-slate-500">Unseen letters never in context</dt>
        <dd>{selected ? `${(selected.unseen * 100).toFixed(1)} %` : "—"}</dd>
      </dl>
      {tables.length > 1 && (
        <MiniChart
          title="Loss of each count table"
          xLabel="letters of context (k)"
          yLabel="loss (nats)"
          x={tables.map((t) => t.k)}
          y={tables.map((t) => t.trainLoss)}
          y2={tables.map((t) => t.heldOutLoss)}
          labels={["training text", "unseen text"]}
          xMin={0}
          xMax={6}
          yMax={Math.ceil(Math.max(...tables.map((t) => t.heldOutLoss), ...tables.map((t) => t.trainLoss)) + 0.2)}
          point={selected ? { x: selected.k, y: selected.heldOutLoss } : undefined}
          format={(v) => (Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1))}
          summary={best ? `Lowest unseen-text loss ${best.heldOutLoss.toFixed(3)} at ${best.k} letters of context.` : "Counting the tables."}
        />
      )}
      {best && <p className="text-xs text-slate-300">Best on unseen text: {best.k} letters ({best.heldOutLoss.toFixed(3)} nats per letter). The training loss keeps falling, but beyond that the table mostly memorises contexts it will never see again.</p>}
    </fieldset>
  );

  const measure = view === "train" ? trainMeasure : view === "predict" ? predictControls : view === "tables" ? tablesControls : <ShannonGame sim={sim} stats={stats} onResult={onShannonResult} />;

  const controls = (
    <>
      <div role="group" aria-label="View" className="grid grid-cols-2 gap-2">
        {VIEWS.map((v) => (
          <button key={v} type="button" aria-pressed={view === v} onClick={() => update({ view: v })} className={toggleClass(view === v)}>
            {VIEW_NAMES[v]}
          </button>
        ))}
      </div>
      {textControls}
      {stats?.status === "loading" && <p className="text-xs text-slate-400">Loading the text…</p>}
      {stats?.status === "error" && <p className="rounded bg-rose-500/15 px-2 py-1 text-xs text-rose-300">{stats.error}</p>}
      {(view === "train" || view === "predict") && modelControls}
      {view === "train" && trainingControls}
      <fieldset className="space-y-3 border-t border-slate-800 pt-4">
        <legend className="text-sm font-semibold text-slate-200">{view === "train" ? "Measure" : VIEW_NAMES[view]}</legend>
        {measure}
      </fieldset>
    </>
  );

  const explanation = (
    <>
      <p>
        A language model does one thing: given some text, it gives a probability to every possible next letter.
        Writing is just picking one, adding it, and asking again. ChatGPT and Claude work the same way, with pieces of
        words instead of letters and billions of numbers instead of thousands.
      </p>
      <p>
        <strong className="text-slate-100">Loss is surprise.</strong> For each real next letter, the loss is −ln of the
        probability the model gave it: 0 if it was certain, ln 65 ≈ 4.17 nats if it guessed blindly among Shakespeare&apos;s
        65 symbols. Training nudges every number downhill to make real text less surprising. The dashed lines are what
        counting alone achieves on the same letters.
      </p>
      <p>
        <strong className="text-slate-100">The transformer</strong> mixes information between letters with
        <em> attention</em>: each letter looks back at the earlier ones and takes what it needs. The Predict view shows
        where each head looks. Everything trains on your GPU, checked against a 64-bit copy of every step.
      </p>
      <p>
        Texts: tiny Shakespeare (Andrej Karpathy&apos;s char-rnn, public domain); 32 000 US baby names (Social Security
        Administration data via makemore, CC0); Sonnet 18 (1609). Your own text never leaves this browser.
      </p>
    </>
  );

  const summary = !stats
    ? undefined
    : stats.status !== "ready"
      ? stats.status === "error"
        ? stats.error
        : "Loading the text."
      : view === "train"
        ? `${TEXT_NAMES[stats.text]}, ${stats.steps} steps: unseen-text loss ${nats(stats.now?.held)} nats per letter.`
        : view === "tables"
          ? `Count tables: ${selected ? `${settings.tk} letters of context, unseen-text loss ${nats(selected.heldOutLoss)}` : "counting"}.`
          : view === "predict"
            ? "Next-letter probabilities for the prompt."
            : "Shannon's guessing game.";

  return (
    <SimLayout
      summary={summary}
      title="How AI Writes"
      subject="cs"
      difficulty="hard"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      challenges={
        <ChallengesPanel
          simId={SIM_ID}
          challenges={LM_CHALLENGES}
          params={settings}
          readouts={readouts}
          onSetup={(patch) => {
            setStats(null);
            setTablesSeen(patch.tk !== undefined ? [patch.tk] : []);
            setShannonResult(null);
            setPlaying(!prefersReducedMotion());
            update(patch);
          }}
        />
      }
      canvasRef={canvasRef}
      onResetDefaults={() => setSettings(LM_DEFAULTS)}
      playing={playing}
      onPlayPause={setPlaying}
      onStep={() => sim?.step()}
      onReset={() => sim?.reset()}
    >
      <canvas ref={canvasRef} className="absolute inset-0 block h-full w-full" />
      <div aria-hidden className="pointer-events-none absolute inset-0">
        {labels.map((l, k) => (
          <span key={k} className={`absolute whitespace-nowrap text-[11px] text-slate-400 ${l.centred ? "-translate-x-1/2" : ""}`} style={{ left: l.x, top: l.y }}>
            {l.text}
          </span>
        ))}
      </div>
      {view === "train" && (
        <div className="pointer-events-none absolute bottom-3 right-3 hidden rounded bg-black/70 sm:block px-2.5 py-1.5 font-mono text-[11px] leading-snug text-slate-200">
          <div className="text-slate-400">Unseen-text loss</div>
          <div className="text-base font-bold text-sky-300" data-testid="lm-overlay-loss">
            {stats?.now ? nats(stats.now.held) : "—"}
          </div>
        </div>
      )}
    </SimLayout>
  );
}
