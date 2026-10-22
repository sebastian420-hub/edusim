"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { ChallengesPanel } from "@/components/ChallengesPanel";
import { MiniChart } from "@/components/MiniChart";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { usePersistedParams } from "@/lib/usePersistedParams";
import { prefersReducedMotion, usePlaying } from "@/lib/usePlaying";
import { NET_CHALLENGES, readoutsFor } from "./challenges";
import { DIGIT_SIZE, toDigit } from "./digits";
import { featureCount, FEATURES, PLANE_SETS, PROBE_S } from "./net";
import type { PlaneSet } from "./net";
import { ACTIVATIONS, BATCH_SIZES, DIGIT_BATCHES, DIGIT_NETS, MAX_HIDDEN_LAYERS, MAX_NEURONS, NET_DEFAULTS, NET_SCHEMA, netConfigOf } from "./settings";
import type { DigitNet, NetSettings, NetView } from "./settings";
import { createNeuralNet } from "./sim";
import type { Label, NetHandle, NetStats, TileMode } from "./sim";

const SIM_ID = "neural-net-viz";

const VIEWS: { id: NetView; label: string }[] = [
  { id: "playground", label: "Playground" },
  { id: "digits", label: "Digits" },
];
const DATA_NAMES: Record<PlaneSet, string> = { blobs: "Two blobs", circle: "Circle", xor: "XOR", spiral: "Spiral" };
const FEATURE_NAMES: Record<(typeof FEATURES)[number], string> = { x: "x", y: "y", x2: "x²", y2: "y²", xy: "x·y", sinx: "sin x", siny: "sin y" };
const ACT_NAMES = { relu: "ReLU", tanh: "tanh", sigmoid: "sigmoid", linear: "linear (none)" } as const;
const DIGIT_NET_NAMES: Record<DigitNet, string> = { linear: "Straight line (no hidden layer)", h128: "One hidden layer (128)", "h128-64": "Two hidden layers (128, 64)" };
const L2_CHOICES = [0, 1e-4, 1e-3, 1e-2, 0.1];

const selectClass = "rounded border border-slate-700 bg-slate-800 p-2 text-sm text-slate-200";
const toggleClass = (on: boolean) =>
  `rounded px-2 py-1.5 text-sm transition-colors ${on ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"}`;
const buttonClass = "rounded bg-slate-800 px-3 py-1.5 text-sm text-slate-300 transition-colors hover:bg-slate-700 disabled:opacity-40";
const smallButton = "h-7 w-7 rounded bg-slate-800 text-sm text-slate-200 transition-colors hover:bg-slate-700 disabled:opacity-40";

const percent = (v: number) => `${(v * 100).toFixed(1)} %`;
const lossText = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "∞ (diverged)");
/** Largest loss in a run (an infinite loss counts as very large). */
const lossMax = (h: { trainLoss: number; testLoss: number }[]) => Math.max(0, ...h.map((p) => (Number.isFinite(p.trainLoss) ? p.trainLoss : 1e9)), ...h.map((p) => (Number.isFinite(p.testLoss) ? p.testLoss : 1e9)));
/** The loss chart's top: the whole curve, but at most 5 (a guess is ln 2 ≈ 0.69, so 5 is already far off). */
const lossTop = (h: { trainLoss: number; testLoss: number }[]) => Math.min(5, Math.max(0.8, Math.ceil(lossMax(h) * 10) / 10));


/** The pad's 2D context; read back on every stroke, so the browser keeps it on the CPU. */
const padContext = (c: HTMLCanvasElement | null) => c?.getContext("2d", { willReadFrequently: true }) ?? null;

/** A 28 × 28 drawing pad: draw with the mouse or a finger; the network's ten probabilities appear below. */
function DrawPad({ sim, stats }: { sim: NetHandle | null; stats: NetStats | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawing = useRef<{ x: number; y: number } | null>(null);
  const [probs, setProbs] = useState<number[] | null>(null);
  const [empty, setEmpty] = useState(true);

  const classify = useCallback(() => {
    const c = ref.current;
    if (!c || !sim) return;
    const ctx = padContext(c);
    if (!ctx) return;
    const { data, width, height } = ctx.getImageData(0, 0, c.width, c.height);
    const ink = new Float64Array(width * height);
    for (let k = 0; k < ink.length; k++) ink[k] = data[4 * k + 3] / 255;
    const digit = toDigit(ink, width, height);
    setEmpty(!digit);
    setProbs(digit ? sim.classify(digit) : null);
  }, [sim]);

  // New weights: read the drawing again.
  useEffect(() => {
    if (!empty) classify();
  }, [stats, classify, empty]);

  const at = (e: PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * e.currentTarget.width, y: ((e.clientY - r.top) / r.height) * e.currentTarget.height };
  };
  const stroke = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const ctx = padContext(ref.current);
    if (!ctx) return;
    ctx.strokeStyle = "#f8fafc";
    ctx.lineWidth = 16;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  };
  const best = probs ? probs.indexOf(Math.max(...probs)) : -1;

  return (
    <div className="space-y-2">
      <div className="flex items-start gap-3">
        <canvas
          ref={ref}
          width={DIGIT_SIZE * 7}
          height={DIGIT_SIZE * 7}
          aria-label="Drawing pad: draw a digit from 0 to 9"
          className="h-36 w-36 touch-none rounded border border-slate-600 bg-slate-950"
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            const p = at(e);
            drawing.current = p;
            stroke(p, { x: p.x + 0.1, y: p.y });
          }}
          onPointerMove={(e) => {
            if (!drawing.current) return;
            const p = at(e);
            stroke(drawing.current, p);
            drawing.current = p;
          }}
          onPointerUp={() => {
            drawing.current = null;
            classify();
          }}
          onPointerCancel={() => {
            drawing.current = null;
          }}
        />
        <div className="flex flex-col gap-2">
          <div className="text-xs text-slate-400">Its reading</div>
          <div className="font-mono text-3xl font-bold text-sky-300" data-testid="nn-pad-guess">
            {best >= 0 ? best : "—"}
          </div>
          <div className="text-xs text-slate-400" data-testid="nn-pad-confidence">
            {best >= 0 && probs ? `${Math.round(probs[best] * 100)} % sure` : empty ? "draw a digit" : "train first"}
          </div>
          <button
            type="button"
            className={buttonClass}
            onClick={() => {
              const c = ref.current;
              if (c) padContext(c)?.clearRect(0, 0, c.width, c.height);
              setProbs(null);
              setEmpty(true);
            }}
          >
            Clear
          </button>
        </div>
      </div>
      <div className="grid grid-cols-10 items-end gap-0.5" aria-label="Probability of each digit" role="img" data-testid="nn-pad-bars">
        {Array.from({ length: 10 }, (_, d) => (
          <div key={d} className="flex flex-col items-center gap-0.5">
            <div className="flex h-12 w-full items-end rounded-sm bg-slate-800/60">
              <div className={`w-full rounded-sm ${d === best ? "bg-sky-400" : "bg-slate-500"}`} style={{ height: `${Math.round((probs?.[d] ?? 0) * 100)}%` }} />
            </div>
            <span className="font-mono text-[10px] text-slate-400">{d}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** The test set's confusion matrix: rows are the true digit, columns the guess; the diagonal is right. */
function Confusion({ matrix }: { matrix: number[][] }) {
  const max = Math.max(1, ...matrix.flatMap((row, i) => row.filter((_, j) => j !== i)));
  return (
    <figure>
      <figcaption className="mb-1 text-xs font-semibold text-slate-300">Confusion: true digit (rows) against its guess (columns)</figcaption>
      <table className="w-full table-fixed border-collapse font-mono text-[9px] text-slate-300" data-testid="nn-confusion">
        <thead>
          <tr>
            <th className="w-4" />
            {matrix.map((_, j) => (
              <th key={j} className="font-normal text-slate-500">
                {j}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="font-normal text-slate-500">{i}</th>
              {row.map((v, j) => (
                <td
                  key={j}
                  title={`${v} × true ${i} read as ${j}`}
                  className="h-4 text-center"
                  style={{ background: i === j ? `rgba(0, 158, 115, ${Math.min(0.85, v / 250)})` : v ? `rgba(213, 94, 0, ${0.15 + 0.7 * (v / max)})` : undefined }}
                >
                  {v || ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

export default function NeuralNetControls() {
  const [stats, setStats] = useState<NetStats | null>(null);
  const [factory] = useState(() => (ctx: Parameters<typeof createNeuralNet>[0]) => createNeuralNet(ctx, { onStats: setStats }));
  const { canvasRef, status, sim, quality } = useGpuSim<NetHandle>(factory);
  const [settings, setSettings] = usePersistedParams(SIM_ID, NET_SCHEMA, NET_DEFAULTS);
  const [playing, setPlaying] = usePlaying(sim, true);
  const [showTest, setShowTest] = useState(true);
  const [tileMode, setTileMode] = useState<TileMode>("all");
  const [labels, setLabels] = useState<Label[]>([]);
  const [hover, setHover] = useState<{ x: number; y: number; p: number } | null>(null);

  const update = (patch: Partial<NetSettings>) => setSettings((prev) => ({ ...prev, ...patch }));
  const digitsView = settings.view === "digits";

  useEffect(() => {
    sim?.configure(netConfigOf(settings, showTest));
  }, [sim, settings, showTest]);
  useEffect(() => {
    sim?.setTileMode(tileMode);
  }, [sim, tileMode]);

  // Labels over the canvas follow the layout: refresh on changes and resizes.
  useEffect(() => {
    if (!sim) return;
    let frame = 0;
    const refresh = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => setLabels(sim.labels()));
    };
    refresh();
    const canvas = canvasRef.current;
    const observer = typeof ResizeObserver !== "undefined" && canvas ? new ResizeObserver(refresh) : null;
    if (canvas) observer?.observe(canvas);
    const later = setTimeout(refresh, 300);
    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(later);
      observer?.disconnect();
    };
  }, [sim, settings, tileMode, canvasRef]);

  const play = stats?.view === "playground" ? stats : null;
  const dig = stats?.view === "digits" ? stats : null;
  const readouts = readoutsFor(stats);

  // ── Playground controls ──
  const setLayer = (k: number, n: number) => update({ layers: settings.layers.map((w, i) => (i === k ? Math.min(MAX_NEURONS, Math.max(1, n)) : w)) });
  const dataControls = (
    <fieldset className="space-y-3">
      <legend className="mb-1 text-sm font-semibold text-slate-200">Data</legend>
      <div role="group" aria-label="Dataset" className="grid grid-cols-4 gap-1.5">
        {PLANE_SETS.map((d) => (
          <button key={d} type="button" aria-pressed={settings.data === d} className={toggleClass(settings.data === d)} onClick={() => update({ data: d })}>
            {DATA_NAMES[d]}
          </button>
        ))}
      </div>
      <ParameterSlider label="Points" unit="" min={100} max={2000} step={100} decimals={0} color="slate" value={settings.points} onChange={(v) => update({ points: v })} />
      <ParameterSlider label="Wrong labels (noise)" unit=" %" min={0} max={40} step={5} decimals={0} color="rose" value={Math.round(settings.noise * 100)} onChange={(v) => update({ noise: v / 100 })} />
      <ParameterSlider label="Used for training" unit=" %" min={10} max={90} step={5} decimals={0} color="green" value={Math.round(settings.train * 100)} onChange={(v) => update({ train: v / 100 })} />
      <label className="flex items-center gap-2 text-sm text-slate-300">
        <input type="checkbox" checked={showTest} onChange={(e) => setShowTest(e.target.checked)} />
        Show test points (rings)
      </label>
    </fieldset>
  );

  const inputControls = (
    <fieldset>
      <legend className="mb-2 text-sm font-semibold text-slate-200">Inputs</legend>
      <div role="group" aria-label="Input features" className="flex flex-wrap gap-1.5">
        {FEATURES.map((f, k) => {
          const on = ((settings.features >> k) & 1) === 1;
          const last = on && featureCount(settings.features) === 1;
          return (
            <button key={f} type="button" aria-pressed={on} disabled={last} className={toggleClass(on)} onClick={() => update({ features: settings.features ^ (1 << k) })}>
              {FEATURE_NAMES[f]}
            </button>
          );
        })}
      </div>
    </fieldset>
  );

  const networkControls = (
    <fieldset className="space-y-3 rounded-lg border border-slate-700/60 p-3">
      <legend className="px-1 text-sm font-semibold text-slate-200">Network</legend>
      <div className="flex items-center gap-2 text-sm text-slate-300">
        <span className="flex-1" data-testid="nn-layer-count">
          {settings.layers.length === 0 ? "No hidden layer (a straight line)" : `${settings.layers.length} hidden layer${settings.layers.length > 1 ? "s" : ""}`}
        </span>
        <button type="button" className={buttonClass} disabled={settings.layers.length === 0} onClick={() => update({ layers: settings.layers.slice(0, -1) })}>
          − layer
        </button>
        <button type="button" className={buttonClass} disabled={settings.layers.length >= MAX_HIDDEN_LAYERS} onClick={() => update({ layers: [...settings.layers, settings.layers.at(-1) ?? 4] })}>
          + layer
        </button>
      </div>
      {settings.layers.map((n, k) => (
        <div key={k} className="flex items-center gap-2 text-sm text-slate-300">
          <span className="flex-1">Layer {k + 1}</span>
          <button type="button" className={smallButton} aria-label={`Fewer neurons in layer ${k + 1}`} disabled={n <= 1} onClick={() => setLayer(k, n - 1)}>
            −
          </button>
          <span className="w-16 text-center font-mono" data-testid={`nn-layer-${k + 1}`}>
            {n} neuron{n > 1 ? "s" : ""}
          </span>
          <button type="button" className={smallButton} aria-label={`More neurons in layer ${k + 1}`} disabled={n >= MAX_NEURONS} onClick={() => setLayer(k, n + 1)}>
            +
          </button>
        </div>
      ))}
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Activation</span>
        <select value={settings.act} onChange={(e) => update({ act: ACTIVATIONS.find((a) => a === e.target.value) ?? settings.act })} className={selectClass}>
          {ACTIVATIONS.map((a) => (
            <option key={a} value={a}>
              {ACT_NAMES[a]}
            </option>
          ))}
        </select>
      </label>
      <fieldset>
        <legend className="mb-1 text-sm font-medium text-slate-200">Start</legend>
        <div className="flex flex-wrap items-center gap-4 text-sm text-slate-300">
          {(["random", "zeros"] as const).map((i) => (
            <label key={i} className="flex items-center gap-1.5">
              <input type="radio" name="nn-init" checked={settings.init === i} onChange={() => update({ init: i })} />
              {i === "random" ? "Random" : "All zeros"}
            </label>
          ))}
          <button type="button" className={buttonClass} onClick={() => update({ seed: (settings.seed % 999999) + 1 })}>
            New random start
          </button>
        </div>
      </fieldset>
    </fieldset>
  );

  const trainingControls = (
    <fieldset className="space-y-3">
      <legend className="mb-1 text-sm font-semibold text-slate-200">Training</legend>
      <ParameterSlider label="Learning rate (step size)" unit="" min={0.001} max={30} log decimals={3} color="amber" value={settings.lr} onChange={(v) => update({ lr: Number(v.toPrecision(2)) })} />
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Batch size</span>
        <select value={settings.batch} onChange={(e) => update({ batch: Number(e.target.value) })} className={selectClass}>
          {BATCH_SIZES.map((b) => (
            <option key={b} value={b}>
              {b === 2000 ? "All training points (gradient descent)" : b === 1 ? "1 point per step" : `${b} points per step`}
            </option>
          ))}
        </select>
      </label>
      <ParameterSlider label="Momentum" unit="" min={0} max={0.99} step={0.01} decimals={2} color="blue" value={settings.mom} onChange={(v) => update({ mom: v })} />
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">L2 (pull towards small weights)</span>
        <select value={settings.l2} onChange={(e) => update({ l2: Number(e.target.value) })} className={selectClass}>
          {L2_CHOICES.map((v) => (
            <option key={v} value={v}>
              {v === 0 ? "None" : v}
            </option>
          ))}
        </select>
      </label>
    </fieldset>
  );

  const probe = play?.probe ?? null;
  const uphill = probe ? probe[8] > probe[0] : null;
  const playgroundMeasure = (
    <>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
        <dt className="text-slate-500">Steps · passes</dt>
        <dd data-testid="nn-steps">{play ? `${play.steps.toLocaleString("en-US")} · ${play.epochs.toFixed(1)}` : "—"}</dd>
        <dt className="text-slate-500">Training loss</dt>
        <dd data-testid="nn-train-loss">{play?.now ? lossText(play.now.trainLoss) : "—"}</dd>
        <dt className="text-slate-500">Test loss</dt>
        <dd data-testid="nn-test-loss">{play?.now ? lossText(play.now.testLoss) : "—"}</dd>
        <dt className="text-slate-500">Training accuracy</dt>
        <dd data-testid="nn-train-acc">{play?.now ? percent(play.now.trainAcc) : "—"}</dd>
        <dt className="text-slate-500">Test accuracy</dt>
        <dd data-testid="nn-test-acc">{play?.now ? percent(play.now.testAcc) : "—"}</dd>
        <dt className="text-slate-500">Hidden neurons differ by</dt>
        <dd data-testid="nn-spread">{play?.spread !== null && play?.spread !== undefined ? (play.spread === 0 ? "0 (all clones)" : play.spread.toFixed(3)) : "—"}</dd>
      </dl>
      {play?.diverged && <p className="rounded bg-rose-500/15 px-2 py-1 text-xs text-rose-300">The loss blew up to infinity: the steps are far too big. Lower the learning rate and press Reset.</p>}
      {play && play.history.length > 1 && (
        <MiniChart
          title={lossTop(play.history) < lossMax(play.history) ? `Loss during training (cut off at ${lossTop(play.history)})` : "Loss during training"}
          xLabel="steps"
          yLabel="loss"
          x={play.history.map((p) => p.step)}
          y={play.history.map((p) => Math.min(lossTop(play.history), p.trainLoss))}
          y2={play.history.map((p) => Math.min(lossTop(play.history), p.testLoss))}
          labels={["training", "test"]}
          yMax={lossTop(play.history)}
          format={(v) => v.toFixed(2)}
          summary={`Training loss ${play.now ? lossText(play.now.trainLoss) : "—"}, test loss ${play.now ? lossText(play.now.testLoss) : "—"} after ${play.steps} steps.`}
        />
      )}
      {probe && probe.every(Number.isFinite) && (
        <>
          <MiniChart
            title="Step probe: the loss along the last step"
            xLabel="0 = before the step, 1 = where it landed, 2 = twice as far"
            yLabel="loss"
            x={[...PROBE_S]}
            y={probe}
            yMin={Math.max(0, Math.min(...probe) * 0.9)}
            yMax={Math.max(...probe) * 1.05 + 1e-6}
            markers={[{ x: 1, label: "landed" }]}
            point={{ x: 1, y: probe[8] }}
            format={(v) => (Number.isInteger(v) ? v.toFixed(0) : v.toFixed(2))}
            summary={`The last step went ${uphill ? "uphill: the loss is higher where it landed than where it started" : "downhill"}.`}
          />
          <p className={`text-xs ${uphill ? "text-rose-300" : "text-emerald-300"}`} data-testid="nn-probe">
            {uphill ? "The last step went uphill: it overshot the valley." : "The last step went downhill."}
          </p>
        </>
      )}
      {play?.gradients && (
        <figure>
          <figcaption className="mb-1 text-xs font-semibold text-slate-300">Gradient by layer (how much error reaches it)</figcaption>
          <div className="space-y-0.5" data-testid="nn-gradients">
            {play.gradients.map((g, k) => {
              const top = Math.max(...play.gradients!, 1e-12);
              const width = g > 0 ? Math.max(1, 100 + 20 * Math.log10(g / top)) : 0; // a decade per fifth of the bar
              return (
                <div key={k} className="flex items-center gap-2 font-mono text-[10px] text-slate-400">
                  <span className="w-12">{k === play.gradients!.length - 1 ? "output" : `layer ${k + 1}`}</span>
                  <div className="h-2 flex-1 rounded-sm bg-slate-800">
                    <div className="h-2 rounded-sm bg-amber-400" style={{ width: `${Math.max(0, width)}%` }} />
                  </div>
                  <span className="w-14 text-right">{g.toExponential(0)}</span>
                </div>
              );
            })}
          </div>
        </figure>
      )}
    </>
  );

  // ── Digits controls ──
  const digitsControls = (
    <>
      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">Network</legend>
        <div className="space-y-1.5 text-sm text-slate-300">
          {DIGIT_NETS.map((n) => (
            <label key={n} className="flex items-center gap-2">
              <input type="radio" name="nn-digits-net" checked={settings.dnet === n} onChange={() => update({ dnet: n })} />
              {DIGIT_NET_NAMES[n]}
            </label>
          ))}
        </div>
      </fieldset>
      <ParameterSlider label="Learning rate (step size)" unit="" min={0.001} max={3} log decimals={3} color="amber" value={settings.dlr} onChange={(v) => update({ dlr: Number(v.toPrecision(2)) })} />
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Batch size</span>
        <select value={settings.dbatch} onChange={(e) => update({ dbatch: Number(e.target.value) })} className={selectClass}>
          {DIGIT_BATCHES.map((b) => (
            <option key={b} value={b}>
              {b} digits per step
            </option>
          ))}
        </select>
      </label>
      <div role="group" aria-label="Which test digits to show" className="grid grid-cols-2 gap-2">
        {(["all", "mistakes"] as const).map((m) => (
          <button key={m} type="button" aria-pressed={tileMode === m} className={toggleClass(tileMode === m)} onClick={() => setTileMode(m)}>
            {m === "all" ? "All test digits" : "Only mistakes"}
          </button>
        ))}
      </div>
    </>
  );

  const digitsMeasure = (
    <>
      {dig?.status === "loading" && <p className="text-xs text-slate-400">Loading 12 000 handwritten digits (2 MB)…</p>}
      {dig?.status === "error" && <p className="rounded bg-rose-500/15 px-2 py-1 text-xs text-rose-300">The digits could not be loaded.</p>}
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
        <dt className="text-slate-500">Steps · passes</dt>
        <dd data-testid="nn-steps">{dig ? `${dig.steps.toLocaleString("en-US")} · ${dig.epochs.toFixed(2)}` : "—"}</dd>
        <dt className="text-slate-500">Training accuracy</dt>
        <dd data-testid="nn-train-acc">{dig?.now ? percent(dig.now.trainAcc) : "—"}</dd>
        <dt className="text-slate-500">Test accuracy</dt>
        <dd data-testid="nn-test-acc">{dig?.now ? percent(dig.now.testAcc) : "—"}</dd>
        <dt className="text-slate-500">Test mistakes</dt>
        <dd data-testid="nn-mistakes">{dig?.now ? `${dig.mistakes} of ${dig.nTest}` : "—"}</dd>
      </dl>
      {dig && dig.history.length > 1 && (
        <MiniChart
          title="Accuracy during training"
          xLabel="passes over the 10 000 training digits"
          yLabel="accuracy (%)"
          x={dig.history.map((p) => p.epoch)}
          y={dig.history.map((p) => p.trainAcc * 100)}
          y2={dig.history.map((p) => p.testAcc * 100)}
          labels={["training", "test"]}
          yMin={Math.min(80, ...dig.history.map((p) => p.testAcc * 100))}
          yMax={100}
          format={(v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1))}
          summary={`Test accuracy ${dig.now ? percent(dig.now.testAcc) : "—"} after ${dig.epochs.toFixed(1)} passes.`}
        />
      )}
      {dig?.confusion && <Confusion matrix={dig.confusion} />}
      <fieldset className="space-y-2 border-t border-slate-800 pt-3">
        <legend className="text-sm font-semibold text-slate-200">Draw a digit</legend>
        <DrawPad sim={sim} stats={stats} />
        <p className="text-xs text-slate-400">Draw it big and centred, like the digits on the canvas.</p>
      </fieldset>
    </>
  );

  const controls = (
    <>
      <div role="group" aria-label="View" className="grid grid-cols-2 gap-2">
        {VIEWS.map((v) => (
          <button key={v.id} type="button" aria-pressed={settings.view === v.id} onClick={() => update({ view: v.id })} className={toggleClass(settings.view === v.id)}>
            {v.label}
          </button>
        ))}
      </div>
      {!digitsView && dataControls}
      {!digitsView && inputControls}
      {!digitsView && networkControls}
      {!digitsView && trainingControls}
      {digitsView && digitsControls}
      <ParameterSlider label="Speed" unit=" steps/frame" min={1} max={50} step={1} decimals={0} color="blue" value={settings.speed} onChange={(v) => update({ speed: v })} />
      <fieldset className="space-y-3 border-t border-slate-800 pt-4">
        <legend className="text-sm font-semibold text-slate-200">Measure</legend>
        {digitsView ? digitsMeasure : playgroundMeasure}
      </fieldset>
    </>
  );

  const explanation = (
    <>
      <p>
        A neural network is a function with thousands of adjustable numbers, its <em>weights</em>. Training shows it
        examples, measures how wrong it is (the <em>loss</em>), and nudges every weight a little downhill:
        <strong className="text-slate-100"> gradient descent</strong>. Nothing is programmed in; the error just gets
        smaller.
      </p>
      <p>
        <strong className="text-slate-100">Backpropagation</strong> finds the downhill direction for every weight at once
        with the chain rule, passing the blame for the error back layer by layer. Here it runs on your GPU, checked
        against a 64-bit copy and against finite differences.
      </p>
      <p>
        <strong className="text-slate-100">Playground.</strong> The background is the network&apos;s answer for every point
        (orange or blue, paler where it is unsure); each small square in the diagram is one neuron&apos;s own picture, and
        the lines are weights (blue positive, orange negative). Rings are test points, never trained on.
      </p>
      <p>
        <strong className="text-slate-100">Digits.</strong> The same mechanism on real handwriting: 10 000 digits from
        the MNIST set (Yann LeCun, Corinna Cortes and Christopher Burges, CC BY-SA 3.0) to learn from, 2 000 to test it on.
      </p>
    </>
  );

  const summary = !stats
    ? undefined
    : stats.view === "playground"
      ? `${DATA_NAMES[settings.data]} data, ${settings.layers.length} hidden layers, ${stats.steps} steps: ${stats.now ? `test accuracy ${percent(stats.now.testAcc)}, training loss ${lossText(stats.now.trainLoss)}` : "not measured yet"}.`
      : stats.status !== "ready"
        ? "Loading handwritten digits."
        : `Handwritten digits, ${stats.epochs.toFixed(1)} passes: ${stats.now ? `test accuracy ${percent(stats.now.testAcc)}, ${stats.mistakes} mistakes` : "not measured yet"}.`;

  return (
    <SimLayout
      summary={summary}
      title="How AI Learns"
      subject="cs"
      difficulty="hard"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      challenges={
        <ChallengesPanel
          simId={SIM_ID}
          challenges={NET_CHALLENGES}
          params={settings}
          readouts={readouts}
          onSetup={(patch) => {
            setStats(null);
            setPlaying(!prefersReducedMotion());
            update(patch);
          }}
        />
      }
      canvasRef={canvasRef}
      onResetDefaults={() => setSettings(NET_DEFAULTS)}
      playing={playing}
      onPlayPause={setPlaying}
      onStep={() => sim?.step()}
      onReset={() => sim?.reset()}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 block h-full w-full touch-none"
        onPointerMove={(e) => {
          if (!sim || digitsView || e.pointerType === "touch") return;
          const r = e.currentTarget.getBoundingClientRect();
          setHover(sim.probabilityAt(e.clientX - r.left, e.clientY - r.top));
        }}
        onPointerLeave={() => setHover(null)}
      />
      <div aria-hidden className="pointer-events-none absolute inset-0">
        {labels.map((l, k) => (
          <span key={k} className="absolute -translate-x-1/2 whitespace-nowrap text-[11px] text-slate-400" style={{ left: l.x, top: l.y }}>
            {l.text}
          </span>
        ))}
      </div>
      {!digitsView && (
        <div className="pointer-events-none absolute left-3 top-3 rounded bg-black/60 px-2.5 py-1.5 font-mono text-[11px] leading-snug text-slate-200">
          <div className="text-slate-400">Test accuracy</div>
          <div className="text-base font-bold text-sky-300" data-testid="nn-overlay-acc">
            {play?.now ? percent(play.now.testAcc) : "—"}
          </div>
          {hover && (
            <div>
              ({hover.x.toFixed(2)}, {hover.y.toFixed(2)}): {Math.round(hover.p * 100)} % blue
            </div>
          )}
        </div>
      )}
    </SimLayout>
  );
}
