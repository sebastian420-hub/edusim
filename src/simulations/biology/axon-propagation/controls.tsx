"use client";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { ChallengesPanel } from "@/components/ChallengesPanel";
import { MiniChart } from "@/components/MiniChart";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { usePersistedParams } from "@/lib/usePersistedParams";
import { usePlaying } from "@/lib/usePlaying";
import { AXON_CHALLENGES, readoutsFor } from "./challenges";
import { AXON_LAYOUT, NERVE_LAYOUT } from "./renderer";
import { AXON_DEFAULTS, AXON_SCHEMA, effectiveDiameter, MAX_MYELINATED, NERVE_COUNTS } from "./settings";
import type { AxonSettings, Drug, Pulses } from "./settings";
import { createAxonSim } from "./sim";
import type { AxonHandle, AxonStats, FrameInfo } from "./sim";
import type { NerveFibre } from "./model";

const SIM_ID = "axon-propagation";

const selectClass = "rounded border border-slate-700 bg-slate-800 p-2 text-sm text-slate-200";
const toggleClass = (on: boolean) =>
  `rounded px-2 py-1.5 text-sm transition-colors ${on ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"}`;

const FIBRES = [
  { label: "Squid giant axon (476 µm)", diameter: 476, myelin: false },
  { label: "Bare 10 µm fibre", diameter: 10, myelin: false },
  { label: "Myelinated 10 µm fibre", diameter: 10, myelin: true },
  { label: "C fibre (1 µm, bare)", diameter: 1, myelin: false },
];

const DRUGS: { id: Drug; label: string }[] = [
  { id: "none", label: "Nothing" },
  { id: "ttx", label: "TTX (all Na⁺ channels blocked)" },
  { id: "lidocaine", label: "Lidocaine (90 % blocked)" },
];

const PULSES: { id: Pulses; label: string }[] = [
  { id: "single", label: "One end" },
  { id: "pair", label: "Twice" },
  { id: "both", label: "Both ends" },
];

const ELECTRODE_COLORS = ["#34d399", "#f472b6"];
const lengthLabel = (cm: number) => (cm >= 1 ? `${cm.toFixed(cm >= 10 ? 0 : 1)} cm` : `${(cm * 10).toFixed(cm >= 0.1 ? 1 : 2)} mm`);
const ms = (v: number | undefined, digits = 2) => (v === undefined ? "—" : `${v.toFixed(digits)} ms`);

interface OverlayLooks {
  settings: AxonSettings;
  drag: { key: "e1" | "e2" | "from" | "to" | "distance"; value: number } | null;
  stats: AxonStats | null;
}

/** Draws labels, electrodes, the treated stretch and (nerve) the compound action potential over the WebGPU canvas. */
function drawOverlay(ctx: CanvasRenderingContext2D, w: number, h: number, info: FrameInfo, looks: OverlayLooks) {
  ctx.clearRect(0, 0, w, h);
  const s = { ...looks.settings };
  if (looks.drag) (s as Record<string, unknown>)[looks.drag.key] = looks.drag.value;
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "middle";
  if (info.view === "axon") {
    const L = AXON_LAYOUT;
    const x0 = L.margin * w;
    const x1 = (1 - L.margin) * w;
    const X = (f: number) => x0 + f * (x1 - x0);
    const tubeTop = L.tubeTop * h;
    const tubeBottom = L.tubeBottom * h;
    const kTop = L.kymoTop * h;
    const kBottom = L.kymoBottom * h;
    // Treated stretch.
    const treated = s.drug !== "none" || (s.myelin && s.myelinLeft < 1);
    if (treated) {
      const a = X(Math.min(s.from, s.to));
      const b = X(Math.max(s.from, s.to));
      ctx.fillStyle = "rgba(167, 139, 250, 0.16)";
      ctx.fillRect(a, tubeTop - 14, b - a, kBottom - tubeTop + 14);
      ctx.strokeStyle = "rgba(167, 139, 250, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(a, tubeTop - 14, b - a, tubeBottom - tubeTop + 14);
      ctx.fillStyle = "#c4b5fd";
      ctx.textAlign = "center";
      const label = s.drug === "ttx" ? "TTX" : s.drug === "lidocaine" ? "lidocaine" : `myelin ${Math.round(s.myelinLeft * 100)} %`;
      ctx.fillText(label, (a + b) / 2, tubeTop - 22);
    }
    // Stimulating electrodes.
    ctx.fillStyle = "#fbbf24";
    ctx.textAlign = "left";
    ctx.fillText("⚡ stimulus", x0, tubeTop - 22);
    if (s.pulses === "both") {
      ctx.textAlign = "right";
      ctx.fillText("stimulus ⚡", x1, tubeTop - 22);
    }
    // Recording electrodes: dashed lines through the tube and the kymograph.
    (["e1", "e2"] as const).forEach((key, k) => {
      const x = X(s[key]);
      ctx.strokeStyle = ELECTRODE_COLORS[k];
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, tubeTop - 6);
      ctx.lineTo(x, kBottom);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ELECTRODE_COLORS[k];
      ctx.beginPath();
      ctx.arc(x, tubeBottom + 10, 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = "#0f172a";
      ctx.textAlign = "center";
      ctx.fillText(String(k + 1), x, tubeBottom + 10.5);
    });
    // Axes: position under the kymograph, time on its left; the sweep's progress.
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "center";
    for (const f of [0, 0.5, 1]) ctx.fillText(lengthLabel(f * info.length), X(f), kBottom + 12);
    ctx.textAlign = "right";
    for (const f of [0, 0.5, 1]) ctx.fillText(`${(f * info.sweepMs).toFixed(f ? 1 : 0)}`, x0 - 6, kTop + f * (kBottom - kTop));
    ctx.save();
    ctx.translate(12, (kTop + kBottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("time (ms) ↓", 0, 0);
    ctx.restore();
    const now = kTop + Math.min(1, info.time / info.sweepMs) * (kBottom - kTop);
    ctx.strokeStyle = "rgba(226, 232, 240, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, now);
    ctx.lineTo(x1, now);
    ctx.stroke();
  } else {
    const L = NERVE_LAYOUT;
    const x0 = L.margin * w;
    const x1 = (1 - L.margin) * w;
    const X = (cm: number) => x0 + (cm / info.length) * (x1 - x0);
    const top = L.top * h;
    const bottom = L.bottom * h;
    // Fibre groups along the right edge.
    const comp: NerveFibre[] = info.composition ?? [];
    const groups = ["Aα/β", "Aδ", "C"] as const;
    ctx.textAlign = "right";
    groups.forEach((g) => {
      const idx = comp.flatMap((f, i) => (f.kind === g ? [i] : []));
      if (!idx.length) return;
      const y = top + ((idx[0] + idx[idx.length - 1] + 1) / 2 / comp.length) * (bottom - top);
      ctx.fillStyle = "#cbd5e1";
      ctx.fillText(g, w - 6, y);
    });
    ctx.fillStyle = "#fbbf24";
    ctx.textAlign = "left";
    ctx.fillText("⚡ stimulus", x0, top - 10);
    const xe = X(s.distance);
    ctx.strokeStyle = ELECTRODE_COLORS[0];
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xe, top - 4);
    ctx.lineTo(xe, bottom + 4);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ELECTRODE_COLORS[0];
    ctx.textAlign = "center";
    ctx.fillText(`recording · ${s.distance.toFixed(2)} cm`, xe, bottom + 14);
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "left";
    ctx.fillText("0", x0, bottom + 14);
    ctx.textAlign = "right";
    ctx.fillText(lengthLabel(info.length), x1, bottom + 14);
    // Compound action potential.
    const stats = looks.stats;
    const capTop = bottom + 34;
    const capBottom = h - 22;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
    ctx.beginPath();
    ctx.moveTo(x0, (capTop + capBottom) / 2);
    ctx.lineTo(x1, (capTop + capBottom) / 2);
    ctx.stroke();
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "left";
    ctx.fillText("compound action potential at the electrode", x0, capTop - 8);
    ctx.textAlign = "center";
    for (const f of [0, 0.5, 1]) ctx.fillText(`${(f * info.sweepMs).toFixed(0)} ms`, x0 + f * (x1 - x0), capBottom + 12);
    if (stats?.view === "nerve" && stats.cap.t.length > 1) {
      const { t, v } = stats.cap;
      const scale = Math.max(1e-12, ...v.map(Math.abs));
      ctx.strokeStyle = ELECTRODE_COLORS[0];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      t.forEach((ti, i) => {
        const x = x0 + (ti / info.sweepMs) * (x1 - x0);
        // Upward deflection = negative extracellular potential (inward current under the electrode), as on a scope.
        const y = (capTop + capBottom) / 2 + (v[i] / scale) * 0.45 * (capBottom - capTop);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }
}

export default function AxonControls() {
  const [stats, setStats] = useState<AxonStats | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const looks = useRef<OverlayLooks>({ settings: AXON_DEFAULTS, drag: null, stats: null });
  const [factory] = useState(
    () => (ctx: Parameters<typeof createAxonSim>[0]) =>
      createAxonSim(ctx, {
        onStats: setStats,
        onFrame: (info) => {
          const canvas = overlayRef.current;
          const ctx2d = canvas?.getContext("2d");
          if (!canvas || !ctx2d) return;
          const width = canvas.clientWidth;
          const height = canvas.clientHeight;
          const dpr = window.devicePixelRatio || 1;
          if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
          }
          ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
          drawOverlay(ctx2d, width, height, info, looks.current);
        },
      }),
  );
  const { canvasRef, status, sim, quality } = useGpuSim<AxonHandle>(factory);
  const [settings, setSettings] = usePersistedParams(SIM_ID, AXON_SCHEMA, AXON_DEFAULTS);
  const [playing, setPlaying] = usePlaying(sim, true);
  const [drag, setDrag] = useState<OverlayLooks["drag"]>(null);

  useEffect(() => {
    looks.current = { settings, drag, stats };
  });
  useEffect(() => {
    sim?.configure(settings);
  }, [sim, settings]);
  useEffect(() => {
    sim?.setSpeed(settings.speed);
  }, [sim, settings.speed]);

  const update = (patch: Partial<AxonSettings>) => setSettings((prev) => ({ ...prev, ...patch }));
  const axon = settings.view === "axon";
  const treated = settings.drug !== "none" || (settings.myelin && settings.myelinLeft < 1);

  // ── Dragging electrodes and the edges of the treated stretch ──
  const fractionAt = (e: PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const m = (axon ? AXON_LAYOUT : NERVE_LAYOUT).margin;
    return { f: Math.min(1, Math.max(0, ((e.clientX - rect.left) / rect.width - m) / (1 - 2 * m))), y: (e.clientY - rect.top) / rect.height, width: rect.width };
  };
  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    const { f, width } = fractionAt(e);
    const reach = (e.pointerType === "touch" ? 28 : 14) / (width * (1 - 2 * AXON_LAYOUT.margin));
    let best: { key: NonNullable<OverlayLooks["drag"]>["key"]; d: number } | null = null;
    const consider = (key: NonNullable<OverlayLooks["drag"]>["key"], at: number) => {
      const d = Math.abs(at - f);
      if (d <= reach && (!best || d < best.d)) best = { key, d };
    };
    if (axon) {
      consider("e1", settings.e1);
      consider("e2", settings.e2);
      if (treated) {
        consider("from", settings.from);
        consider("to", settings.to);
      }
    } else {
      consider("distance", settings.distance / 2);
    }
    const hit = best as { key: NonNullable<OverlayLooks["drag"]>["key"]; d: number } | null;
    if (!hit) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDrag({ key: hit.key, value: hit.key === "distance" ? settings.distance : settings[hit.key] });
  };
  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!drag) return;
    const { f } = fractionAt(e);
    const value = drag.key === "distance" ? Math.min(1.9, Math.max(0.3, f * 2)) : drag.key === "from" || drag.key === "to" ? f : Math.min(0.98, Math.max(0.02, f));
    setDrag({ ...drag, value: Number(value.toFixed(3)) });
  };
  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    if (!drag) return;
    update({ [drag.key]: drag.value } as Partial<AxonSettings>);
    setDrag(null);
  };

  // ── Readouts ──
  const live = stats?.view === "axon" ? stats.live : null;
  const done = stats?.view === "axon" ? stats.done : null;
  const shown = done ?? live;
  const cap = stats?.view === "nerve" ? (stats.done ?? stats.cap) : null;
  const readouts = readoutsFor(stats);
  const d = effectiveDiameter(settings);
  const presetValue = FIBRES.findIndex((p) => p.diameter === d && p.myelin === settings.myelin);

  const axonControls = (
    <>
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Fibre</span>
        <select
          value={presetValue < 0 ? "" : presetValue}
          onChange={(e) => {
            const p = FIBRES[Number(e.target.value)];
            if (p) update({ diameter: p.diameter, myelin: p.myelin });
          }}
          className={selectClass}
        >
          {presetValue < 0 && <option value="">Custom: {d} µm{settings.myelin ? ", myelinated" : ""}</option>}
          {FIBRES.map((p, i) => (
            <option key={p.label} value={i}>
              {p.label}
            </option>
          ))}
        </select>
      </label>
      <ParameterSlider label="Diameter" unit=" µm" min={0.5} max={settings.myelin ? MAX_MYELINATED : 1000} log decimals={d < 10 ? 1 : 0} color="blue" value={d} onChange={(v) => update({ diameter: v })} />
      <label className="flex items-center gap-2 text-sm text-slate-300">
        <input type="checkbox" checked={settings.myelin} onChange={(e) => update({ myelin: e.target.checked, diameter: e.target.checked ? Math.min(settings.diameter, 10) : settings.diameter })} />
        Myelin (nodes of Ranvier every 100 diameters)
      </label>
      <ParameterSlider label="Temperature" unit=" °C" min={0} max={40} step={0.1} decimals={1} color="rose" value={settings.temp} onChange={(v) => update({ temp: v })} />

      <fieldset className="space-y-3 rounded-lg border border-slate-700/60 p-3">
        <legend className="px-1 text-sm font-semibold text-slate-200">Stimulus</legend>
        <div role="group" aria-label="Stimulate" className="grid grid-cols-3 gap-2">
          {PULSES.map((p) => (
            <button key={p.id} type="button" aria-pressed={settings.pulses === p.id} onClick={() => update({ pulses: p.id })} className={toggleClass(settings.pulses === p.id)}>
              {p.label}
            </button>
          ))}
        </div>
        <ParameterSlider label="Stimulus" unit="× threshold" min={0.2} max={6} step={0.1} decimals={1} color="amber" value={settings.stim} onChange={(v) => update({ stim: v })} />
        {settings.pulses === "pair" && <ParameterSlider label="Interval" unit=" ms" min={0.5} max={20} step={0.5} decimals={1} color="amber" value={settings.gap} onChange={(v) => update({ gap: v })} />}
      </fieldset>

      <fieldset className="space-y-3 rounded-lg border border-slate-700/60 p-3">
        <legend className="px-1 text-sm font-semibold text-slate-200">Treated stretch</legend>
        <label className="flex flex-col gap-1">
          <span className="text-xs text-slate-400">Drug</span>
          <select aria-label="Drug" value={settings.drug} onChange={(e) => update({ drug: e.target.value as Drug })} className={selectClass}>
            {DRUGS.map((x) => (
              <option key={x.id} value={x.id}>
                {x.label}
              </option>
            ))}
          </select>
        </label>
        {settings.myelin && <ParameterSlider label="Myelin left" unit=" %" min={0} max={100} step={1} decimals={0} color="purple" value={Math.round(settings.myelinLeft * 100)} onChange={(v) => update({ myelinLeft: v / 100 })} />}
        <ParameterSlider label="Stretch starts" unit=" %" min={0} max={100} step={1} decimals={0} color="slate" value={Math.round(settings.from * 100)} onChange={(v) => update({ from: v / 100 })} />
        <ParameterSlider label="Stretch ends" unit=" %" min={0} max={100} step={1} decimals={0} color="slate" value={Math.round(settings.to * 100)} onChange={(v) => update({ to: v / 100 })} />
        <p className="text-xs text-slate-400">Drag the electrodes (1, 2) and the edges of the shaded stretch on the canvas.</p>
      </fieldset>
    </>
  );

  const nerveControls = (
    <>
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Fibres</span>
        <select
          value={settings.fibres}
          onChange={(e) => {
            const fibres = NERVE_COUNTS.find((c) => String(c) === e.target.value);
            if (fibres) update({ fibres });
          }}
          className={selectClass}
        >
          {NERVE_COUNTS.map((c) => (
            <option key={c} value={c}>
              {c.toLocaleString("en-US")} fibres
            </option>
          ))}
        </select>
      </label>
      <ParameterSlider label="Shock strength" unit="" min={0.2} max={12} step={0.1} decimals={1} color="amber" value={settings.nerveStim} onChange={(v) => update({ nerveStim: v })} />
      <ParameterSlider label="Recording distance" unit=" cm" min={0.3} max={1.9} step={0.05} decimals={2} color="green" value={settings.distance} onChange={(v) => update({ distance: v })} />
      <ParameterSlider label="Temperature" unit=" °C" min={0} max={40} step={0.1} decimals={1} color="rose" value={settings.temp} onChange={(v) => update({ temp: v })} />
      <p className="text-xs text-slate-400">A weak shock excites only the thickest fibres; raise it to recruit the thin ones too.</p>
    </>
  );

  const controls = (
    <>
      <div role="group" aria-label="View" className="grid grid-cols-2 gap-2">
        {(["axon", "nerve"] as const).map((v) => (
          <button key={v} type="button" aria-pressed={settings.view === v} onClick={() => update({ view: v })} className={toggleClass(settings.view === v)}>
            {v === "axon" ? "Axon" : "Nerve"}
          </button>
        ))}
      </div>
      {axon ? axonControls : nerveControls}
      <ParameterSlider label="Slow motion" unit=" ms/s" min={0.2} max={20} step={0.1} decimals={1} color="blue" value={settings.speed} onChange={(v) => update({ speed: v })} />

      <fieldset className="space-y-3 border-t border-slate-800 pt-4">
        <legend className="text-sm font-semibold text-slate-200">Measure</legend>
        {axon && (
          <>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
              <dt className="text-slate-500">Arrives at 1</dt>
              <dd data-testid="ax-t1">{ms(shown?.t1)}</dd>
              <dt className="text-slate-500">Arrives at 2</dt>
              <dd data-testid="ax-t2">{ms(shown?.t2)}</dd>
              <dt className="text-slate-500">Electrodes apart</dt>
              <dd>{shown ? lengthLabel(shown.distance) : "—"}</dd>
              <dt className="text-slate-500">Speed</dt>
              <dd data-testid="ax-velocity">{shown?.velocity !== undefined ? `${shown.velocity.toFixed(shown.velocity < 10 ? 2 : 1)} m/s` : "—"}</dd>
              <dt className="text-slate-500">Spike height at 2</dt>
              <dd data-testid="ax-peak">{shown?.peak2 !== undefined && shown.t2 !== undefined ? `${shown.peak2.toFixed(1)} mV` : "—"}</dd>
              <dt className="text-slate-500">Spikes at 1 / 2</dt>
              <dd data-testid="ax-spikes">{shown ? `${shown.spikes1} / ${shown.spikes2}` : "—"}</dd>
              <dt className="text-slate-500">Last sweep</dt>
              <dd data-testid="ax-outcome">{done ? (done.outcome === "conducted" ? "got through" : done.outcome === "blocked" ? "blocked" : "no spike") : "running…"}</dd>
            </dl>
            {stats?.view === "axon" && stats.traces.t.length > 1 && (
              <MiniChart
                title="Voltage at the electrodes"
                xLabel="time (ms)"
                yLabel="V (mV)"
                x={stats.traces.t}
                y={stats.traces.v1}
                y2={stats.traces.v2}
                labels={["electrode 1", "electrode 2"]}
                xMin={0}
                xMax={stats.sweepMs}
                yMin={-90}
                yMax={50}
                format={(v) => v.toFixed(0)}
                summary={shown?.velocity !== undefined ? `The spike reaches electrode 2 ${(shown.t2! - shown.t1!).toFixed(2)} ms after electrode 1.` : "No spike has passed both electrodes yet."}
              />
            )}
          </>
        )}
        {!axon && (
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
            <dt className="text-slate-500">A wave latency</dt>
            <dd data-testid="nv-a">{ms(cap?.aLatency)}</dd>
            <dt className="text-slate-500">→ A fibre speed</dt>
            <dd>{cap?.aLatency !== undefined ? `${((settings.distance / (cap.aLatency - 0.5)) * 10).toFixed(1)} m/s` : "—"}</dd>
            <dt className="text-slate-500">C wave latency</dt>
            <dd data-testid="nv-c">{ms(cap?.cLatency, 1)}</dd>
            <dt className="text-slate-500">→ C fibre speed</dt>
            <dd>{cap?.cLatency !== undefined ? `${((settings.distance / (cap.cLatency - 0.5)) * 10).toFixed(2)} m/s` : "—"}</dd>
          </dl>
        )}
      </fieldset>
    </>
  );

  const explanation = (
    <>
      <p>
        A nerve signal is not a current running down a wire. Each patch of membrane fires its own action potential
        (the Hodgkin–Huxley channels from the neuron simulation), and its local currents push the next patch past
        threshold: the spike is <em>regenerated</em> all the way.
      </p>
      <p>
        <strong className="text-slate-100">Axon.</strong> The tube shows the voltage along the axon now; the picture
        below is the whole sweep, position across and time downwards. A travelling spike draws a line: the steeper, the
        slower. Two recording electrodes measure its speed.
      </p>
      <p>
        <strong className="text-slate-100">Nerve.</strong> Hundreds of fibres of every size, each a full cable on the
        GPU, stimulated together. The electrode records their summed signal: the compound action potential of the
        classic frog-nerve lab.
      </p>
      <p className="text-slate-400">Squid-axon channels (Hodgkin & Huxley 1952): above about 30 °C they stop working (heat block).</p>
    </>
  );

  const summary = !stats
    ? undefined
    : stats.view === "axon"
      ? `Sweep ${stats.time.toFixed(1)} of ${stats.sweepMs} ms. ${shown?.velocity !== undefined ? `The spike travels at ${shown.velocity.toFixed(1)} m/s between the electrodes.` : done?.outcome === "blocked" ? "The spike was blocked." : "No spike has passed both electrodes yet."}`
      : `${stats.fibres} fibres, sweep ${stats.time.toFixed(1)} of ${stats.sweepMs} ms.${cap?.aLatency !== undefined ? ` A wave at ${cap.aLatency.toFixed(1)} ms` : ""}${cap?.cLatency !== undefined ? `, C wave at ${cap.cLatency.toFixed(1)} ms.` : ""}`;

  return (
    <SimLayout
      summary={summary}
      title="Axons & Nerves"
      subject="biology"
      difficulty="medium"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      challenges={
        <ChallengesPanel
          simId={SIM_ID}
          challenges={AXON_CHALLENGES}
          params={settings}
          readouts={readouts}
          onSetup={(patch) => {
            setStats(null);
            setPlaying(true);
            update(patch);
          }}
        />
      }
      canvasRef={canvasRef}
      onResetDefaults={() => setSettings(AXON_DEFAULTS)}
      playing={playing}
      onPlayPause={setPlaying}
      onStep={() => sim?.step()}
      onReset={() => sim?.reset()}
    >
      <canvas
        ref={canvasRef}
        className={`absolute inset-0 block h-full w-full touch-none ${drag ? "cursor-grabbing" : "cursor-ew-resize"}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      />
      <canvas ref={overlayRef} aria-hidden className="pointer-events-none absolute inset-0 h-full w-full" />
    </SimLayout>
  );
}
