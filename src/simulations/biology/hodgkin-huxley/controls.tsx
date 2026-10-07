"use client";
import { useDeferredValue, useEffect, useMemo, useState } from "react";
import type { PointerEvent } from "react";
import { ChallengesPanel } from "@/components/ChallengesPanel";
import { MiniChart } from "@/components/MiniChart";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { usePersistedParams } from "@/lib/usePersistedParams";
import { usePlaying } from "@/lib/usePlaying";
import { HH_CHALLENGES } from "./challenges";
import { fiAnalysis, PULSE_PERIOD, sampleAt, WINDOW_MS } from "./hh";
import { computeReadouts, HH_DEFAULTS, HH_SCHEMA, toParams } from "./settings";
import type { HHSettings } from "./settings";
import { createHodgkinHuxley } from "./sim";
import type { HHSnapshot } from "./sim";

const SIM_ID = "hodgkin-huxley";

const PRESETS: Record<string, { label: string; settings: Partial<HHSettings> }> = {
  normal: { label: "Normal AP", settings: { g_Na: 120, g_K: 36, I_inj: 10, pulse_mode: 1 } },
  ttx: { label: "TTX Block", settings: { g_Na: 0, g_K: 36, I_inj: 20, pulse_mode: 1 } },
  tea: { label: "TEA Block", settings: { g_Na: 120, g_K: 0, I_inj: 10, pulse_mode: 1 } },
  anode: { label: "Anode Break", settings: { g_Na: 120, g_K: 36, I_inj: -20, pulse_mode: 1 } },
};

const LEGEND = [
  // Okabe–Ito palette, as in render.wgsl: safe for colour-vision deficiencies.
  { color: "bg-[#56b4e9]", label: "Voltage (mV)" },
  { color: "bg-[#e69f00]", label: "m gate" },
  { color: "bg-[#cc79a7]", label: "h gate" },
  { color: "bg-[#f0e442]", label: "n gate" },
];

/** Voltage axis ticks, positioned like render.wgsl does: -100 mV (bottom) to +60 mV (top). */
const V_TICKS = [
  { mV: 60, label: "+60 mV" },
  { mV: 0, label: "0" },
  { mV: -65, label: "−65 rest" },
  { mV: -100, label: "−100" },
];
const vToPercent = (mV: number) => (1 - (mV + 100) / 160) * 100;

const STIMULI = [
  [0, "DC"],
  [1, "Pulse"],
  [2, "Twin"],
] as const;

export default function HHControls() {
  const { canvasRef, status, sim, quality } = useGpuSim(createHodgkinHuxley);
  const [playing, setPlaying] = usePlaying(sim, true);
  const [settings, setSettings, resetSettings] = usePersistedParams(SIM_ID, HH_SCHEMA, HH_DEFAULTS);
  const params = useMemo(() => toParams(settings), [settings]);

  useEffect(() => {
    sim?.setParams(settings);
  }, [sim, settings]);

  const update = (patch: Partial<HHSettings>) => setSettings((prev) => ({ ...prev, ...patch }));

  // The firing-rate curve depends only on conductances and temperature, and is a little expensive
  // (a few dozen model runs), so it follows the sliders at a lower priority.
  const curveInputs = useDeferredValue({ g_Na: settings.g_Na, g_K: settings.g_K, temperature: settings.temperature });
  const fi = useMemo(() => fiAnalysis(toParams({ ...HH_DEFAULTS, ...curveInputs, pulse_mode: 0 })), [curveInputs]);
  const readouts = useMemo(() => computeReadouts(settings, fi), [settings, fi]);

  // Cursor readout: while hovering the trace, poll the GPU history (a 32 KB read-back).
  const [hover, setHover] = useState<number | null>(null);
  const [snapshot, setSnapshot] = useState<HHSnapshot | null>(null);
  const hovering = hover !== null;
  useEffect(() => {
    if (!sim || !hovering) return;
    let cancelled = false;
    const poll = () => void sim.snapshot().then((s) => !cancelled && setSnapshot(s));
    poll();
    const id = setInterval(poll, 200);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [sim, hovering]);

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHover(Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)));
  };
  const cursor = hover !== null && snapshot ? sampleAt(snapshot.history, snapshot.head, hover) : null;

  const applyPreset = (key: string) => {
    update(PRESETS[key].settings);
    sim?.reset();
  };

  const dc = settings.pulse_mode === 0;
  const rateText = dc
    ? readouts.rate > 0
      ? `${readouts.rate.toFixed(0)} Hz`
      : "silent"
    : `${readouts.spikesPerCycle} spike${readouts.spikesPerCycle === 1 ? "" : "s"} per ${PULSE_PERIOD} ms cycle`;

  const controls = (
    <>
      <ParameterSlider label="Injected current" unit=" µA/cm²" min={-20} max={50} step={0.5} color="green" value={settings.I_inj} onChange={(v) => update({ I_inj: v })} />
      <ParameterSlider label="Temperature" unit=" °C" min={0} max={40} step={0.1} color="amber" value={settings.temperature} onChange={(v) => update({ temperature: v })} />
      <ParameterSlider label="Na⁺ conductance (TTX)" unit=" mS/cm²" min={0} max={200} step={1} color="red" value={settings.g_Na} onChange={(v) => update({ g_Na: v })} />
      <ParameterSlider label="K⁺ conductance (TEA)" unit=" mS/cm²" min={0} max={80} step={1} color="blue" value={settings.g_K} onChange={(v) => update({ g_K: v })} />
      <ParameterSlider label="Playback speed" unit=" ms/s" min={5} max={200} step={5} color="slate" value={settings.timeScale} onChange={(v) => update({ timeScale: v })} />

      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">Stimulus</legend>
        <div className="flex gap-4 text-sm text-slate-300">
          {STIMULI.map(([mode, label]) => (
            <label key={mode} className="flex items-center gap-1.5">
              <input type="radio" name="hh-stimulus" checked={settings.pulse_mode === mode} onChange={() => update({ pulse_mode: mode })} />
              {label}
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">Presets</legend>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(PRESETS).map(([key, preset]) => (
            <button key={key} type="button" onClick={() => applyPreset(key)} className="rounded bg-slate-800 py-1.5 text-sm text-slate-200 transition-colors hover:bg-slate-700">
              {preset.label}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset className="space-y-3 border-t border-slate-800 pt-4">
        <legend className="text-sm font-semibold text-slate-200">Measure</legend>
        <p className="text-xs text-slate-500">Hover the trace to read the exact voltage and gate values at any moment.</p>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={settings.showCurve} onChange={(e) => update({ showCurve: e.target.checked })} />
          Firing-rate curve
        </label>
        {settings.showCurve && (
          <>
            <MiniChart
              title="Firing rate vs injected current"
              xLabel="current (µA/cm²)"
              yLabel="rate (Hz)"
              x={fi.currents}
              y={fi.rates}
              markers={fi.rheobase !== undefined ? [{ x: fi.rheobase, label: `threshold ${fi.rheobase.toFixed(1)}` }] : []}
              point={dc && settings.I_inj >= 0 && settings.I_inj <= 30 ? { x: settings.I_inj, y: readouts.rate } : undefined}
              summary={
                fi.rheobase !== undefined
                  ? `The neuron starts firing repeatedly at about ${fi.rheobase.toFixed(1)} microamps per square centimetre.`
                  : "The neuron does not fire repeatedly at any current up to 30 microamps per square centimetre."
              }
            />
            <p className="text-xs text-slate-400">
              {fi.rheobase !== undefined
                ? `Starts firing at ≈ ${fi.rheobase.toFixed(1)} µA/cm². The dot marks your current setting.`
                : "No repetitive firing up to 30 µA/cm² with these channel settings."}
            </p>
          </>
        )}
      </fieldset>
    </>
  );

  const explanation = (
    <>
      <p>
        The Hodgkin–Huxley model describes how a neuron’s membrane voltage produces an action potential.
        Sodium channels (gates <em>m</em>, <em>h</em>) open quickly and depolarise the cell; potassium
        channels (gate <em>n</em>) open more slowly and repolarise it.
      </p>
      <p>
        <strong className="text-slate-100">Try it.</strong> Block sodium channels with the TTX preset and the
        spike disappears; block potassium (TEA) and the cell struggles to repolarise. Warmer temperatures
        speed up every gate.
      </p>
      <p className="text-slate-400">The trace shows the last {WINDOW_MS.toFixed(0)} ms of neuron time.</p>
    </>
  );

  const summary = dc
    ? readouts.rate > 0
      ? `The neuron fires steadily at ${readouts.rate.toFixed(0)} spikes per second.`
      : "The neuron is silent: no repeated spikes at this current."
    : `Pulse stimulus: ${readouts.spikesPerCycle} spike${readouts.spikesPerCycle === 1 ? "" : "s"} per ${PULSE_PERIOD} ms cycle.`;

  return (
    <SimLayout
      summary={summary}
      title="Hodgkin–Huxley Neuron"
      subject="biology"
      difficulty="medium"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      challenges={
        <ChallengesPanel
          simId={SIM_ID}
          challenges={HH_CHALLENGES}
          params={settings}
          readouts={readouts}
          onSetup={(patch) => {
            update(patch);
            sim?.reset();
          }}
        />
      }
      canvasRef={canvasRef}
      onResetDefaults={() => {
        resetSettings();
        sim?.reset();
      }}
      playing={playing}
      onPlayPause={setPlaying}
      onReset={() => sim?.reset()}
    >
      <div className="absolute inset-0" onPointerMove={onPointerMove} onPointerLeave={() => setHover(null)}>
        <canvas ref={canvasRef} className="block h-full w-full" />

        {/* Voltage axis */}
        {V_TICKS.map((t) => (
          <span
            key={t.mV}
            className="pointer-events-none absolute left-1.5 font-mono text-[10px] text-sky-300/80"
            style={{ top: `${vToPercent(t.mV)}%`, transform: t.mV === -100 ? "translateY(-110%)" : "translateY(-50%)" }}
          >
            {t.label}
          </span>
        ))}

        {/* Legend */}
        <div className="pointer-events-none absolute left-14 top-3 flex flex-wrap gap-x-4 gap-y-1 rounded bg-black/50 px-2 py-1 font-mono text-xs text-slate-200">
          {LEGEND.map((item) => (
            <div key={item.label} className="flex items-center gap-1.5">
              <span className={`h-3 w-3 rounded-full ${item.color}`} /> {item.label}
            </div>
          ))}
        </div>

        {/* Firing-rate readout */}
        <div className="pointer-events-none absolute right-3 top-3 rounded bg-black/60 px-2.5 py-1.5 font-mono text-[11px] leading-snug text-slate-200" aria-live="polite">
          <div className="text-slate-400">{dc ? "Steady firing rate (model)" : "Response to stimulus (model)"}</div>
          <div className="text-base font-bold text-sky-300" data-testid="hh-rate">
            {rateText}
          </div>
        </div>

        {/* Cursor readout */}
        {hover !== null && (
          <>
            <div className="pointer-events-none absolute inset-y-0 w-px bg-white/40" style={{ left: `${hover * 100}%` }} />
            {cursor && (
              <div
                data-testid="hh-cursor"
                className="pointer-events-none absolute top-1/2 -translate-y-1/2 rounded bg-black/75 px-2 py-1 font-mono text-[11px] leading-snug text-slate-100"
                style={hover > 0.7 ? { right: `${(1 - hover) * 100 + 1}%` } : { left: `${hover * 100 + 1}%` }}
              >
                <div className="text-slate-400">{cursor.msAgo.toFixed(1)} ms ago</div>
                <div className="text-[#56b4e9]">V = {cursor.V.toFixed(1)} mV</div>
                <div className="text-[#e69f00]">m = {cursor.m.toFixed(3)}</div>
                <div className="text-[#cc79a7]">h = {cursor.h.toFixed(3)}</div>
                <div className="text-[#f0e442]">n = {cursor.n.toFixed(3)}</div>
              </div>
            )}
          </>
        )}

        <div className="pointer-events-none absolute bottom-2 right-3 font-mono text-[10px] text-slate-400">
          window {WINDOW_MS.toFixed(0)} ms · stimulus: {params.pulse_mode === 0 ? "DC" : params.pulse_mode === 1 ? "pulse" : "twin pulse"}
        </div>
      </div>
    </SimLayout>
  );
}
