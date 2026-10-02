"use client";
import { useEffect, useState } from "react";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { WINDOW_MS } from "./hh";
import { createHodgkinHuxley, DEFAULT_SIM_PARAMS } from "./sim";
import type { HHSimParams } from "./sim";

const PRESETS: Record<string, { label: string; params: Partial<HHSimParams> }> = {
  normal: { label: "Normal AP", params: { g_Na: 120, g_K: 36, I_inj: 10, pulse_mode: 1 } },
  ttx: { label: "TTX Block", params: { g_Na: 0, g_K: 36, I_inj: 20, pulse_mode: 1 } },
  tea: { label: "TEA Block", params: { g_Na: 120, g_K: 0, I_inj: 10, pulse_mode: 1 } },
  anode: { label: "Anode Break", params: { g_Na: 120, g_K: 36, I_inj: -20, pulse_mode: 1 } },
};

const LEGEND = [
  { color: "bg-[#33ff66]", label: "Voltage (mV)" },
  { color: "bg-[#ff3333]", label: "m gate" },
  { color: "bg-[#3366ff]", label: "h gate" },
  { color: "bg-[#ffcc33]", label: "n gate" },
];

export default function HHControls() {
  const { canvasRef, status, sim, quality } = useGpuSim(createHodgkinHuxley);
  const [params, setParams] = useState<HHSimParams>(DEFAULT_SIM_PARAMS);

  useEffect(() => {
    sim?.setParams(params);
  }, [sim, params]);

  const update = (patch: Partial<HHSimParams>) => setParams((prev) => ({ ...prev, ...patch }));

  const applyPreset = (key: keyof typeof PRESETS) => {
    update(PRESETS[key].params);
    sim?.reset();
  };

  const controls = (
    <>
      <ParameterSlider label="Injected current" unit=" µA/cm²" min={-20} max={50} step={0.5} color="green" value={params.I_inj} onChange={(v) => update({ I_inj: v })} />
      <ParameterSlider label="Temperature" unit=" °C" min={0} max={40} step={0.1} color="amber" value={params.temperature} onChange={(v) => update({ temperature: v })} />
      <ParameterSlider label="Na⁺ conductance (TTX)" unit=" mS/cm²" min={0} max={200} step={1} color="red" value={params.g_Na} onChange={(v) => update({ g_Na: v })} />
      <ParameterSlider label="K⁺ conductance (TEA)" unit=" mS/cm²" min={0} max={80} step={1} color="blue" value={params.g_K} onChange={(v) => update({ g_K: v })} />
      <ParameterSlider label="Playback speed" unit=" ms/s" min={5} max={200} step={5} color="slate" value={params.timeScale} onChange={(v) => update({ timeScale: v })} />

      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">Stimulus</legend>
        <div className="flex gap-4 text-sm text-slate-300">
          {([
            [0, "DC"],
            [1, "Pulse"],
            [2, "Twin"],
          ] as const).map(([mode, label]) => (
            <label key={mode} className="flex items-center gap-1.5">
              <input type="radio" name="hh-stimulus" checked={params.pulse_mode === mode} onChange={() => update({ pulse_mode: mode })} />
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

  return (
    <SimLayout
      title="Hodgkin–Huxley Neuron"
      subject="biology"
      difficulty="medium"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      onPlayPause={(playing) => (playing ? sim?.play() : sim?.pause())}
      onReset={() => sim?.reset()}
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-x-4 gap-y-1 rounded bg-black/50 px-2 py-1 font-mono text-xs text-slate-200">
        {LEGEND.map((item) => (
          <div key={item.label} className="flex items-center gap-1.5">
            <span className={`h-3 w-3 rounded-full ${item.color}`} /> {item.label}
          </div>
        ))}
      </div>
      <div className="pointer-events-none absolute bottom-2 left-3 font-mono text-[10px] text-slate-400">
        voltage axis: −100 mV (bottom) to +60 mV (top) · window {WINDOW_MS.toFixed(0)} ms
      </div>
    </SimLayout>
  );
}
