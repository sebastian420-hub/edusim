"use client";
import { useEffect, useState } from "react";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { createWaveSim, DEFAULTS } from "./sim";
import type { WaveMode, WaveParams, WaveView } from "./sim";

const MODES: { id: WaveMode; label: string }[] = [
  { id: "point", label: "Single Point" },
  { id: "two-points", label: "Two Points" },
  { id: "single-slit", label: "Single Slit" },
  { id: "double-slit", label: "Double Slit" },
];

const VIEWS: { id: WaveView; label: string }[] = [
  { id: "amplitude", label: "Amplitude" },
  { id: "intensity", label: "Intensity" },
  { id: "water", label: "3D Water" },
];

export default function WaveInterferenceControls() {
  const { canvasRef, status, sim } = useGpuSim(createWaveSim);
  const [params, setParams] = useState<WaveParams>(DEFAULTS);

  useEffect(() => {
    sim?.setParams(params);
  }, [sim, params]);

  const update = <K extends keyof WaveParams>(key: K, value: WaveParams[K]) =>
    setParams((prev) => ({ ...prev, [key]: value }));

  const hasSeparation = params.mode === "two-points" || params.mode === "double-slit";
  const hasSlits = params.mode === "single-slit" || params.mode === "double-slit";

  const controls = (
    <>
      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">Source</legend>
        <div className="grid grid-cols-2 gap-2">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              aria-pressed={params.mode === m.id}
              onClick={() => update("mode", m.id)}
              className={`rounded px-2 py-1.5 text-sm transition-colors ${
                params.mode === m.id ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">View</legend>
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {VIEWS.map((v) => (
            <label key={v.id} className="flex items-center gap-1.5 text-sm text-slate-300">
              <input
                type="radio"
                name="wave-view"
                checked={params.view === v.id}
                onChange={() => update("view", v.id)}
              />
              {v.label}
            </label>
          ))}
        </div>
      </fieldset>

      <ParameterSlider label="Frequency" min={0.5} max={10} step={0.1} value={params.frequency} onChange={(v) => update("frequency", v)} />
      <ParameterSlider label="Amplitude" min={0.1} max={2} step={0.1} value={params.amplitude} onChange={(v) => update("amplitude", v)} />
      <ParameterSlider label="Wave Speed" min={0.1} max={3} step={0.1} value={params.waveSpeed} onChange={(v) => update("waveSpeed", v)} />
      <ParameterSlider label="Damping" min={0} max={0.1} step={0.01} value={params.damping} onChange={(v) => update("damping", v)} />
      {hasSlits && (
        <ParameterSlider label="Slit Width" min={0.05} max={0.5} step={0.05} value={params.slitWidth} onChange={(v) => update("slitWidth", v)} />
      )}
      {hasSeparation && (
        <ParameterSlider
          label={params.mode === "double-slit" ? "Slit Separation" : "Source Separation"}
          min={0.1}
          max={1.5}
          step={0.1}
          value={params.separation}
          onChange={(v) => update("separation", v)}
        />
      )}
    </>
  );

  const explanation = (
    <>
      <p>
        Waves from different sources add together. Where crests meet crests the waves reinforce
        (constructive interference); where a crest meets a trough they cancel (destructive).
      </p>
      <p>
        <strong className="text-slate-100">Slits.</strong> By the Huygens–Fresnel principle every point in a
        slit acts as a new source. A narrower slit spreads the wave out more (diffraction); two slits produce
        fringes spaced by roughly <em>λL / d</em>.
      </p>
      <p>
        Switch to <strong className="text-slate-100">Intensity</strong> to see the time-averaged pattern — the
        bright and dark fringes stay fixed even though the waves keep moving.
      </p>
    </>
  );

  return (
    <SimLayout
      title="Wave Interference & Diffraction"
      subject="physics"
      difficulty="easy"
      status={status}
      controls={controls}
      explanation={explanation}
      onPlayPause={(playing) => (playing ? sim?.play() : sim?.pause())}
      onReset={() => sim?.reset()}
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
    </SimLayout>
  );
}
