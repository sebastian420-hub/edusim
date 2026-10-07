"use client";
import { useEffect, useMemo, useState } from "react";
import { ChallengesPanel } from "@/components/ChallengesPanel";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { usePersistedParams } from "@/lib/usePersistedParams";
import { usePlaying } from "@/lib/usePlaying";
import { boolField, enumField, numberField } from "@/lib/urlState";
import type { Schema } from "@/lib/urlState";
import { WAVE_CHALLENGES } from "./challenges";
import { createWaveSim } from "./sim";
import { DEFAULTS, detectorReadouts, MODES, VIEWS } from "./wave";
import type { WaveMode, WaveParams, WaveView } from "./wave";
import { WaveOverlay } from "./WaveOverlay";
import type { MeasureTool } from "./WaveOverlay";

const SIM_ID = "wave-interference";

const SCHEMA: Schema<WaveParams> = {
  frequency: numberField(0.5, 10),
  amplitude: numberField(0.1, 2),
  damping: numberField(0, 0.1),
  waveSpeed: numberField(0.1, 3),
  mode: enumField(MODES),
  separation: numberField(0.1, 1.5),
  slitWidth: numberField(0.05, 0.5),
  view: enumField(VIEWS),
  detectorX: numberField(-1, 3),
  detector: boolField,
};

const MODE_LABELS: { id: WaveMode; label: string }[] = [
  { id: "point", label: "Single Point" },
  { id: "two-points", label: "Two Points" },
  { id: "single-slit", label: "Single Slit" },
  { id: "double-slit", label: "Double Slit" },
];

const VIEW_LABELS: { id: WaveView; label: string }[] = [
  { id: "amplitude", label: "Amplitude" },
  { id: "intensity", label: "Intensity" },
  { id: "water", label: "3D Water" },
];

const TOOLS: { id: MeasureTool; label: string }[] = [
  { id: "none", label: "None" },
  { id: "probe", label: "Probe" },
  { id: "ruler", label: "Ruler" },
];

const pillClass = (active: boolean) =>
  `rounded px-2 py-1.5 text-sm transition-colors ${active ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"}`;

export default function WaveInterferenceControls() {
  const { canvasRef, status, sim, quality } = useGpuSim(createWaveSim);
  const [playing, setPlaying] = usePlaying(sim, true);
  const [params, setParams, resetParams] = usePersistedParams(SIM_ID, SCHEMA, DEFAULTS);
  const [tool, setTool] = useState<MeasureTool>("none");
  const readouts = useMemo(() => detectorReadouts(params), [params]);

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
          {MODE_LABELS.map((m) => (
            <button key={m.id} type="button" aria-pressed={params.mode === m.id} onClick={() => update("mode", m.id)} className={pillClass(params.mode === m.id)}>
              {m.label}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">View</legend>
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {VIEW_LABELS.map((v) => (
            <label key={v.id} className="flex items-center gap-1.5 text-sm text-slate-300">
              <input type="radio" name="wave-view" checked={params.view === v.id} onChange={() => update("view", v.id)} />
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

      <fieldset className="space-y-3 border-t border-slate-800 pt-4">
        <legend className="text-sm font-semibold text-slate-200">Measure</legend>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={params.detector} onChange={(e) => update("detector", e.target.checked)} />
          Detector screen (drag it on the canvas)
        </label>
        {params.detector && (
          <ParameterSlider label="Detector position" min={-1} max={3} step={0.05} value={params.detectorX} onChange={(v) => update("detectorX", v)} />
        )}
        <div className="grid grid-cols-3 gap-2" role="group" aria-label="Measuring tool">
          {TOOLS.map((t) => (
            <button key={t.id} type="button" aria-pressed={tool === t.id} onClick={() => setTool(t.id)} className={pillClass(tool === t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500">
          {tool === "probe" && "Click the canvas to read the intensity (I) and peak amplitude (|ψ|) at a point."}
          {tool === "ruler" && "Drag on the canvas to measure a distance in simulation units."}
          {tool === "none" && "Pick Probe or Ruler to measure on the canvas."}
        </p>
      </fieldset>
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
      <p>
        The <strong className="text-slate-100">detector</strong> (dashed line) plots intensity along a vertical
        screen. In double-slit mode the corner readout compares the fringe spacing it measures with the λL/d
        prediction.
      </p>
    </>
  );

  const MODE_NAMES: Record<WaveMode, string> = { point: "One point source", "two-points": "Two point sources", "single-slit": "Single slit", "double-slit": "Double slit" };
  const summary =
    `${MODE_NAMES[params.mode]}, ${params.view === "water" ? "3D water" : params.view} view.` +
    (readouts.measuredSpacing !== undefined
      ? ` Bright fringes on the detector screen are ${readouts.measuredSpacing.toFixed(2)} units apart${readouts.theorySpacing !== undefined ? ` (theory λL/d: ${readouts.theorySpacing.toFixed(2)})` : ""}.`
      : "");

  return (
    <SimLayout
      summary={summary}
      title="Wave Interference & Diffraction"
      subject="physics"
      difficulty="easy"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      challenges={
        <ChallengesPanel
          simId={SIM_ID}
          challenges={WAVE_CHALLENGES}
          params={params}
          readouts={readouts}
          onSetup={(patch) => setParams((prev) => ({ ...prev, ...patch }))}
        />
      }
      canvasRef={canvasRef}
      onResetDefaults={resetParams}
      playing={playing}
      onPlayPause={setPlaying}
      onReset={() => sim?.reset()}
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      <WaveOverlay
        params={params}
        readouts={readouts}
        tool={tool}
        onDetectorX={(x) => update("detectorX", x)}
      />
    </SimLayout>
  );
}
