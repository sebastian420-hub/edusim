"use client";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent, WheelEvent } from "react";
import { ChallengesPanel } from "@/components/ChallengesPanel";
import { MiniChart } from "@/components/MiniChart";
import { ParameterSlider } from "@/components/ParameterSlider";
import { createPinchTracker } from "@/lib/gestures";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { usePersistedParams } from "@/lib/usePersistedParams";
import { usePlaying } from "@/lib/usePlaying";
import { CA_CHALLENGES } from "./challenges";
import { niceMax } from "./graph";
import { patterns } from "./patterns";
import { RULE_PRESETS } from "./rules";
import { CA_DEFAULTS, CA_SCHEMA, sanitizeRuleDigits, THEMES } from "./settings";
import type { CASettings } from "./settings";
import { createCellularAutomata } from "./sim";
import type { CellularAutomataHandle, CellularAutomataStats } from "./sim";
import { GRID_SIZES } from "./sim-constants";
import { describeStatus } from "./tracker";
import type { PatternStatus } from "./tracker";

const SIM_ID = "cellular-automata";

type Tool = "pan" | "draw" | "erase";

const TOOLS: { id: Tool; label: string }[] = [
  { id: "pan", label: "Pan" },
  { id: "draw", label: "Draw" },
  { id: "erase", label: "Erase" },
];

const THEME_NAMES = ["Classic Green", "Cyberpunk Neon", "Minimal White", "Amber"];

const selectClass = "rounded border border-slate-700 bg-slate-800 p-2 text-sm text-slate-200";

/** What challenges see while there is no measurement yet (just after a restart): nothing can be solved by it. */
const NO_STATS: CellularAutomataStats = { generation: 0, population: 0, cells: 0, gens: [], counts: [], status: { kind: "evolving" } };

const STATUS_TONE: Record<PatternStatus["kind"], string> = {
  empty: "text-red-300",
  still: "text-green-300",
  oscillator: "text-green-300",
  moving: "text-green-300",
  evolving: "text-slate-300",
};

/** Puts the simulation into the starting state that a `pattern` setting names. */
function startFrom(sim: CellularAutomataHandle, pattern: string) {
  if (pattern === "random") sim.randomize();
  else if (pattern === "clear") sim.clear();
  else sim.loadPattern(pattern);
}

export default function CellularAutomataControls() {
  const [generation, setGeneration] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [stats, setStats] = useState<CellularAutomataStats | null>(null);
  // The factory must be stable, so it is created once and closes over the React state setters.
  const [factory] = useState(
    () => (ctx: Parameters<typeof createCellularAutomata>[0]) =>
      createCellularAutomata(ctx, { onGeneration: setGeneration, onZoom: setZoom, onStats: setStats }),
  );
  const { canvasRef, status, sim, quality } = useGpuSim<CellularAutomataHandle>(factory);
  // Two fingers pan and pinch-zoom whatever the tool; one finger uses the selected tool.
  const [pinch] = useState(createPinchTracker);

  const [settings, setSettings] = usePersistedParams(SIM_ID, CA_SCHEMA, CA_DEFAULTS);
  const [tool, setTool] = useState<Tool>("pan");
  const [playing, setPlaying] = usePlaying(sim, false);
  const dragging = useRef(false);
  const lastPointer = useRef({ x: 0, y: 0 });

  useEffect(() => {
    sim?.setParams({ birth: settings.birth, survive: settings.survive, speed: settings.speed, theme: settings.theme });
  }, [sim, settings.birth, settings.survive, settings.speed, settings.theme]);

  useEffect(() => {
    sim?.setGridSize(settings.gridSize);
  }, [sim, settings.gridSize]);

  useEffect(() => {
    if (sim) startFrom(sim, settings.pattern);
  }, [sim, settings.pattern]);


  const update = (patch: Partial<CASettings>) => setSettings((prev) => ({ ...prev, ...patch }));

  /**
   * Changes settings. A new starting pattern reaches the simulation through the effect above, but choosing the
   * pattern that is already selected changes no state, so that case restarts it by hand.
   */
  const apply = (patch: Partial<CASettings>) => {
    update(patch);
    if (sim && patch.pattern !== undefined && patch.pattern === settings.pattern) startFrom(sim, patch.pattern);
  };

  const local = (e: PointerEvent | WheelEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  // A touch that may become a pinch: drawing waits for the first movement (or a tap) so a second finger can still cancel it.
  const pendingTouch = useRef<{ x: number; y: number } | null>(null);

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!sim) return;
    const p = local(e);
    e.currentTarget.setPointerCapture(e.pointerId);
    pinch.down(e.pointerId, p.x, p.y);
    if (pinch.active) {
      // A second finger: stop drawing, this is a pinch.
      if (dragging.current && !pendingTouch.current) sim.endStroke();
      pendingTouch.current = null;
      dragging.current = false;
      return;
    }
    dragging.current = true;
    lastPointer.current = { x: e.clientX, y: e.clientY };
    if (tool === "pan") return;
    if (e.pointerType === "touch") pendingTouch.current = p;
    else sim.paint(p.x, p.y, tool === "draw");
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const p = local(e);
    const gesture = pinch.move(e.pointerId, p.x, p.y);
    if (gesture) {
      sim?.panBy(gesture.dx, gesture.dy);
      sim?.zoomAt(gesture.factor, gesture.x, gesture.y);
      return;
    }
    if (!sim || !dragging.current) return;
    if (tool === "pan") {
      sim.panBy(e.clientX - lastPointer.current.x, e.clientY - lastPointer.current.y);
      lastPointer.current = { x: e.clientX, y: e.clientY };
      return;
    }
    const start = pendingTouch.current;
    pendingTouch.current = null;
    if (start) sim.paint(start.x, start.y, tool === "draw");
    sim.paint(p.x, p.y, tool === "draw");
  };

  const endDrag = (e: PointerEvent<HTMLCanvasElement>) => {
    pinch.up(e.pointerId);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    if (!dragging.current) return;
    dragging.current = false;
    const tap = pendingTouch.current;
    pendingTouch.current = null;
    if (tap && e.type === "pointerup") sim?.paint(tap.x, tap.y, tool === "draw"); // a tap draws one cell
    sim?.endStroke();
  };

  const onWheel = (e: WheelEvent<HTMLCanvasElement>) => {
    const p = local(e);
    sim?.zoomAt(e.deltaY > 0 ? 0.9 : 1.1, p.x, p.y);
  };

  const presetName =
    RULE_PRESETS.find(
      (r) => sanitizeRuleDigits(r.birth) === sanitizeRuleDigits(settings.birth) && sanitizeRuleDigits(r.survive) === sanitizeRuleDigits(settings.survive),
    )?.name ?? "";

  const series = stats && stats.counts.length > 1 ? stats : null;

  const controls = (
    <>
      <div className="flex items-center justify-between rounded-lg border border-slate-700/50 bg-slate-800/50 p-3">
        <span className="text-sm text-slate-400">Generation</span>
        <span className="font-mono text-xl text-green-400">{generation}</span>
      </div>

      <ParameterSlider label="Speed" unit=" gen/s" min={1} max={60} step={1} color="green" value={settings.speed} onChange={(v) => update({ speed: v })} />

      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Grid size</span>
        <select
          value={settings.gridSize}
          onChange={(e) => {
            const size = GRID_SIZES.find((s) => String(s) === e.target.value);
            if (size) update({ gridSize: size });
          }}
          className={selectClass}
        >
          {GRID_SIZES.map((s) => (
            <option key={s} value={s}>
              {s} × {s}
            </option>
          ))}
        </select>
      </label>

      <fieldset>
        <legend className="mb-2 text-sm font-medium text-slate-200">Rules (B/S)</legend>
        <div className="flex gap-3">
          {(["birth", "survive"] as const).map((key) => (
            <label key={key} className="flex flex-1 flex-col gap-1 text-xs text-slate-400">
              {key === "birth" ? "Birth" : "Survive"}
              <span className="flex items-center gap-2">
                <span className="text-slate-500">{key === "birth" ? "B" : "S"}</span>
                <input
                  type="text"
                  inputMode="numeric"
                  value={settings[key]}
                  onChange={(e) => update({ [key]: e.target.value.replace(/[^0-8]/g, "") })}
                  className="w-full rounded border border-slate-700 bg-slate-800 p-1.5 font-mono text-sm text-slate-200"
                />
              </span>
            </label>
          ))}
        </div>
        <select
          aria-label="Rule preset"
          value={presetName}
          onChange={(e) => {
            const preset = RULE_PRESETS.find((r) => r.name === e.target.value);
            if (preset) update({ birth: preset.birth, survive: preset.survive });
          }}
          className={`mt-2 w-full ${selectClass}`}
        >
          <option value="">Custom rule</option>
          {RULE_PRESETS.map((r) => (
            <option key={r.name} value={r.name}>
              {r.name} (B{r.birth}/S{r.survive})
            </option>
          ))}
        </select>
      </fieldset>

      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Starting pattern</span>
        <select value={settings.pattern} onChange={(e) => apply({ pattern: e.target.value })} className={selectClass}>
          <optgroup label="Pattern library">
            {patterns.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}
              </option>
            ))}
          </optgroup>
          <option value="random">Random soup</option>
          <option value="clear">Empty grid</option>
        </select>
        <span className="text-xs text-slate-500">Reset (R) starts over from it.</span>
      </label>

      <div className="grid grid-cols-2 gap-2">
        <button type="button" onClick={() => apply({ pattern: "clear" })} className="rounded bg-red-900/50 px-3 py-2 text-sm font-medium text-red-200 transition-colors hover:bg-red-800/50">
          Clear
        </button>
        <button type="button" onClick={() => apply({ pattern: "random" })} className="rounded bg-slate-700 px-3 py-2 text-sm font-medium text-slate-100 transition-colors hover:bg-slate-600">
          Random
        </button>
      </div>

      <fieldset>
        <legend className="mb-2 text-sm font-medium text-slate-200">Mouse tool</legend>
        <div className="grid grid-cols-3 gap-2">
          {TOOLS.map((t) => (
            <button
              key={t.id}
              type="button"
              aria-pressed={tool === t.id}
              onClick={() => setTool(t.id)}
              className={`rounded px-2 py-1.5 text-sm transition-colors ${
                tool === t.id ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </fieldset>

      <div className="grid grid-cols-2 gap-2">
        <button type="button" onClick={() => void sim?.fitPattern()} className="rounded bg-slate-800 px-2 py-1.5 text-sm text-slate-300 transition-colors hover:bg-slate-700">
          Fit pattern
        </button>
        <button type="button" onClick={() => sim?.fitGrid()} className="rounded bg-slate-800 px-2 py-1.5 text-sm text-slate-300 transition-colors hover:bg-slate-700">
          Fit grid
        </button>
      </div>

      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Color theme</span>
        <select
          value={settings.theme}
          onChange={(e) => {
            const theme = THEMES.find((t) => String(t) === e.target.value);
            if (theme !== undefined) update({ theme });
          }}
          className={selectClass}
        >
          {THEMES.map((t) => (
            <option key={t} value={t}>
              {THEME_NAMES[t]}
            </option>
          ))}
        </select>
      </label>

      <fieldset className="space-y-3 border-t border-slate-800 pt-4">
        <legend className="text-sm font-semibold text-slate-200">Measure</legend>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={settings.showGraph} onChange={(e) => update({ showGraph: e.target.checked })} />
          Population graph
        </label>
        {settings.showGraph &&
          (series ? (
            <>
              <MiniChart
                title="Live cells per generation"
                xLabel="generation"
                yLabel="live cells"
                x={series.gens}
                y={series.counts}
                yMax={niceMax(Math.max(...series.counts))}
                point={{ x: series.gens[series.gens.length - 1], y: series.counts[series.counts.length - 1] }}
                summary={`${series.population} live cells now. ${describeStatus(series.status)}.`}
              />
              <p className="text-xs text-slate-400">The last {series.gens.length} generations. The dot marks now.</p>
            </>
          ) : (
            <p className="text-xs text-slate-500">Press Play or Step to start recording the population.</p>
          ))}
      </fieldset>
    </>
  );

  const explanation = (
    <>
      <p>
        A cellular automaton is a grid of cells that live or die according to a simple local rule. In{" "}
        <strong className="text-slate-100">Conway’s Game of Life</strong> (B3/S23) a dead cell with exactly 3
        living neighbours is born, and a living cell with 2 or 3 neighbours survives.
      </p>
      <p>
        Every cell updates at once, which is why this runs well on the GPU: one compute shader invocation per
        cell, millions of cells per generation. The grid wraps around like a torus.
      </p>
      <p>
        <strong className="text-slate-100">Measure it.</strong> The GPU also counts the live cells every
        generation, and the panel on the canvas names what it sees: a still life, an oscillator and its period,
        or a moving pattern such as a glider.
      </p>
      <p className="text-slate-400">
        Pick the Draw or Erase tool to edit cells. Scroll to zoom at the cursor. Copy link shares the rules,
        grid, speed, colours and starting pattern — not cells you draw by hand.
      </p>
    </>
  );

  const summary = stats
    ? `Generation ${stats.generation}: ${stats.population.toLocaleString("en-US")} live cells. ${describeStatus(stats.status)}.`
    : `Generation ${generation}.`;

  return (
    <SimLayout
      summary={summary}
      title="Cellular Automata"
      subject="cs"
      difficulty="easy"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      challenges={
        <ChallengesPanel
          simId={SIM_ID}
          challenges={CA_CHALLENGES}
          params={settings}
          readouts={stats ?? NO_STATS}
          onSetup={(patch) => {
            setStats(null); // measurements of the previous experiment must not count towards this one
            setPlaying(false); // start paused: arrange the cells, then press Play
            if (patch.pattern === "clear") setTool("draw");
            apply(patch);
          }}
        />
      }
      canvasRef={canvasRef}
      onResetDefaults={() => {
        setTool("pan");
        setPlaying(false);
        apply(CA_DEFAULTS);
      }}
      playing={playing}
      onPlayPause={setPlaying}
      onStep={() => sim?.step()}
      onReset={() => sim?.reset()}
    >
      <canvas
        ref={canvasRef}
        className={`absolute inset-0 block h-full w-full touch-none ${tool === "pan" ? "cursor-grab" : "cursor-crosshair"}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onWheel={onWheel}
      />

      {/* Population and what the history says about the pattern, measured on the GPU */}
      <div className="pointer-events-none absolute left-3 top-3 max-w-[55%] rounded bg-black/60 px-2.5 py-1.5 font-mono text-[11px] leading-snug text-slate-200">
        <div className="text-slate-400">Population</div>
        <div className="text-base font-bold text-green-300" data-testid="ca-population">
          {stats ? stats.population.toLocaleString("en-US") : "—"}
        </div>
        <div aria-live="polite" data-testid="ca-status" className={stats ? STATUS_TONE[stats.status.kind] : "text-slate-400"}>
          {stats ? describeStatus(stats.status) : "Measuring…"}
        </div>
        {stats?.status.kind === "empty" && <div className="text-slate-500">Draw cells or pick a pattern.</div>}
      </div>

      <div className="pointer-events-none absolute right-3 top-3 flex gap-4 rounded border border-slate-800 bg-slate-900/80 px-3 py-1.5 text-xs text-slate-400 backdrop-blur">
        <span>Zoom: {zoom.toFixed(2)}×</span>
        <span className="hidden sm:inline">{tool === "pan" ? "Drag to pan · scroll to zoom" : tool === "draw" ? "Drag to draw cells" : "Drag to erase cells"}</span>
      </div>
    </SimLayout>
  );
}
