"use client";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent, WheelEvent } from "react";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { createCellularAutomata, GRID_SIZES } from "./sim";
import type { CellularAutomataHandle, CellularAutomataParams } from "./sim";
import { DEFAULT_PATTERN, patterns } from "./patterns";
import { RULE_PRESETS } from "./rules";

type Tool = "pan" | "draw" | "erase";

const TOOLS: { id: Tool; label: string }[] = [
  { id: "pan", label: "Pan" },
  { id: "draw", label: "Draw" },
  { id: "erase", label: "Erase" },
];

const selectClass = "rounded border border-slate-700 bg-slate-800 p-2 text-sm text-slate-200";

export default function CellularAutomataControls() {
  const [generation, setGeneration] = useState(0);
  const [zoom, setZoom] = useState(1);
  // The factory must be stable, so it is created once and closes over the React state setters.
  const [factory] = useState(
    () => (ctx: Parameters<typeof createCellularAutomata>[0]) =>
      createCellularAutomata(ctx, { onGeneration: setGeneration, onZoom: setZoom }),
  );
  const { canvasRef, status, sim } = useGpuSim<CellularAutomataHandle>(factory);

  const [params, setParams] = useState<Pick<CellularAutomataParams, "birth" | "survive" | "speed" | "theme">>({
    birth: "3",
    survive: "23",
    speed: 10,
    theme: 0,
  });
  const [gridSize, setGridSize] = useState<number>(256);
  const [tool, setTool] = useState<Tool>("pan");
  const dragging = useRef(false);
  const lastPointer = useRef({ x: 0, y: 0 });

  useEffect(() => {
    sim?.setParams(params);
  }, [sim, params]);

  useEffect(() => {
    sim?.setGridSize(gridSize);
  }, [sim, gridSize]);

  const update = (patch: Partial<typeof params>) => setParams((prev) => ({ ...prev, ...patch }));

  const local = (e: PointerEvent | WheelEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!sim) return;
    dragging.current = true;
    lastPointer.current = { x: e.clientX, y: e.clientY };
    e.currentTarget.setPointerCapture(e.pointerId);
    if (tool !== "pan") {
      const p = local(e);
      sim.paint(p.x, p.y, tool === "draw");
    }
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!sim || !dragging.current) return;
    if (tool === "pan") {
      sim.panBy(e.clientX - lastPointer.current.x, e.clientY - lastPointer.current.y);
      lastPointer.current = { x: e.clientX, y: e.clientY };
    } else {
      const p = local(e);
      sim.paint(p.x, p.y, tool === "draw");
    }
  };

  const endDrag = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!dragging.current) return;
    dragging.current = false;
    sim?.endStroke();
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const onWheel = (e: WheelEvent<HTMLCanvasElement>) => {
    const p = local(e);
    sim?.zoomAt(e.deltaY > 0 ? 0.9 : 1.1, p.x, p.y);
  };

  const controls = (
    <>
      <div className="flex items-center justify-between rounded-lg border border-slate-700/50 bg-slate-800/50 p-3">
        <span className="text-sm text-slate-400">Generation</span>
        <span className="font-mono text-xl text-green-400">{generation}</span>
      </div>

      <ParameterSlider label="Speed" unit=" gen/s" min={1} max={60} step={1} color="green" value={params.speed} onChange={(v) => update({ speed: v })} />

      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Grid size</span>
        <select value={gridSize} onChange={(e) => setGridSize(Number(e.target.value))} className={selectClass}>
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
                  value={params[key]}
                  onChange={(e) => update({ [key]: e.target.value.replace(/[^0-8]/g, "") })}
                  className="w-full rounded border border-slate-700 bg-slate-800 p-1.5 font-mono text-sm text-slate-200"
                />
              </span>
            </label>
          ))}
        </div>
        <select
          aria-label="Rule preset"
          value=""
          onChange={(e) => {
            const preset = RULE_PRESETS.find((r) => r.name === e.target.value);
            if (preset) update({ birth: preset.birth, survive: preset.survive });
          }}
          className={`mt-2 w-full ${selectClass}`}
        >
          <option value="">Rule presets…</option>
          {RULE_PRESETS.map((r) => (
            <option key={r.name} value={r.name}>
              {r.name} (B{r.birth}/S{r.survive})
            </option>
          ))}
        </select>
      </fieldset>

      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Pattern library</span>
        <select value="" onChange={(e) => e.target.value && sim?.loadPattern(e.target.value)} className={selectClass}>
          <option value="">Load pattern…</option>
          {patterns.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </select>
      </label>

      <div className="grid grid-cols-2 gap-2">
        <button type="button" onClick={() => sim?.clear()} className="rounded bg-red-900/50 px-3 py-2 text-sm font-medium text-red-200 transition-colors hover:bg-red-800/50">
          Clear
        </button>
        <button type="button" onClick={() => sim?.randomize()} className="rounded bg-slate-700 px-3 py-2 text-sm font-medium text-slate-100 transition-colors hover:bg-slate-600">
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

      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Color theme</span>
        <select value={params.theme} onChange={(e) => update({ theme: Number(e.target.value) })} className={selectClass}>
          <option value={0}>Classic Green</option>
          <option value={1}>Cyberpunk Neon</option>
          <option value={2}>Minimal White</option>
        </select>
      </label>
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
      <p className="text-slate-400">
        Pick the Draw or Erase tool to edit cells. Scroll to zoom at the cursor. Default pattern: {DEFAULT_PATTERN}.
      </p>
    </>
  );

  return (
    <SimLayout
      title="Cellular Automata"
      subject="cs"
      difficulty="easy"
      status={status}
      controls={controls}
      explanation={explanation}
      initiallyPlaying={false}
      onPlayPause={(playing) => (playing ? sim?.play() : sim?.pause())}
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
      <div className="pointer-events-none absolute right-3 top-3 flex gap-4 rounded border border-slate-800 bg-slate-900/80 px-3 py-1.5 text-xs text-slate-400 backdrop-blur">
        <span>Zoom: {zoom.toFixed(2)}×</span>
        <span>{tool === "pan" ? "Drag to pan · scroll to zoom" : tool === "draw" ? "Drag to draw cells" : "Drag to erase cells"}</span>
      </div>
    </SimLayout>
  );
}
