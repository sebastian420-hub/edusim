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
import { NBODY_CHALLENGES, readoutsFor } from "./challenges";
import { circularSpeed, primaryIndex } from "./nbody";
import type { Body } from "./nbody";
import { arrowYears, drawOverlay, hitTest } from "./overlay";
import type { OverlayState } from "./overlay";
import { ORBIT_PRESETS } from "./presets";
import { bodyNames, encodeBodies, GALAXY_COUNTS, MAX_ORBIT_BODIES, NBODY_DEFAULTS, NBODY_SCHEMA, setupBodies } from "./settings";
import type { GalaxyPreset, NBodySettings } from "./settings";
import { createNBody, orbitLook } from "./sim";
import type { FrameInfo, NBodyHandle, NBodyStats } from "./sim";

const SIM_ID = "n-body";

const MASSES = [
  { m: 1e-12, label: "Asteroid (10⁻¹² M☉)" },
  { m: 3.003e-6, label: "Earth (3 × 10⁻⁶ M☉)" },
  { m: 9.546e-4, label: "Jupiter (10⁻³ M☉)" },
  { m: 0.1, label: "Red dwarf (0.1 M☉)" },
  { m: 0.5, label: "Small star (0.5 M☉)" },
  { m: 1, label: "Sun (1 M☉)" },
];

const GALAXY_PRESETS: { id: GalaxyPreset; label: string }[] = [
  { id: "collision", label: "Two colliding galaxies" },
  { id: "disk", label: "Rotating disk galaxy" },
  { id: "cluster", label: "Star cluster (Plummer)" },
];

const selectClass = "rounded border border-slate-700 bg-slate-800 p-2 text-sm text-slate-200";
const toggleClass = (on: boolean) =>
  `rounded px-2 py-1.5 text-sm transition-colors ${on ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"}`;

const rgb = ([r, g, b]: number[]) => `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
const fixed = (v: number, digits: number) => (Number.isFinite(v) ? v.toFixed(digits) : "∞");
const signedPercent = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toPrecision(2)} %`;

/** Moves body `i` to distance `r` from the primary (same direction), or gives it speed `v` (same direction). */
function withDistance(bodies: Body[], i: number, r: number): Body[] {
  const p = bodies[primaryIndex(bodies)];
  const b = bodies[i];
  const dx = b.x - p.x;
  const dy = b.y - p.y;
  const d = Math.hypot(dx, dy) || 1;
  return bodies.map((x, k) => (k === i ? { ...b, x: p.x + (dx / d) * r, y: p.y + (dy / d) * r } : x));
}
function withSpeed(bodies: Body[], i: number, v: number): Body[] {
  const p = bodies[primaryIndex(bodies)];
  const b = bodies[i];
  let ux = b.vx - p.vx;
  let uy = b.vy - p.vy;
  let u = Math.hypot(ux, uy);
  if (u === 0) {
    // At rest: launch it counter-clockwise, perpendicular to the line to the primary.
    const dx = b.x - p.x;
    const dy = b.y - p.y;
    const d = Math.hypot(dx, dy) || 1;
    ux = -dy / d;
    uy = dx / d;
    u = 1;
  }
  return bodies.map((x, k) => (k === i ? { ...b, vx: p.vx + (ux / u) * v, vy: p.vy + (uy / u) * v } : x));
}

export default function NBodyControls() {
  const [stats, setStats] = useState<NBodyStats | null>(null);
  const frameRef = useRef<FrameInfo | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const overlayState = useRef<OverlayState | null>(null);
  const looks = useRef<{ names: string[]; colors: string[]; selected: number | null; arrows: boolean; arrowTime: number }>({ names: [], colors: [], selected: null, arrows: true, arrowTime: 0.05 });

  // The factory must be stable: created once, it closes over a ref that draws the overlay every frame.
  const [factory] = useState(
    () => (ctx: Parameters<typeof createNBody>[0]) =>
      createNBody(ctx, {
        onStats: setStats,
        onFrame: (info) => {
          frameRef.current = info;
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
          const state: OverlayState = { bodies: info.bodies, view: info.view, pixelRatio: info.pixelRatio, width, height, ...looks.current };
          overlayState.current = state;
          drawOverlay(ctx2d, state);
        },
      }),
  );
  const { canvasRef, status, sim, quality } = useGpuSim<NBodyHandle>(factory);
  // Two fingers pan and pinch-zoom; one finger keeps dragging bodies, arrows or the view.
  const [pinch] = useState(createPinchTracker);
  const [settings, setSettings] = usePersistedParams(SIM_ID, NBODY_SCHEMA, NBODY_DEFAULTS);
  const [playing, setPlaying] = useState(false);
  const [selected, setSelected] = useState<number | null>(1);
  const [arrows, setArrows] = useState(true);
  const drag = useRef<{ kind: "body" | "arrow" | "pan"; index: number; base: Body[]; last: Body[]; x: number; y: number; arrowTime: number } | null>(null);
  // Bumped whenever the view should re-frame the system (new preset, challenge, defaults, distance edits).
  const [fitToken, setFitToken] = useState(0);
  const lastFit = useRef(-1);

  const orbit = settings.mode === "orbit";
  const setup = setupBodies(settings);
  const names = bodyNames(settings.preset, setup.length);
  const selectedIndex = selected !== null && selected < setup.length ? selected : null;

  useEffect(() => {
    looks.current = {
      names,
      colors: setup.map((_, i) => rgb(orbitLook(setup, i))),
      selected: selectedIndex,
      arrows,
      arrowTime: arrowYears(setup),
    };
  });

  // Settings → simulation.
  useEffect(() => {
    if (!sim) return;
    if (settings.mode === "orbit") {
      const fit = lastFit.current !== fitToken;
      lastFit.current = fitToken;
      sim.setOrbitSystem(setupBodies(settings), fit);
    } else {
      lastFit.current = -1;
      sim.setGalaxy(settings.galaxy, settings.count);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reload only when the system itself changes
  }, [sim, settings.mode, settings.preset, settings.bodies, settings.galaxy, settings.count, fitToken]);

  useEffect(() => {
    sim?.setSpeed(orbit ? settings.speed : settings.galaxySpeed);
  }, [sim, orbit, settings.speed, settings.galaxySpeed]);
  useEffect(() => {
    sim?.setIntegrator(settings.integrator);
  }, [sim, settings.integrator]);
  useEffect(() => {
    sim?.setTrails(settings.trails);
  }, [sim, settings.trails]);
  useEffect(() => {
    if (playing) sim?.play();
    else sim?.pause();
  }, [sim, playing]);

  const update = (patch: Partial<NBodySettings>) => setSettings((prev) => ({ ...prev, ...patch }));

  /** The picture edits start from: the running system if it has moved, else the stored setup. */
  const editBase = () => (sim && stats && stats.time > 0 ? sim.currentBodies() : setup);
  const commitBodies = (bodies: Body[], fit = false) => {
    // Measurements of the old setup are stale now; showing them would also push old values back into the sliders.
    setStats(null);
    setPlaying(false);
    if (fit) setFitToken((t) => t + 1);
    update({ bodies: encodeBodies(bodies) });
  };

  // ── Pointer: drag a body or its arrow; otherwise pan. Wheel zooms at the cursor. ──
  const local = (e: PointerEvent | WheelEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };
  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!sim) return;
    const p = local(e);
    e.currentTarget.setPointerCapture(e.pointerId);
    pinch.down(e.pointerId, p.x, p.y);
    if (pinch.active) {
      // A second finger: this is a pinch, not an edit. Undo any half-done drag.
      const d = drag.current;
      if (d && d.kind !== "pan" && d.last !== d.base) sim.setOrbitSystem(d.base);
      drag.current = null;
      return;
    }
    const hit = orbit && overlayState.current ? hitTest(overlayState.current, p.x, p.y) : null;
    if (hit) {
      setSelected(hit.index);
      setPlaying(false);
      const base = sim.currentBodies();
      drag.current = { kind: hit.kind, index: hit.index, base, last: base, ...p, arrowTime: overlayState.current!.arrowTime };
    } else {
      drag.current = { kind: "pan", index: -1, base: [], last: [], ...p, arrowTime: 0 };
    }
  };
  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const p = local(e);
    const gesture = pinch.move(e.pointerId, p.x, p.y);
    if (gesture) {
      sim?.panBy(gesture.dx, gesture.dy);
      sim?.zoomAt(gesture.factor, gesture.x, gesture.y);
      return;
    }
    const d = drag.current;
    if (!sim || !d) return;
    if (d.kind === "pan") {
      sim.panBy(p.x - d.x, p.y - d.y);
      d.x = p.x;
      d.y = p.y;
      return;
    }
    const [wx, wy] = sim.toWorld(p.x, p.y);
    const b = d.base[d.index];
    const moved = d.kind === "body" ? { ...b, x: wx, y: wy } : { ...b, vx: (wx - b.x) / d.arrowTime, vy: (wy - b.y) / d.arrowTime };
    d.last = d.base.map((x, i) => (i === d.index ? moved : x));
    sim.setOrbitSystem(d.last);
  };
  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    pinch.up(e.pointerId);
    const d = drag.current;
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    if (d && d.kind !== "pan" && d.last !== d.base) commitBodies(d.last);
  };
  const onWheel = (e: WheelEvent<HTMLCanvasElement>) => {
    const p = local(e);
    sim?.zoomAt(e.deltaY > 0 ? 1 / 1.12 : 1.12, p.x, p.y);
  };

  const selectedReadout = stats?.orbits.find((o) => o.index === selectedIndex);
  const primary = primaryIndex(setup);
  const selectedBody = selectedIndex !== null ? setup[selectedIndex] : null;
  const isPrimary = selectedIndex === primary;
  // Slider values: live once the system has moved, else the setup's own.
  const setupRel = selectedBody && !isPrimary ? { r: Math.hypot(selectedBody.x - setup[primary].x, selectedBody.y - setup[primary].y), v: Math.hypot(selectedBody.vx - setup[primary].vx, selectedBody.vy - setup[primary].vy) } : null;
  const moving = !!stats && stats.time > 0;
  const liveR = (moving ? selectedReadout?.r : undefined) ?? setupRel?.r ?? 1;
  const liveV = (moving ? selectedReadout?.v : undefined) ?? setupRel?.v ?? 0;

  const addPlanet = () => {
    const base = editBase();
    if (base.length >= MAX_ORBIT_BODIES) return;
    const p = base[primaryIndex(base)];
    const farthest = Math.max(0.6, ...base.map((b) => Math.hypot(b.x - p.x, b.y - p.y)));
    const r = Number((farthest * 1.35).toFixed(2));
    const angle = (3 * Math.PI) / 4;
    const v = circularSpeed(r, p.m);
    const planet: Body = { m: 3.003e-6, x: p.x + r * Math.cos(angle), y: p.y + r * Math.sin(angle), vx: p.vx - v * Math.sin(angle), vy: p.vy + v * Math.cos(angle) };
    setSelected(base.length);
    commitBodies([...base, planet]);
  };
  const removeSelected = () => {
    if (selectedIndex === null || setup.length <= 1) return;
    commitBodies(editBase().filter((_, i) => i !== selectedIndex));
    setSelected(null);
  };

  const readouts = readoutsFor(settings, stats);
  const driftScale = Math.max(1e-6, ...(stats?.history.drift.map(Math.abs) ?? [0])) * 1.25;
  const keplerPoints = (stats?.kepler ?? []).map((k) => ({ x: k.a ** 3, y: k.T ** 2 }));
  const keplerMax = Math.max(1, ...keplerPoints.map((p) => Math.max(p.x, p.y))) * 1.1;

  const orbitControls = (
    <>
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">System</span>
        <select value={settings.bodies === "" ? settings.preset : ""} onChange={(e) => {
            if (!e.target.value) return;
            setSelected(1);
            setFitToken((t) => t + 1);
            update({ preset: e.target.value, bodies: "" });
          }} className={selectClass}>
          {settings.bodies !== "" && <option value="">Edited: {ORBIT_PRESETS.find((p) => p.id === settings.preset)?.label}</option>}
          {ORBIT_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </label>

      <ParameterSlider label="Speed" unit=" yr/s" min={0.05} max={3} step={0.05} decimals={2} color="blue" value={settings.speed} onChange={(v) => update({ speed: v })} />

      <fieldset className="space-y-3 rounded-lg border border-slate-700/60 p-3">
        <legend className="px-1 text-sm font-semibold text-slate-200">Body</legend>
        <div className="flex flex-wrap gap-1.5">
          {setup.map((_, i) => (
            <button key={i} type="button" aria-pressed={selectedIndex === i} onClick={() => setSelected(i)} className={toggleClass(selectedIndex === i)}>
              {names[i]}
            </button>
          ))}
        </div>
        {selectedBody && (
          <>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-400">Mass</span>
              <select
                aria-label="Mass"
                value={MASSES.some((m) => m.m === selectedBody.m) ? selectedBody.m : "custom"}
                onChange={(e) => {
                  const m = Number(e.target.value);
                  if (Number.isFinite(m)) commitBodies(editBase().map((b, i) => (i === selectedIndex ? { ...b, m } : b)));
                }}
                className={selectClass}
              >
                {!MASSES.some((m) => m.m === selectedBody.m) && <option value="custom">{selectedBody.m.toPrecision(3)} M☉</option>}
                {MASSES.map((m) => (
                  <option key={m.m} value={m.m}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            {!isPrimary && (
              <>
                <ParameterSlider label="Distance from star" unit=" AU" min={0.2} max={12} step={0.01} decimals={2} color="amber" value={Math.min(12, Math.max(0.2, liveR))} onChange={(r) => commitBodies(withDistance(editBase(), selectedIndex!, r), true)} />
                <ParameterSlider label="Orbital speed" unit=" AU/yr" min={0} max={20} step={0.01} decimals={2} color="green" value={Math.min(20, liveV)} onChange={(v) => commitBodies(withSpeed(editBase(), selectedIndex!, v))} />
              </>
            )}
          </>
        )}
        <div className="grid grid-cols-2 gap-2">
          <button type="button" onClick={addPlanet} disabled={setup.length >= MAX_ORBIT_BODIES} className="rounded bg-slate-700 px-3 py-1.5 text-sm text-slate-100 transition-colors hover:bg-slate-600 disabled:opacity-40">
            Add planet
          </button>
          <button type="button" onClick={removeSelected} disabled={selectedIndex === null || setup.length <= 1} className="rounded bg-red-900/50 px-3 py-1.5 text-sm text-red-200 transition-colors hover:bg-red-800/50 disabled:opacity-40">
            Remove
          </button>
        </div>
        <p className="text-xs text-slate-500">Drag a body to move it, or the tip of its arrow to change its velocity.</p>
        <button type="button" onClick={() => sim?.fit()} className="rounded bg-slate-800 px-3 py-1.5 text-sm text-slate-300 transition-colors hover:bg-slate-700">
          Fit view
        </button>
      </fieldset>

      <div className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-slate-300">
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={settings.trails} onChange={(e) => update({ trails: e.target.checked })} />
          Trails
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={arrows} onChange={(e) => setArrows(e.target.checked)} />
          Velocity arrows
        </label>
      </div>
    </>
  );

  const galaxyControls = (
    <>
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">System</span>
        <select value={settings.galaxy} onChange={(e) => update({ galaxy: e.target.value as GalaxyPreset })} className={selectClass}>
          {GALAXY_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Stars</span>
        <select
          value={settings.count}
          onChange={(e) => {
            const count = GALAXY_COUNTS.find((c) => String(c) === e.target.value);
            if (count) update({ count });
          }}
          className={selectClass}
        >
          {GALAXY_COUNTS.map((c) => (
            <option key={c} value={c}>
              {c.toLocaleString("en-US")} ({((c * c) / 1e6).toFixed(0)} M pairs per step)
            </option>
          ))}
        </select>
      </label>
      <ParameterSlider label="Speed" unit=" t/s" min={0.1} max={4} step={0.1} decimals={1} color="blue" value={settings.galaxySpeed} onChange={(v) => update({ galaxySpeed: v })} />
    </>
  );

  const controls = (
    <>
      <div role="group" aria-label="Mode" className="grid grid-cols-2 gap-2">
        {(["orbit", "galaxy"] as const).map((m) => (
          <button key={m} type="button" aria-pressed={settings.mode === m} onClick={() => update({ mode: m })} className={toggleClass(settings.mode === m)}>
            {m === "orbit" ? "Orbit lab" : "Galaxies"}
          </button>
        ))}
      </div>

      {orbit ? orbitControls : galaxyControls}

      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">Integrator</legend>
        <div className="flex gap-4 text-sm text-slate-300">
          {(["leapfrog", "euler"] as const).map((i) => (
            <label key={i} className="flex items-center gap-1.5">
              <input type="radio" name="nbody-integrator" checked={settings.integrator === i} onChange={() => update({ integrator: i })} />
              {i === "leapfrog" ? "Leapfrog" : "Euler"}
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset className="space-y-3 border-t border-slate-800 pt-4">
        <legend className="text-sm font-semibold text-slate-200">Measure</legend>
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
          <dt className="text-slate-500">Energy E</dt>
          <dd data-testid="nb-energy">{stats ? stats.total.toPrecision(4) : "—"}</dd>
          <dt className="text-slate-500">Kinetic / potential</dt>
          <dd>{stats ? `${stats.kinetic.toPrecision(3)} / ${stats.potential.toPrecision(3)}` : "—"}</dd>
          <dt className="text-slate-500">Energy drift</dt>
          <dd data-testid="nb-drift">{stats ? signedPercent(stats.drift) : "—"}</dd>
          <dt className="text-slate-500">Angular momentum drift</dt>
          <dd>{stats ? signedPercent(stats.angularDrift) : "—"}</dd>
        </dl>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={settings.showGraph} onChange={(e) => update({ showGraph: e.target.checked })} />
          Graphs
        </label>
        {settings.showGraph && stats && stats.history.t.length > 1 && (
          <MiniChart
            title="Total energy change (%)"
            xLabel={orbit ? "time (years)" : "time"}
            yLabel="ΔE / |E₀| (%)"
            x={stats.history.t}
            y={stats.history.drift.map((d) => d * 100)}
            yMin={-driftScale * 100}
            yMax={driftScale * 100}
            format={(v) => (Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 0.1 ? v.toFixed(1) : v.toExponential(0))}
            summary={`The total energy has changed by ${signedPercent(stats.drift)} since the start.`}
          />
        )}
        {settings.showGraph && orbit && keplerPoints.length > 0 && (
          <MiniChart
            title="Kepler's third law: T² against a³"
            xLabel="a³ (AU³)"
            yLabel="T² (years²)"
            x={[0, keplerMax]}
            y={[0, keplerMax / (setup[primary]?.m || 1)]}
            yMax={keplerMax}
            points={keplerPoints}
            format={(v) => (v >= 10 ? v.toFixed(0) : v.toFixed(1))}
            summary={`${keplerPoints.length} measured orbits; the line is T² = a³ / M.`}
          />
        )}
      </fieldset>
    </>
  );

  const explanation = (
    <>
      <p>
        Every body pulls on every other with Newton’s law of gravity, <em>F = G·m₁·m₂ / r²</em>. The GPU adds up all
        those pulls — for thousands of stars that is millions of pairs every step — and moves the bodies forward in time.
      </p>
      <p>
        <strong className="text-slate-100">Orbit lab</strong> uses astronomical units, years and solar masses, so Earth’s
        orbit takes exactly 1 year. The lab <em>measures</em> each planet’s period from its motion and plots T² against
        a³: Kepler’s third law appears as a straight line.
      </p>
      <p>
        <strong className="text-slate-100">Integrator.</strong> Leapfrog keeps the energy error bounded for ever; the
        simpler Euler method slowly pumps energy in, so orbits spiral outwards. Watch the energy graph.
      </p>
      <p className="text-slate-400">Drag empty space to pan, scroll to zoom. Copy link shares the exact starting setup.</p>
    </>
  );

  return (
    <SimLayout
      title="N-Body Orbital Mechanics"
      subject="physics"
      difficulty="medium"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      challenges={
        <ChallengesPanel
          simId={SIM_ID}
          challenges={NBODY_CHALLENGES}
          params={settings}
          readouts={readouts}
          onSetup={(patch) => {
            setStats(null);
            setPlaying(false);
            setSelected(1);
            setFitToken((t) => t + 1);
            update({ ...patch, bodies: patch.bodies ?? "" });
          }}
        />
      }
      canvasRef={canvasRef}
      onResetDefaults={() => {
        setPlaying(false);
        setSelected(1);
        setFitToken((t) => t + 1);
        setSettings(NBODY_DEFAULTS);
      }}
      playing={playing}
      onPlayPause={setPlaying}
      onStep={() => sim?.step()}
      onReset={() => sim?.reset()}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 block h-full w-full touch-none cursor-crosshair"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
      />
      {orbit && <canvas ref={overlayRef} aria-hidden className="pointer-events-none absolute inset-0 h-full w-full" />}

      <div className="pointer-events-none absolute left-3 top-3 max-w-[60%] rounded bg-black/60 px-2.5 py-1.5 font-mono text-[11px] leading-snug text-slate-200">
        <div className="text-slate-400">{orbit ? "Time" : `${(stats?.n ?? settings.count).toLocaleString("en-US")} stars · time`}</div>
        <div className="text-base font-bold text-sky-300" data-testid="nb-time">
          {stats ? `${stats.time.toFixed(2)}${orbit ? " yr" : ""}` : "—"}
        </div>
        {orbit && selectedIndex !== null && !isPrimary && (
          <div aria-live="polite">
            <span className="text-slate-400">{names[selectedIndex]}: </span>
            {selectedReadout ? (
              selectedReadout.energy >= 0 ? (
                <span data-testid="nb-status" className="text-amber-300">
                  escaping
                </span>
              ) : (
                <>
                  <span data-testid="nb-period">period {selectedReadout.period !== undefined ? `${selectedReadout.period.toFixed(3)} yr` : "measuring…"}</span>
                  <span className="text-slate-400"> · e </span>
                  <span data-testid="nb-e">{fixed(selectedReadout.e, 3)}</span>
                  <span className="text-slate-400"> · a </span>
                  <span>{fixed(selectedReadout.a, 2)} AU</span>
                </>
              )
            ) : (
              "—"
            )}
          </div>
        )}
      </div>
    </SimLayout>
  );
}
