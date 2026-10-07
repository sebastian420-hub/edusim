"use client";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent, WheelEvent } from "react";
import { ChallengesPanel } from "@/components/ChallengesPanel";
import { MiniChart } from "@/components/MiniChart";
import { ParameterSlider } from "@/components/ParameterSlider";
import { SimLayout } from "@/components/SimLayout";
import { createPinchTracker } from "@/lib/gestures";
import { useGpuSim } from "@/lib/gpu/useGpuSim";
import { usePersistedParams } from "@/lib/usePersistedParams";
import { prefersReducedMotion, usePlaying } from "@/lib/usePlaying";
import { PENDULUM_CHALLENGES, readoutsFor } from "./challenges";
import { deg, FULL_WINDOW, mapAngles, normalModes } from "./pendulum";
import type { FractalWindow } from "./pendulum";
import { CROWD_COUNTS, FRACTAL_SIZES, isSymmetric, panWindow, PENDULUM_DEFAULTS, PENDULUM_SCHEMA, physicsOf, startOf, windowOf, zoomWindow } from "./settings";
import type { PendulumSettings, PendulumView } from "./settings";
import { createDoublePendulum } from "./sim";
import type { PendulumConfig, PendulumHandle, PendulumStats } from "./sim";

const SIM_ID = "double-pendulum";

const VIEWS: { id: PendulumView; label: string }[] = [
  { id: "pendulum", label: "Pendulum" },
  { id: "butterfly", label: "Butterfly" },
  { id: "fractal", label: "Fractal" },
];

const selectClass = "rounded border border-slate-700 bg-slate-800 p-2 text-sm text-slate-200";
const toggleClass = (on: boolean) =>
  `rounded px-2 py-1.5 text-sm transition-colors ${on ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"}`;
const buttonClass = "rounded bg-slate-800 px-3 py-1.5 text-sm text-slate-300 transition-colors hover:bg-slate-700";

const signedPercent = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toPrecision(2)} %`;
/** Angle in degrees folded into (−180°, 180°], as a person would read it. */
const wrapDeg = (r: number) => {
  const d = deg(Math.atan2(Math.sin(r), Math.cos(r)));
  return `${d.toFixed(1)}°`;
};
const superscript = (n: number) => String(n).replace(/-/g, "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]);
const tenTo = (exponent: number) => `10${superscript(Math.round(exponent))}`;
/** Release angles as stored in a link: one decimal of a degree. */
const roundAngle = (r: number) => Math.round(deg(Math.atan2(Math.sin(r), Math.cos(r))) * 10) / 10;

interface Preview {
  start?: [number, number];
  window?: FractalWindow;
}

function configOf(s: PendulumSettings, preview: Preview = {}): PendulumConfig {
  return {
    view: s.view,
    start: preview.start ?? startOf(s),
    params: physicsOf(s),
    integrator: s.integrator,
    count: s.count,
    nudge: 10 ** s.nudge,
    fractalSize: s.res,
    window: preview.window ?? windowOf(s),
    boundary: s.boundary,
  };
}

export default function DoublePendulumControls() {
  const [stats, setStats] = useState<PendulumStats | null>(null);
  // The factory must be stable: created once.
  const [factory] = useState(() => (ctx: Parameters<typeof createDoublePendulum>[0]) => createDoublePendulum(ctx, { onStats: setStats }));
  const { canvasRef, status, sim, quality } = useGpuSim<PendulumHandle>(factory);
  const [settings, setSettings] = usePersistedParams(SIM_ID, PENDULUM_SCHEMA, PENDULUM_DEFAULTS);
  const [playing, setPlaying] = usePlaying(sim, true);
  const [pinch] = useState(createPinchTracker);
  const [hover, setHover] = useState<{ a1: number; a2: number; t: number | null } | null>(null);
  const preview = useRef<Preview>({});
  const drag = useRef<{ kind: "bob1" | "bob2" | "pan"; x: number; y: number; moved: boolean; window: FractalWindow } | null>(null);
  const commitTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const probing = useRef(false);

  const update = (patch: Partial<PendulumSettings>) => setSettings((prev) => ({ ...prev, ...patch }));
  const fractal = settings.view === "fractal";
  const symmetric = isSymmetric(settings);

  // Settings → simulation. A committed setting supersedes any preview (a drag in progress, a pending zoom).
  useEffect(() => {
    if (!sim) return;
    preview.current = {};
    sim.configure(configOf(settings));
  }, [sim, settings]);
  useEffect(() => {
    sim?.setSpeed(settings.speed);
  }, [sim, settings.speed]);
  useEffect(() => {
    sim?.setTrail(settings.trail);
  }, [sim, settings.trail]);
  useEffect(
    () => () => {
      if (commitTimer.current) clearTimeout(commitTimer.current);
    },
    [],
  );

  const showPreview = (next: Preview) => {
    preview.current = { ...preview.current, ...next };
    sim?.configure(configOf(settings, preview.current));
  };

  // ── Fractal window: zoom and pan show at once, and are written to the link once the gesture pauses. ──
  const currentWindow = () => preview.current.window ?? windowOf(settings);
  const commitWindowSoon = () => {
    if (commitTimer.current) clearTimeout(commitTimer.current);
    commitTimer.current = setTimeout(() => {
      commitTimer.current = null;
      const w = preview.current.window;
      if (w) update({ fx: w.cx, fy: w.cy, fspan: w.span });
    }, 300);
  };
  const zoomBy = (factor: number, u = 0.5, v = 0.5) => {
    showPreview({ window: zoomWindow(currentWindow(), factor, u, v) });
    commitWindowSoon();
  };

  // ── Pointer ──
  const local = (e: PointerEvent | WheelEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };
  const mapSide = (e: PointerEvent | WheelEvent) => Math.min(e.currentTarget.clientWidth, e.currentTarget.clientHeight) || 1;

  const anglesFromPointer = (kind: "bob1" | "bob2", x: number, y: number): [number, number] => {
    const [wx, wy] = sim!.toWorld(x, y);
    const start = preview.current.start ?? startOf(settings);
    if (kind === "bob1") return [Math.atan2(wx, -wy), start[1]];
    const p = physicsOf(settings);
    const x1 = p.l1 * Math.sin(start[0]);
    const y1 = -p.l1 * Math.cos(start[0]);
    return [start[0], Math.atan2(wx - x1, -(wy - y1))];
  };

  const onPointerDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!sim) return;
    const p = local(e);
    e.currentTarget.setPointerCapture(e.pointerId);
    pinch.down(e.pointerId, p.x, p.y);
    if (pinch.active) {
      drag.current = null;
      return;
    }
    if (fractal) {
      drag.current = { kind: "pan", ...p, moved: false, window: currentWindow() };
      return;
    }
    const { x1, y1, x2, y2 } = sim.bobs();
    const [sx1, sy1] = sim.toScreen(x1, y1);
    const [sx2, sy2] = sim.toScreen(x2, y2);
    const d1 = Math.hypot(p.x - sx1, p.y - sy1);
    const d2 = Math.hypot(p.x - sx2, p.y - sy2);
    const reach = e.pointerType === "touch" ? 36 : 24;
    if (Math.min(d1, d2) > reach) return;
    const kind = d2 <= d1 ? "bob2" : "bob1";
    setPlaying(false);
    drag.current = { kind, ...p, moved: false, window: currentWindow() };
  };

  const onPointerMove = (e: PointerEvent<HTMLCanvasElement>) => {
    const p = local(e);
    const gesture = pinch.move(e.pointerId, p.x, p.y);
    if (gesture && fractal && sim) {
      const side = mapSide(e);
      let w = panWindow(currentWindow(), gesture.dx / side, gesture.dy / side);
      const [u, v] = sim.mapUV(gesture.x, gesture.y);
      w = zoomWindow(w, gesture.factor, u, v);
      showPreview({ window: w });
      commitWindowSoon();
      return;
    }
    const d = drag.current;
    if (sim && d) {
      if (Math.hypot(p.x - d.x, p.y - d.y) > 4) d.moved = true;
      if (!d.moved) return;
      if (d.kind === "pan") {
        const side = mapSide(e);
        showPreview({ window: panWindow(d.window, (p.x - d.x) / side, (p.y - d.y) / side) });
      } else {
        showPreview({ start: anglesFromPointer(d.kind, p.x, p.y) });
      }
      return;
    }
    // Hovering the map: read the flip time under the cursor (one probe in flight at a time).
    if (fractal && sim && e.pointerType !== "touch" && !probing.current) {
      const [u, v] = sim.mapUV(p.x, p.y);
      if (u < 0 || v < 0 || u >= 1 || v >= 1) {
        setHover(null);
        return;
      }
      const [a1, a2] = mapAngles(u, v, currentWindow());
      probing.current = true;
      sim.probe(p.x, p.y).then((t) => {
        probing.current = false;
        setHover({ a1, a2, t });
      });
    }
  };

  const onPointerUp = (e: PointerEvent<HTMLCanvasElement>) => {
    const wasPinch = pinch.active;
    pinch.up(e.pointerId);
    const d = drag.current;
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    if (!d || wasPinch || !sim) return;
    if (d.kind === "pan") {
      if (d.moved) {
        const w = preview.current.window;
        if (w) update({ fx: w.cx, fy: w.cy, fspan: w.span });
      } else {
        // A click on the map opens that start as a real pendulum.
        const [u, v] = sim.mapUV(local(e).x, local(e).y);
        if (u < 0 || v < 0 || u >= 1 || v >= 1) return;
        const [a1, a2] = mapAngles(u, v, currentWindow());
        setHover(null);
        update({ view: "pendulum", a1: roundAngle(a1), a2: roundAngle(a2) });
        setPlaying(!prefersReducedMotion());
      }
      return;
    }
    const start = preview.current.start;
    if (d.moved && start) update({ a1: roundAngle(start[0]), a2: roundAngle(start[1]) });
    // Letting go of the bob releases the pendulum.
    setPlaying(!prefersReducedMotion());
  };

  const onWheel = (e: WheelEvent<HTMLCanvasElement>) => {
    if (!fractal || !sim) return;
    const p = local(e);
    const [u, v] = sim.mapUV(p.x, p.y);
    zoomBy(e.deltaY > 0 ? 1 / 1.25 : 1.25, Math.min(1, Math.max(0, u)), Math.min(1, Math.max(0, v)));
  };

  // ── Readouts ──
  const swing = stats && stats.view !== "fractal" ? stats.swing : null;
  const crowd = stats?.view === "butterfly" ? stats.crowd : null;
  const map = stats?.view === "fractal" ? stats.fractal : null;
  const modes = normalModes({ g: settings.g, l1: 1 });
  const lastPeriod = swing?.periods.at(-1);
  const readouts = readoutsFor(stats);
  const phaseDeg = swing ? { x: swing.phase.t1.map(deg), y: swing.phase.t2.map(deg) } : null;
  const phaseRange = phaseDeg && phaseDeg.y.length > 1 ? Math.max(5, ...phaseDeg.y.map(Math.abs), ...phaseDeg.x.map(Math.abs)) * 1.1 : 0;

  const releaseControls = (
    <fieldset className="space-y-3">
      <legend className="mb-1 text-sm font-semibold text-slate-200">Release from rest</legend>
      <ParameterSlider label="Upper arm θ₁" unit="°" min={-180} max={180} step={1} decimals={1} color="blue" value={settings.a1} onChange={(v) => update({ a1: v })} />
      <ParameterSlider label="Lower arm θ₂" unit="°" min={-180} max={180} step={1} decimals={1} color="purple" value={settings.a2} onChange={(v) => update({ a2: v })} />
      <p className="text-xs text-slate-400">Or drag a bob on the canvas and let go.</p>
    </fieldset>
  );

  const physicsControls = (
    <fieldset className="space-y-3 rounded-lg border border-slate-700/60 p-3">
      <legend className="px-1 text-sm font-semibold text-slate-200">Pendulum</legend>
      <ParameterSlider label="Lower mass m₂" unit=" kg" min={0.1} max={5} step={0.1} decimals={1} color="amber" value={settings.m2} onChange={(v) => update({ m2: v })} />
      <ParameterSlider label="Lower arm l₂" unit=" m" min={0.25} max={2} step={0.05} decimals={2} color="green" value={settings.l2} onChange={(v) => update({ l2: v })} />
      <ParameterSlider label="Gravity g" unit=" m/s²" min={1} max={25} step={0.01} decimals={2} color="rose" value={settings.g} onChange={(v) => update({ g: v })} />
      <p className="text-xs text-slate-400">The upper bob is 1 kg on a 1 m arm.</p>
    </fieldset>
  );

  const butterflyControls = (
    <>
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Pendulums</span>
        <select
          value={settings.count}
          onChange={(e) => {
            const count = CROWD_COUNTS.find((c) => String(c) === e.target.value);
            if (count) update({ count });
          }}
          className={selectClass}
        >
          {CROWD_COUNTS.map((c) => (
            <option key={c} value={c}>
              {c.toLocaleString("en-US")}
            </option>
          ))}
        </select>
      </label>
      <ParameterSlider label="Nudge between neighbours (10ˣ rad)" unit="" min={-12} max={-1} step={1} decimals={0} color="amber" value={settings.nudge} onChange={(v) => update({ nudge: v })} />
      <p className="text-xs text-slate-400">
        White: two pendulums computed in 64-bit, exactly one nudge apart. Colours: the crowd, computed on the GPU in 32-bit,
        which cannot store differences below about 2·10⁻⁷ rad, so tiny nudges start one 32-bit step apart.
      </p>
    </>
  );

  const fractalControls = (
    <>
      <label className="flex flex-col gap-1">
        <span className="text-sm font-medium text-slate-200">Detail</span>
        <select
          value={settings.res}
          onChange={(e) => {
            const res = FRACTAL_SIZES.find((c) => String(c) === e.target.value);
            if (res) update({ res });
          }}
          className={selectClass}
        >
          {FRACTAL_SIZES.map((c) => (
            <option key={c} value={c}>
              {c} × {c} ({((c * c) / 1e6).toFixed(c === 512 ? 2 : 1)} M pendulums)
            </option>
          ))}
        </select>
      </label>
      <div className="grid grid-cols-3 gap-2">
        <button type="button" className={buttonClass} onClick={() => zoomBy(2)}>
          Zoom in
        </button>
        <button type="button" className={buttonClass} onClick={() => zoomBy(0.5)}>
          Zoom out
        </button>
        <button type="button" className={buttonClass} onClick={() => update({ fx: FULL_WINDOW.cx, fy: FULL_WINDOW.cy, fspan: FULL_WINDOW.span })}>
          Full map
        </button>
      </div>
      <label className="flex items-center gap-2 text-sm text-slate-300">
        <input type="checkbox" checked={settings.boundary} onChange={(e) => update({ boundary: e.target.checked })} />
        Energy boundary (no flips inside)
      </label>
      <p className="text-xs text-slate-400">Drag to pan, scroll or pinch to zoom, click a pixel to release that start.</p>
    </>
  );

  const controls = (
    <>
      <div role="group" aria-label="View" className="grid grid-cols-3 gap-2">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            aria-pressed={settings.view === v.id}
            onClick={() => {
              setHover(null);
              update({ view: v.id });
            }}
            className={toggleClass(settings.view === v.id)}
          >
            {v.label}
          </button>
        ))}
      </div>

      {!fractal && releaseControls}
      {settings.view === "butterfly" && butterflyControls}
      {fractal && fractalControls}
      {!fractal && <ParameterSlider label="Speed" unit=" s/s" min={0.1} max={2} step={0.05} decimals={2} color="blue" value={settings.speed} onChange={(v) => update({ speed: v })} />}
      {physicsControls}

      <fieldset>
        <legend className="mb-2 text-sm font-semibold text-slate-200">Integrator</legend>
        <div className="flex gap-4 text-sm text-slate-300">
          {(["rk4", "euler"] as const).map((i) => (
            <label key={i} className="flex items-center gap-1.5">
              <input type="radio" name="pendulum-integrator" checked={settings.integrator === i} onChange={() => update({ integrator: i })} />
              {i === "rk4" ? "Runge–Kutta 4" : "Euler"}
            </label>
          ))}
        </div>
      </fieldset>
      {!fractal && (
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input type="checkbox" checked={settings.trail} onChange={(e) => update({ trail: e.target.checked })} />
          Trail of the lower bob
        </label>
      )}

      <fieldset className="space-y-3 border-t border-slate-800 pt-4">
        <legend className="text-sm font-semibold text-slate-200">Measure</legend>
        {!fractal && (
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
            <dt className="text-slate-500">Energy E</dt>
            <dd data-testid="dp-energy">{swing ? `${swing.total.toFixed(4)} J` : "—"}</dd>
            <dt className="text-slate-500">Energy drift</dt>
            <dd data-testid="dp-drift">{swing ? signedPercent(swing.drift) : "—"}</dd>
            <dt className="text-slate-500">Flips over the top</dt>
            <dd data-testid="dp-flips">{swing ? `${swing.flips}${swing.firstFlip !== undefined ? ` (first at ${swing.firstFlip.toFixed(2)} s)` : ""}` : "—"}</dd>
            <dt className="text-slate-500">Period of θ₁</dt>
            <dd data-testid="dp-period">{lastPeriod !== undefined ? `${lastPeriod.toFixed(3)} s` : "—"}</dd>
            {symmetric && (
              <>
                <dt className="text-slate-500">Normal modes 2π/ω</dt>
                <dd>
                  {modes.slow.period.toFixed(3)} s · {modes.fast.period.toFixed(3)} s
                </dd>
              </>
            )}
          </dl>
        )}
        {settings.view === "pendulum" && phaseDeg && phaseRange > 0 && (
          <MiniChart
            title="Phase portrait: θ₂ against θ₁"
            xLabel="θ₁ (°)"
            yLabel="θ₂ (°)"
            x={phaseDeg.x}
            y={phaseDeg.y}
            xMin={-phaseRange}
            xMax={phaseRange}
            yMin={-phaseRange}
            yMax={phaseRange}
            format={(v) => v.toFixed(0)}
            summary={`The last ${phaseDeg.x.length} samples of the two angles. A closed loop or a neat figure means regular motion; a tangle means chaos.`}
          />
        )}
        {crowd && (
          <>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
              <dt className="text-slate-500">Gap (white pair)</dt>
              <dd data-testid="dp-spread">{crowd.spread > 0 ? `${crowd.spread.toExponential(1)} rad` : "0"}</dd>
              <dt className="text-slate-500">Growth rate λ</dt>
              <dd data-testid="dp-lambda">{crowd.lambda !== undefined ? `${crowd.lambda.toFixed(2)} /s` : "—"}</dd>
              <dt className="text-slate-500">Parted (gap &gt; 1 rad)</dt>
              <dd data-testid="dp-visible">{crowd.visibleAt !== undefined ? `at ${crowd.visibleAt.toFixed(1)} s` : "not yet"}</dd>
              <dt className="text-slate-500">GPU 32-bit vs CPU 64-bit</dt>
              <dd data-testid="dp-precision">
                {crowd.precisionPartedAt !== undefined ? `parted at ${crowd.precisionPartedAt.toFixed(1)} s` : crowd.precisionGap !== undefined ? `${crowd.precisionGap.toExponential(1)} rad` : "—"}
              </dd>
            </dl>
            {crowd.history.t.length > 1 && (
              <MiniChart
                title="Gap between the white pair (log scale)"
                xLabel="time (s)"
                yLabel="log₁₀ gap (rad)"
                x={crowd.history.t}
                y={crowd.history.log10}
                yMin={-12}
                yMax={1}
                markers={crowd.visibleAt !== undefined ? [{ x: crowd.visibleAt, label: "parted" }] : []}
                format={(v) => v.toFixed(0)}
                summary={`The gap is ${crowd.spread.toExponential(1)} rad${crowd.lambda !== undefined ? `, growing by a factor e every ${(1 / crowd.lambda).toFixed(2)} s` : ""}.`}
              />
            )}
          </>
        )}
        {fractal && (
          <>
            <div aria-hidden className="h-3 w-full rounded" style={{ background: "linear-gradient(to right, #fde725, #5ec962, #21918c, #3b528b, #440154)" }} />
            <div className="flex justify-between text-xs text-slate-400">
              <span>flips at once</span>
              <span>after 30 s</span>
            </div>
            <p className="text-xs text-slate-400">Dark: no flip (yet). The white curve is the energy boundary.</p>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-xs text-slate-300">
              <dt className="text-slate-500">Each pendulum has run</dt>
              <dd data-testid="dp-map-time">{map ? `${map.time.toFixed(1)} of ${map.tMax} s` : "—"}</dd>
              <dt className="text-slate-500">Under the cursor</dt>
              <dd data-testid="dp-hover">{hover ? `${deg(hover.a1).toFixed(1)}°, ${deg(hover.a2).toFixed(1)}°` : "—"}</dd>
              <dt className="text-slate-500">First flip</dt>
              <dd data-testid="dp-hover-flip">{hover ? (hover.t === null ? "—" : hover.t < 0 ? "none yet" : `${hover.t.toFixed(2)} s`) : "—"}</dd>
            </dl>
          </>
        )}
      </fieldset>
    </>
  );

  const explanation = (
    <>
      <p>
        Two weights on rigid arms: about as simple as a machine gets, and it follows Newton’s laws exactly. Yet from a
        high release its motion is <em>chaotic</em>: deterministic, but unpredictable in the long run.
      </p>
      <p>
        <strong className="text-slate-100">Pendulum.</strong> Drag a bob and let go. The energy readout checks the
        integrator (Runge–Kutta 4 keeps it to a millionth); the phase portrait shows regular loops for small swings and
        a tangle for big ones.
      </p>
      <p>
        <strong className="text-slate-100">Butterfly.</strong> A crowd of pendulums released almost identically. The gap
        between them grows exponentially, at a rate λ (the Lyapunov exponent), until they have nothing in common.
      </p>
      <p>
        <strong className="text-slate-100">Fractal.</strong> One pendulum per pixel, released from rest at the angles of
        that pixel (θ₁ across, θ₂ up), coloured by how soon an arm first swings over the top: up to a million pendulums at
        once on the GPU.
      </p>
    </>
  );

  const summary = !stats
    ? undefined
    : stats.view === "fractal"
      ? `Flip-time map, ${stats.fractal.size} × ${stats.fractal.size} pendulums, ${stats.fractal.time.toFixed(1)} of ${stats.fractal.tMax} s.`
      : stats.view === "butterfly"
        ? `${stats.crowd.n} pendulums, time ${stats.crowd.time.toFixed(1)} s, gap ${stats.crowd.spread.toExponential(1)} rad${stats.crowd.visibleAt !== undefined ? `, parted at ${stats.crowd.visibleAt.toFixed(1)} s` : ""}.`
        : `Time ${stats.swing.time.toFixed(1)} s, angles ${wrapDeg(stats.swing.angles[0])} and ${wrapDeg(stats.swing.angles[1])}, ${stats.swing.flips} flips, energy changed by ${signedPercent(stats.swing.drift)}.`;

  return (
    <SimLayout
      summary={summary}
      title="Double Pendulum Chaos"
      subject="physics"
      difficulty="easy"
      status={status}
      quality={quality}
      controls={controls}
      explanation={explanation}
      challenges={
        <ChallengesPanel
          simId={SIM_ID}
          challenges={PENDULUM_CHALLENGES}
          params={settings}
          readouts={readouts}
          onSetup={(patch) => {
            setStats(null);
            setHover(null);
            setPlaying(!prefersReducedMotion());
            update(patch);
          }}
        />
      }
      canvasRef={canvasRef}
      onResetDefaults={() => {
        setHover(null);
        setSettings(PENDULUM_DEFAULTS);
      }}
      playing={playing}
      onPlayPause={setPlaying}
      onStep={() => sim?.step()}
      onReset={() => sim?.reset()}
    >
      <canvas
        ref={canvasRef}
        className={`absolute inset-0 block h-full w-full touch-none ${fractal ? "cursor-crosshair" : "cursor-grab"}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => setHover(null)}
        onWheel={onWheel}
      />
      <div className="pointer-events-none absolute left-3 top-3 max-w-[60%] rounded bg-black/60 px-2.5 py-1.5 font-mono text-[11px] leading-snug text-slate-200">
        {fractal ? (
          <>
            <div className="text-slate-400">{settings.res.toLocaleString("en-US")}² pendulums · time</div>
            <div className="text-base font-bold text-sky-300" data-testid="dp-time">
              {map ? `${map.time.toFixed(1)} s` : "—"}
            </div>
            {hover && (
              <div>
                θ₁ {deg(hover.a1).toFixed(1)}°, θ₂ {deg(hover.a2).toFixed(1)}°: {hover.t === null ? "—" : hover.t < 0 ? "no flip yet" : `flips at ${hover.t.toFixed(2)} s`}
              </div>
            )}
          </>
        ) : (
          <>
            <div className="text-slate-400">{settings.view === "butterfly" ? `${settings.count.toLocaleString("en-US")} pendulums, ${tenTo(settings.nudge)} rad apart · time` : "Time"}</div>
            <div className="text-base font-bold text-sky-300" data-testid="dp-time">
              {swing ? `${swing.time.toFixed(2)} s` : "—"}
            </div>
            {swing && (
              <div aria-live="off">
                θ₁ {wrapDeg(swing.angles[0])} · θ₂ {wrapDeg(swing.angles[1])}
              </div>
            )}
            {crowd && <div>gap {crowd.spread > 0 ? crowd.spread.toExponential(1) : "0"} rad</div>}
          </>
        )}
      </div>
    </SimLayout>
  );
}
