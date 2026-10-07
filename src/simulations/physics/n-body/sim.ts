import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import { createEngine } from "./engine";
import type { Engine, Snapshot } from "./engine";
import { bodiesFromState, collidingGalaxies, conserved, diskGalaxy, G_ORBIT, orbitalElements, OrbitTracker, plummerCluster, primaryIndex, toCentreOfMassFrame } from "./nbody";
import type { Body, Cloud, Integrator } from "./nbody";
import { createRenderer } from "./renderer";
import type { Renderer, View } from "./renderer";
import type { GalaxyPreset } from "./settings";
import { MAX_ORBIT_BODIES } from "./settings";

/** Orbit-lab time step (years): 1000 steps per Earth orbit keeps leapfrog's energy error around 10⁻⁶. */
const ORBIT_DT = 1e-3;
/** Galaxy time step (G = 1 units). */
const GALAXY_DT = 0.01;
/** Softening: tiny in the orbit lab (Kepler stays exact), larger for star clouds (no close-encounter spikes). */
const ORBIT_EPS2 = 1e-6;
const GALAXY_EPS2 = 0.05 * 0.05;
/** Upper bound on pair interactions per frame, so a big cloud at high speed cannot stall the page. */
const PAIRS_PER_FRAME = 6e8;
const MAX_STEPS_PER_FRAME = 64;
/**
 * Orbit lab: the state is read back after at most this many steps, so even Mercury turns less than half a radian
 * between samples and its period is measured correctly however slow the frame rate (a sparser sampling would alias).
 */
const SAMPLE_STEPS = 16;
const STATS_INTERVAL_MS = 100;
const GALAXY_READ_INTERVAL_MS = 250;
const HISTORY = 300;

export interface OrbitReadout {
  index: number;
  /** Distance and speed relative to the primary (most massive body). */
  r: number;
  v: number;
  /** Osculating two-body orbit around the primary. */
  a: number;
  e: number;
  /** Specific orbital energy: < 0 bound, ≥ 0 escaping. */
  energy: number;
  /** Measured duration of the last full orbit (undefined until one is complete). */
  period?: number;
  orbits: number;
}

export interface KeplerPoint {
  body: number;
  /** Semi-major axis (AU) when the orbit completed. */
  a: number;
  /** Measured period (years). */
  T: number;
}

export interface NBodyStats {
  mode: "orbit" | "galaxy";
  /** Simulation time (years in the orbit lab). */
  time: number;
  n: number;
  kinetic: number;
  potential: number;
  total: number;
  /** Relative change of the total energy since the start, (E − E₀)/|E₀|. */
  drift: number;
  /** Relative change of the angular momentum since the start. */
  angularDrift: number;
  /** Energy drift over time (fractions), for the graph. */
  history: { t: number[]; drift: number[] };
  primary: number;
  orbits: OrbitReadout[];
  kepler: KeplerPoint[];
}

/** Positions for the on-canvas labels and arrows, extrapolated to the moment of drawing. */
export interface FrameInfo {
  bodies: Body[];
  view: View;
  /** Device pixels per CSS pixel of the canvas. */
  pixelRatio: number;
  time: number;
}

export interface NBodyOptions {
  onStats?: (stats: NBodyStats | null) => void;
  onFrame?: (info: FrameInfo) => void;
}

export interface NBodyHandle extends SimHandle {
  /** Loads an orbit-lab system (velocities are taken relative to the centre of mass). `fit` re-frames the view. */
  setOrbitSystem(bodies: Body[], fit?: boolean): void;
  setGalaxy(preset: GalaxyPreset, count: number): void;
  /** Simulated time units per second of real time. */
  setSpeed(speed: number): void;
  setIntegrator(integrator: Integrator): void;
  setTrails(on: boolean): void;
  play(): void;
  pause(): void;
  step(): void;
  reset(): void;
  panBy(dxCss: number, dyCss: number): void;
  zoomAt(factor: number, xCss: number, yCss: number): void;
  fit(): void;
  /** Canvas CSS pixels → world coordinates. */
  toWorld(xCss: number, yCss: number): [number, number];
  /** The current orbit-lab bodies (from the latest read-back, extrapolated to now). */
  currentBodies(): Body[];
}

type System = { kind: "orbit"; bodies: Body[] } | { kind: "galaxy"; preset: GalaxyPreset; count: number };

const STAR = [1, 0.84, 0.55];
const PLANET_COLORS = [
  [0.38, 0.66, 1],
  [1, 0.55, 0.35],
  [0.4, 0.9, 0.7],
  [0.85, 0.6, 1],
  [1, 0.85, 0.35],
  [0.55, 0.85, 1],
  [1, 0.5, 0.65],
];

/** Body colour and radius (px) for the orbit lab: stars warm and large, planets coloured by index. */
export function orbitLook(bodies: readonly Body[], i: number): [number, number, number, number] {
  const m = bodies[i].m;
  const radius = Math.min(11, 2.5 + 1.1 * Math.log10(m * 1e7 + 1));
  const planetIndex = bodies.slice(0, i).filter((b) => b.m < 0.05).length;
  const [r, g, b] = m >= 0.05 ? STAR : PLANET_COLORS[planetIndex % PLANET_COLORS.length];
  return [r, g, b, radius];
}

function galaxyCloud(preset: GalaxyPreset, count: number): Cloud {
  if (preset === "cluster") return plummerCluster(count, 7);
  if (preset === "disk") return diskGalaxy(count, 3);
  return collidingGalaxies(count, 5);
}

const GALAXY_EXTENT: Record<GalaxyPreset, number> = { cluster: 3.5, disk: 4, collision: 9 };

export function createNBody({ gpu, canvas, surface, loop: frameLoop }: SimContext, { onStats, onFrame }: NBodyOptions = {}): NBodyHandle {
  let engine: Engine | null = null;
  let renderer: Renderer | null = null;
  let system: System = { kind: "orbit", bodies: [] };
  let speed = 0.5;
  let integrator: Integrator = "leapfrog";
  let trails = true;
  let playing = false;
  let debt = 0; // simulated time owed to the step loop
  let stepsRequested = 0;

  const view: View = { center: [0, 0], pxPerUnit: 100 };
  let extent = 1.5;

  // Measurement state (reset with every new system).
  let epoch = 0;
  let inFlight = 0;
  let readSeq = 0; // read-backs resolve in order in practice, but are ingested strictly in issue order anyway
  let nextIngest = 0;
  const arrived = new Map<number, Snapshot | null>();
  let lastRead = 0;
  let lastEmit = 0;
  let latest: Snapshot | null = null;
  let e0: number | undefined;
  let l0: number | undefined;
  let trackers: OrbitTracker[] = [];
  let kepler: KeplerPoint[] = [];
  let history = { t: [] as number[], drift: [] as number[] };
  let disposed = false;

  const dt = () => (system.kind === "orbit" ? ORBIT_DT : GALAXY_DT);

  const ensureCapacity = (n: number, trailBodies: number) => {
    if (engine && engine.capacity >= n && engine.trailBodies >= trailBodies) return;
    renderer?.dispose();
    engine?.dispose();
    engine = createEngine(gpu, Math.max(64, n), trailBodies);
    renderer = createRenderer(gpu, engine);
    renderer.setTrails(trails);
  };

  const fitView = () => {
    const [w, h] = surface.size;
    view.pxPerUnit = Math.min(w, h) / 2 / extent;
    view.center = [0, 0];
    if (system.kind === "orbit") {
      const bodies = currentBodies();
      const mass = bodies.reduce((a, b) => a + b.m, 0) || 1;
      view.center = [bodies.reduce((a, b) => a + b.m * b.x, 0) / mass, bodies.reduce((a, b) => a + b.m * b.y, 0) / mass];
    }
  };

  const restartMeasurements = () => {
    epoch++;
    arrived.clear();
    nextIngest = readSeq;
    latest = null;
    e0 = undefined;
    l0 = undefined;
    trackers = system.kind === "orbit" ? system.bodies.map(() => new OrbitTracker()) : [];
    kepler = [];
    history = { t: [], drift: [] };
    lastEmit = 0;
    onStats?.(null);
  };

  const load = () => {
    debt = 0;
    if (system.kind === "orbit") {
      const bodies = toCentreOfMassFrame(system.bodies);
      ensureCapacity(bodies.length, MAX_ORBIT_BODIES);
      const pos = new Float32Array(bodies.length * 4);
      const vel = new Float32Array(bodies.length * 4);
      const looks = new Float32Array(Math.max(64, bodies.length) * 4);
      bodies.forEach((b, i) => {
        pos.set([b.x, b.y, 0, b.m], i * 4);
        vel.set([b.vx, b.vy, 0, 0], i * 4);
        looks.set(orbitLook(bodies, i), i * 4);
      });
      engine!.upload(pos, vel, bodies.length, G_ORBIT, ORBIT_EPS2);
      renderer!.setLooks(looks);
      renderer!.setGlow(0.55);
    } else {
      const cloud = galaxyCloud(system.preset, system.count);
      ensureCapacity(cloud.n, 0);
      const looks = new Float32Array(cloud.n * 4);
      for (let i = 0; i < cloud.n; i++) {
        const g = cloud.group[i];
        looks.set(g === 2 ? [1, 0.92, 0.75, 3.2] : g === 0 ? [0.38, 0.58, 1, 1.4] : [1, 0.6, 0.38, 1.4], i * 4);
      }
      engine!.upload(new Float32Array(cloud.pos), new Float32Array(cloud.vel), cloud.n, 1, GALAXY_EPS2);
      renderer!.setLooks(looks);
      renderer!.setGlow(0.9);
    }
    restartMeasurements();
    readNow(); // the starting energy, before any step is queued
  };

  // ── Measurement: read the state back, derive energies, orbits and measured periods ──
  const ingest = (snap: Snapshot) => {
    latest = snap;
    const c = conserved(snap);
    if (e0 === undefined) {
      e0 = c.total;
      l0 = c.angularMomentum;
    }
    if (system.kind === "orbit") {
      const bodies = bodiesFromState(snap);
      const p = primaryIndex(bodies);
      bodies.forEach((b, i) => {
        if (i === p) return;
        const tracker = trackers[i];
        const before = tracker.orbits;
        tracker.sample(snap.time, Math.atan2(b.y - bodies[p].y, b.x - bodies[p].x));
        if (tracker.orbits > before && tracker.period !== undefined) {
          const el = orbitalElements(b, bodies[p], G_ORBIT);
          if (el.a < Infinity) kepler = [...kepler, { body: i, a: el.a, T: tracker.period }].slice(-60);
        }
      });
    }
  };

  const emit = (now: number) => {
    if (!latest || e0 === undefined) return;
    if (now - lastEmit < STATS_INTERVAL_MS) return;
    lastEmit = now;
    const snap = latest;
    const c = conserved(snap);
    const drift = (c.total - e0) / Math.abs(e0 || 1);
    history.t.push(snap.time);
    history.drift.push(drift);
    if (history.t.length > HISTORY) {
      history.t.shift();
      history.drift.shift();
    }
    let orbits: OrbitReadout[] = [];
    let primary = 0;
    if (system.kind === "orbit") {
      const bodies = bodiesFromState(snap);
      primary = primaryIndex(bodies);
      orbits = bodies.flatMap((b, i) => {
        if (i === primary) return [];
        const el = orbitalElements(b, bodies[primary], G_ORBIT);
        return [{ index: i, r: el.r, v: el.v, a: el.a, e: el.e, energy: el.energy, period: trackers[i].period, orbits: trackers[i].orbits }];
      });
    }
    onStats?.({
      mode: system.kind,
      time: snap.time,
      n: snap.n,
      kinetic: c.kinetic,
      potential: c.potential,
      total: c.total,
      drift,
      angularDrift: l0 ? (c.angularMomentum - l0) / Math.abs(l0) : 0,
      history: { t: [...history.t], drift: [...history.drift] },
      primary,
      orbits,
      kepler,
    });
  };

  function readNow() {
    if (!engine) return;
    const seq = readSeq++;
    const myEpoch = epoch;
    inFlight++;
    lastRead = performance.now();
    const settle = (snap: Snapshot | null) => {
      inFlight--;
      if (myEpoch !== epoch || disposed) return;
      arrived.set(seq, snap);
      while (arrived.has(nextIngest)) {
        const next = arrived.get(nextIngest);
        arrived.delete(nextIngest);
        nextIngest++;
        if (next) ingest(next);
      }
    };
    // A failed read means the device went away (page closing); the GPU error handler reports real faults.
    engine.read().then(settle, () => settle(null));
  }

  /** Keeps a fresh snapshot coming while nothing else asks for one (paused, or galaxy mode). */
  const pump = () => {
    if (!engine || inFlight > 0) return;
    const interval = system.kind === "orbit" ? 0 : GALAXY_READ_INTERVAL_MS;
    if (performance.now() - lastRead >= interval) readNow();
  };

  /** Bodies extrapolated from the latest read-back to the current simulation time (labels keep up with discs). */
  const currentBodies = (): Body[] => {
    if (!latest || !engine) return system.kind === "orbit" ? toCentreOfMassFrame(system.bodies) : [];
    const ahead = engine.time - latest.time;
    return bodiesFromState(latest).map((b) => ({ ...b, x: b.x + b.vx * ahead, y: b.y + b.vy * ahead }));
  };

  const cssToDevice = () => {
    const rect = canvas.getBoundingClientRect();
    return rect.width > 0 ? surface.size[0] / rect.width : 1;
  };

  const loop = frameLoop((frame, frameDt) => {
    if (!engine || !renderer) return;
    let steps = stepsRequested;
    stepsRequested = 0;
    if (playing) {
      debt += frameDt * speed;
      const cap = Math.max(1, Math.min(MAX_STEPS_PER_FRAME, Math.floor(PAIRS_PER_FRAME / Math.max(1, engine.n * engine.n))));
      const due = Math.floor(debt / dt());
      steps += Math.min(due, cap);
      debt = due > cap ? 0 : debt - due * dt();
    }
    if (steps > 0) {
      if (system.kind === "orbit") {
        // Small chunks: a trail point and a measurement sample after each (smooth trails, no aliasing).
        const chunks = Math.max(Math.min(steps, 4), Math.ceil(steps / SAMPLE_STEPS));
        let left = steps;
        for (let c = chunks; c > 0; c--) {
          const k = Math.ceil(left / c);
          engine.step(dt(), k, integrator);
          left -= k;
          if (trails) engine.recordTrail();
          readNow();
        }
      } else {
        engine.step(dt(), steps, integrator);
      }
    }
    pump();
    emit(performance.now());

    const draw = renderer.prepare(surface, view);
    frame.pass({ target: surface, clear: [0.016, 0.02, 0.03, 1] }, draw);
    if (onFrame && system.kind === "orbit") onFrame({ bodies: currentBodies(), view: { center: [...view.center], pxPerUnit: view.pxPerUnit }, pixelRatio: cssToDevice(), time: engine.time });
  });

  const handle: NBodyHandle = {
    setOrbitSystem(bodies, fit = false) {
      const first = !engine;
      system = { kind: "orbit", bodies: bodies.map((b) => ({ ...b })) };
      load();
      if (fit || first) handle.fit();
    },
    setGalaxy(preset, count) {
      system = { kind: "galaxy", preset, count };
      extent = GALAXY_EXTENT[preset];
      load();
      fitView();
    },
    setSpeed(next) {
      speed = next;
    },
    setIntegrator(next) {
      if (next === integrator) return;
      integrator = next;
      load(); // a fair comparison starts from the same state
    },
    setTrails(on) {
      trails = on;
      renderer?.setTrails(on);
      engine?.clearTrails();
    },
    play: () => {
      playing = true;
    },
    pause: () => {
      playing = false;
    },
    step: () => {
      stepsRequested += Math.max(1, Math.round((system.kind === "orbit" ? 0.01 : 0.05) / dt()));
    },
    reset: () => load(),
    panBy(dxCss, dyCss) {
      const k = cssToDevice() / view.pxPerUnit;
      view.center = [view.center[0] - dxCss * k, view.center[1] + dyCss * k];
    },
    zoomAt(factor, xCss, yCss) {
      const [wx, wy] = handle.toWorld(xCss, yCss);
      view.pxPerUnit = Math.min(1e6, Math.max(1e-3, view.pxPerUnit * factor));
      const [nx, ny] = handle.toWorld(xCss, yCss);
      view.center = [view.center[0] + wx - nx, view.center[1] + wy - ny];
    },
    fit() {
      if (system.kind === "orbit") {
        const bodies = currentBodies();
        const mass = bodies.reduce((a, b) => a + b.m, 0) || 1;
        const cx = bodies.reduce((a, b) => a + b.m * b.x, 0) / mass;
        const cy = bodies.reduce((a, b) => a + b.m * b.y, 0) / mass;
        extent = Math.max(1, 1.25 * Math.max(0, ...bodies.map((b) => Math.hypot(b.x - cx, b.y - cy))));
      }
      fitView();
    },
    toWorld(xCss, yCss) {
      const r = cssToDevice();
      const [w, h] = surface.size;
      return [view.center[0] + (xCss * r - w / 2) / view.pxPerUnit, view.center[1] - (yCss * r - h / 2) / view.pxPerUnit];
    },
    currentBodies,
    dispose() {
      disposed = true;
      loop.stop();
      renderer?.dispose();
      engine?.dispose();
    },
  };
  return handle;
}
