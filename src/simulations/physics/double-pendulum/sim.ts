import type { FramePass } from "vgpu";
import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import { createEngine } from "./engine";
import type { Engine } from "./engine";
import { SpreadObserver, SwingObserver } from "./measure";
import type { SpreadStats, SwingStats } from "./measure";
import { crowdAngles, positions, separation, step } from "./pendulum";
import type { FractalWindow, Integrator, PendulumParams, State } from "./pendulum";
import { createRenderer } from "./renderer";
import type { Renderer, View } from "./renderer";
import type { PendulumView } from "./settings";

/** Time step of the pendulum and butterfly views (s): RK4 keeps the energy error below 10⁻⁶ for a minute. */
export const DT = 1e-3;
/** Time step of the fractal (s): coarser, so a million pendulums develop in seconds. */
export const FRACTAL_DT = 4e-3;
/** How long the fractal map runs (s of simulated time per pendulum). */
export const FRACTAL_T_MAX = 30;
/** Fractal budget: pendulum-steps per frame (≈ 2 GFLOP: a few ms on integrated graphics). */
const FRACTAL_BUDGET = 6e6;
const MAX_STEPS_PER_FRAME = 200;
/** A trail point and a measurement sample every this many steps. */
const SAMPLE_STEPS = 8;
const STATS_INTERVAL_MS = 100;
const GPU_READ_INTERVAL_MS = 250;

export interface CrowdStats extends SpreadStats {
  n: number;
  /** Gap between the GPU's 32-bit copy of pendulum 0 and the CPU's 64-bit one (rad), from the last read-back. */
  precisionGap?: number;
  /** When that gap first passed 1 rad. */
  precisionPartedAt?: number;
}

export interface FractalStats {
  time: number;
  tMax: number;
  size: number;
}

export type PendulumStats =
  | { view: "pendulum"; swing: SwingStats }
  | { view: "butterfly"; swing: SwingStats; crowd: CrowdStats }
  | { view: "fractal"; fractal: FractalStats };

export interface PendulumConfig {
  view: PendulumView;
  /** Release angles (rad), from rest. */
  start: [number, number];
  params: PendulumParams;
  integrator: Integrator;
  count: number;
  /** Gap between neighbouring pendulums of the butterfly crowd (rad). */
  nudge: number;
  fractalSize: number;
  window: FractalWindow;
  boundary: boolean;
}

export interface PendulumOptions {
  onStats?: (stats: PendulumStats | null) => void;
}

export interface PendulumHandle extends SimHandle {
  /** Applies settings; restarts the motion only when something it depends on changed. */
  configure(config: PendulumConfig): void;
  setSpeed(speed: number): void;
  setTrail(on: boolean): void;
  play(): void;
  pause(): void;
  step(): void;
  reset(): void;
  /** Canvas CSS pixels → world metres (pivot at the origin, y up) and back. */
  toWorld(xCss: number, yCss: number): [number, number];
  toScreen(x: number, y: number): [number, number];
  /** Where the bobs are drawn now (world metres). */
  bobs(): { x1: number; y1: number; x2: number; y2: number };
  /** Fractal: canvas CSS pixels → position in the square map, (0, 0) top-left to (1, 1); may lie outside. */
  mapUV(xCss: number, yCss: number): [number, number];
  /** Fractal: the first-flip time of the pixel under a canvas point (−1 = not yet), or null outside the map. */
  probe(xCss: number, yCss: number): Promise<number | null>;
}

const sameConfig = (a: PendulumConfig | null, b: PendulumConfig, keys: (keyof PendulumConfig)[]) =>
  !!a && keys.every((k) => JSON.stringify(a[k]) === JSON.stringify(b[k]));

export function createDoublePendulum({ gpu, canvas, surface, loop: frameLoop }: SimContext, { onStats }: PendulumOptions = {}): PendulumHandle {
  // The pendulums integrated on the CPU in 64-bit (the one in the Pendulum view, the measured pair in the Butterfly
  // view) are only drawn by the GPU; the crowd and the fractal are integrated on the GPU in 32-bit.
  const solo = createEngine(gpu, 2);
  const soloRenderer = createRenderer(gpu, solo);
  let crowd: Engine | null = null;
  let crowdRenderer: Renderer | null = null;
  let map: Engine | null = null;
  let mapRenderer: Renderer | null = null;

  let config: PendulumConfig | null = null;
  let speed = 1;
  let trail = true;
  let playing = false;
  let debt = 0;
  let stepsRequested = 0;
  let disposed = false;

  // CPU state.
  let time = 0;
  let a: State = [0, 0, 0, 0];
  let b: State = [0, 0, 0, 0];
  let swing: SwingObserver | null = null;
  let spread: SpreadObserver | null = null;
  let epoch = 0;
  let lastEmit = 0;
  let lastRecord = 0;
  // GPU read-backs (butterfly: pendulum 0 in 32-bit).
  let reading = false;
  let lastRead = 0;
  let precisionGap: number | undefined;
  let precisionPartedAt: number | undefined;

  const view: View = { center: [0, 0], pxPerUnit: 100 };
  const fit = () => {
    const [w, h] = surface.size;
    const reach = (config?.params.l1 ?? 1) + (config?.params.l2 ?? 1);
    view.pxPerUnit = (Math.min(w, h) / 2 / reach) * 0.88;
    view.center = [0, 0];
  };

  const ensureCrowd = (n: number) => {
    if (crowd && crowd.capacity >= n) return crowd;
    crowdRenderer?.dispose();
    crowd?.dispose();
    crowd = createEngine(gpu, Math.max(64, n));
    crowdRenderer = createRenderer(gpu, crowd);
    return crowd;
  };
  const ensureMap = (size: number) => {
    if (map && map.capacity >= size * size) return map;
    mapRenderer?.dispose();
    map?.dispose();
    map = createEngine(gpu, size * size);
    mapRenderer = createRenderer(gpu, map);
    return map;
  };

  const uploadSolo = () => {
    const pair = config?.view === "butterfly" ? 2 : 1;
    solo.setStates(new Float32Array(pair === 2 ? [...a, ...b] : a), pair, time);
  };

  const load = () => {
    if (!config) return;
    const c = config;
    epoch++;
    debt = 0;
    time = 0;
    precisionGap = undefined;
    precisionPartedAt = undefined;
    lastEmit = 0;
    lastRecord = 0;
    solo.setParams(c.params);
    if (c.view === "fractal") {
      const engine = ensureMap(c.fractalSize);
      engine.setParams(c.params);
      engine.seedFractal(c.fractalSize, c.window);
      swing = null;
      spread = null;
    } else {
      a = [c.start[0], c.start[1], 0, 0];
      b = [c.start[0] + c.nudge, c.start[1], 0, 0];
      swing = new SwingObserver(a, c.params);
      spread = c.view === "butterfly" ? new SpreadObserver() : null;
      solo.upload(new Float32Array(c.view === "butterfly" ? [...a, ...b] : a), c.view === "butterfly" ? 2 : 1);
      if (c.view === "butterfly") {
        const engine = ensureCrowd(c.count);
        engine.setParams(c.params);
        const states = new Float32Array(c.count * 4);
        const angles = crowdAngles(c.start[0], c.nudge, c.count);
        for (let i = 0; i < c.count; i++) states.set([angles[i], c.start[1], 0, 0], i * 4);
        engine.upload(states, c.count);
      }
      if (trail) solo.recordTrail();
    }
    onStats?.(null);
    emit(0, true);
  };

  const emit = (now: number, force = false) => {
    if (!config || (!force && now - lastEmit < STATS_INTERVAL_MS)) return;
    lastEmit = now;
    if (config.view === "fractal") {
      onStats?.({ view: "fractal", fractal: { time: map?.time ?? 0, tMax: FRACTAL_T_MAX, size: config.fractalSize } });
      return;
    }
    if (!swing) return;
    if (config.view === "pendulum") onStats?.({ view: "pendulum", swing: swing.stats() });
    else if (spread) onStats?.({ view: "butterfly", swing: swing.stats(), crowd: { ...spread.stats(), n: config.count, precisionGap, precisionPartedAt } });
  };

  /** Butterfly: compares the GPU's 32-bit pendulum 0 with the CPU's 64-bit one at the same moment. */
  const readCrowd = () => {
    if (!crowd || reading) return;
    reading = true;
    lastRead = performance.now();
    const myEpoch = epoch;
    const reference = a;
    const at = time;
    crowd.read().then(
      (snap) => {
        reading = false;
        if (disposed || myEpoch !== epoch || snap.n < 1) return;
        const gap = separation([snap.state[0], snap.state[1], snap.state[2], snap.state[3]], reference);
        precisionGap = gap;
        if (precisionPartedAt === undefined && gap > 1) precisionPartedAt = at;
      },
      () => {
        reading = false;
      },
    );
  };

  const advance = (steps: number) => {
    if (!config || steps <= 0) return;
    const c = config;
    if (c.view === "fractal") {
      if (!map || map.time >= FRACTAL_T_MAX) return;
      map.step(FRACTAL_DT, Math.min(steps, Math.ceil((FRACTAL_T_MAX - map.time) / FRACTAL_DT - 1e-9)), c.integrator);
      return;
    }
    // CPU pendulums in chunks: a trail point and a measurement sample after each.
    let left = steps;
    while (left > 0) {
      const k = Math.min(left, SAMPLE_STEPS);
      for (let i = 0; i < k; i++) {
        a = step(a, DT, c.params, c.integrator);
        if (spread) b = step(b, DT, c.params, c.integrator);
        time += DT;
        swing?.sample(time, a); // every step: flips and zero crossings are timed to the millisecond
      }
      spread?.sample(time, a, b);
      left -= k;
      if (trail) {
        uploadSolo();
        solo.recordTrail();
      }
    }
    if (c.view === "butterfly") crowd?.step(DT, steps, c.integrator);
  };

  const fractalSteps = () => {
    const n = (config?.fractalSize ?? 512) ** 2;
    return Math.max(1, Math.min(64, Math.floor(FRACTAL_BUDGET / n)));
  };

  const loop = frameLoop((frame, frameDt) => {
    if (!config) return;
    const c = config;
    let steps = stepsRequested;
    stepsRequested = 0;
    if (playing) {
      if (c.view === "fractal") {
        steps += fractalSteps();
      } else {
        debt += frameDt * speed;
        const due = Math.floor(debt / DT);
        steps += Math.min(due, MAX_STEPS_PER_FRAME);
        debt = due > MAX_STEPS_PER_FRAME ? 0 : debt - due * DT;
      }
    }
    advance(steps);
    const now = performance.now();
    if (c.view !== "fractal") {
      uploadSolo();
      if (steps > 0 && now - lastRecord >= 50) {
        lastRecord = now;
        swing?.record();
        spread?.record();
      }
      if (c.view === "butterfly" && now - lastRead >= GPU_READ_INTERVAL_MS) readCrowd();
    }
    emit(now);

    fit();
    let draw: ((pass: FramePass) => void)[];
    if (c.view === "fractal") {
      draw = map && mapRenderer ? [mapRenderer.prepare(surface, view, { kind: "fractal", size: c.fractalSize, window: c.window, tMax: FRACTAL_T_MAX, boundary: c.boundary }, false)] : [];
    } else {
      draw = [];
      if (c.view === "butterfly" && crowdRenderer && crowd) draw.push(crowdRenderer.prepare(surface, view, { kind: "pendulums", crowd: true }, false));
      draw.push(soloRenderer.prepare(surface, view, { kind: "pendulums", crowd: false, color: c.view === "butterfly" ? [1, 1, 1, 0.9] : undefined }, trail));
    }
    frame.pass({ target: surface, clear: [0.016, 0.02, 0.03, 1] }, (pass) => {
      for (const d of draw) d(pass);
    });
  });

  const cssToDevice = () => {
    const rect = canvas.getBoundingClientRect();
    return rect.width > 0 ? surface.size[0] / rect.width : 1;
  };

  const handle: PendulumHandle = {
    configure(next) {
      const prev = config;
      config = next;
      // The fractal does not depend on the release angles or the crowd; the pendulums not on the window.
      const keys: (keyof PendulumConfig)[] =
        next.view === "fractal" ? ["view", "params", "integrator", "fractalSize", "window"] : next.view === "butterfly" ? ["view", "start", "params", "integrator", "count", "nudge"] : ["view", "start", "params", "integrator"];
      if (!sameConfig(prev, next, keys)) load();
    },
    setSpeed(next) {
      speed = next;
    },
    setTrail(on) {
      trail = on;
      solo.clearTrail();
      if (on) solo.recordTrail();
    },
    play: () => {
      playing = true;
    },
    pause: () => {
      playing = false;
    },
    step: () => {
      stepsRequested += config?.view === "fractal" ? fractalSteps() : Math.round(0.05 / DT);
    },
    reset: () => load(),
    toWorld(xCss, yCss) {
      const r = cssToDevice();
      const [w, h] = surface.size;
      return [view.center[0] + (xCss * r - w / 2) / view.pxPerUnit, view.center[1] - (yCss * r - h / 2) / view.pxPerUnit];
    },
    toScreen(x, y) {
      const r = cssToDevice();
      const [w, h] = surface.size;
      return [((x - view.center[0]) * view.pxPerUnit + w / 2) / r, (h / 2 - (y - view.center[1]) * view.pxPerUnit) / r];
    },
    bobs: () => positions(a, config?.params ?? { l1: 1, l2: 1 }),
    mapUV(xCss, yCss) {
      const r = cssToDevice();
      const [w, h] = surface.size;
      const side = Math.min(w, h);
      return [(xCss * r - (w - side) / 2) / side, (yCss * r - (h - side) / 2) / side];
    },
    async probe(xCss, yCss) {
      if (!config || config.view !== "fractal" || !map) return null;
      const [u, v] = handle.mapUV(xCss, yCss);
      if (u < 0 || v < 0 || u >= 1 || v >= 1) return null;
      const n = config.fractalSize;
      const i = Math.min(n - 1, Math.floor(u * n));
      const j = Math.min(n - 1, Math.floor(v * n));
      try {
        return await map.readFlipAt(j * n + i);
      } catch {
        return null;
      }
    },
    dispose() {
      disposed = true;
      loop.stop();
      soloRenderer.dispose();
      solo.dispose();
      crowdRenderer?.dispose();
      crowd?.dispose();
      mapRenderer?.dispose();
      map?.dispose();
    },
  };
  return handle;
}
