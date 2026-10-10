import type { FramePass } from "vgpu";
import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import { DT } from "./cable";
import type { Fibre } from "./cable";
import { createEngine, stimulusWeights } from "./engine";
import type { Engine } from "./engine";
import { axonReadout } from "./measure";
import type { AxonReadout } from "./measure";
import { axonFibre, compartmentAt, fibreLength, NERVE_HEIGHT, NERVE_LENGTH, NERVE_SWEEP_MS, nerveComposition, nerveFibre, nerveStimulus, sweepFor } from "./model";
import type { NerveFibre } from "./model";
import { createRenderer } from "./renderer";
import type { Renderer } from "./renderer";
import type { AxonSettings } from "./settings";

/** Kymograph rows per sweep. */
const HIST_ROWS = 480;
const PROBE_LEN = 4096;
const MAX_STEPS_PER_FRAME = 60;
const READ_INTERVAL_MS = 150;
/** Real-time pause between sweeps (ms), so the finished picture and its readouts can be seen. */
const HOLD_MS = 1500;
/** The nerve runs this many times faster than the slow-motion setting (its sweep is 40 ms). */
const NERVE_SPEEDUP = 4;
const TRACE_POINTS = 300;

export interface Traces {
  t: number[];
  v1: number[];
  v2: number[];
}

export interface CapReadout {
  t: number[];
  v: number[];
  /** Latency of the fast (A) and slow (C) waves at the electrode, ms, when present. */
  aLatency?: number;
  cLatency?: number;
}

export type AxonStats =
  | { view: "axon"; time: number; sweepMs: number; sweeps: number; live: AxonReadout; done: AxonReadout | null; traces: Traces }
  | { view: "nerve"; time: number; sweepMs: number; sweeps: number; fibres: number; cap: CapReadout; done: CapReadout | null };

export interface FrameInfo {
  view: "axon" | "nerve";
  /** Fibre length (cm) and, for the Axon view, the compartment positions of nodes (myelinated) for the overlay. */
  length: number;
  sweepMs: number;
  time: number;
  composition?: NerveFibre[];
}

export interface AxonOptions {
  onStats?: (stats: AxonStats | null) => void;
  onFrame?: (info: FrameInfo) => void;
}

export interface AxonHandle extends SimHandle {
  configure(settings: AxonSettings): void;
  setSpeed(msPerSecond: number): void;
  play(): void;
  pause(): void;
  /** Advances a little (paused). */
  step(): void;
  /** Starts the sweep again from rest. */
  reset(): void;
}

/** Settings that change the fibres themselves (the engine is rebuilt) vs only the stimuli and recording. */
const geometryKey = (s: AxonSettings) =>
  s.view === "axon" ? JSON.stringify(["axon", s.diameter, s.myelin, s.from, s.to, s.drug, s.myelinLeft]) : JSON.stringify(["nerve", s.fibres]);
const sweepKey = (s: AxonSettings) => JSON.stringify([s.temp, s.stim, s.pulses, s.gap, s.e1, s.e2, s.nerveStim, s.distance]);

/** Points of a ring of samples, oldest first, decimated to at most `max`. */
function decimate<T>(n: number, max: number, at: (i: number) => T): T[] {
  const stride = Math.max(1, Math.ceil(n / max));
  const out: T[] = [];
  for (let i = 0; i < n; i += stride) out.push(at(i));
  return out;
}

/** The stimulus (0.5–0.7 ms) and its artifact are blanked from the recording, as an amplifier would. */
export const BLANK_UNTIL = 0.8;

/** Fast (A) and slow (C) wave latencies: the largest deflection before and after 6 ms (after the blanking). */
export function capLatencies(t: readonly number[], v: readonly number[]): { aLatency?: number; cLatency?: number } {
  const scale = Math.max(1e-12, ...v.filter((_, i) => t[i] >= BLANK_UNTIL).map(Math.abs));
  const peakIn = (lo: number, hi: number) => {
    let best = -1;
    for (let i = 0; i < t.length; i++) if (t[i] >= lo && t[i] < hi && (best < 0 || Math.abs(v[i]) > Math.abs(v[best]))) best = i;
    return best >= 0 && Math.abs(v[best]) > 0.02 * scale ? t[best] : undefined;
  };
  return { aLatency: peakIn(BLANK_UNTIL, 6), cLatency: peakIn(6, Infinity) };
}

export function createAxonSim({ gpu, surface, loop: frameLoop }: SimContext, { onStats, onFrame }: AxonOptions = {}): AxonHandle {
  let engine: Engine | null = null;
  let renderer: Renderer | null = null;
  let settings: AxonSettings | null = null;
  let fibre: Fibre | null = null;
  let composition: NerveFibre[] = [];
  let geometry = "";
  let sweepSettings = "";
  let sweepMs = 10;
  let histEvery = 1;
  let speed = 2;
  let playing = false;
  let debt = 0;
  let stepsRequested = 0;
  let holdUntil = 0;
  let sweeps = 0;
  let epoch = 0;
  let reading = false;
  let lastRead = 0;
  let disposed = false;
  let live: AxonReadout | null = null;
  let traces: Traces = { t: [], v1: [], v2: [] };
  let cap: CapReadout = { t: [], v: [] };
  let done: AxonReadout | CapReadout | null = null;
  let finishing = false;

  const nerve = () => settings?.view === "nerve";
  const probes = (): [number, number] => (fibre && settings ? [compartmentAt(fibre, Math.min(settings.e1, settings.e2)), compartmentAt(fibre, Math.max(settings.e1, settings.e2))] : [0, 0]);

  const rebuild = (s: AxonSettings) => {
    renderer = null;
    engine?.dispose();
    if (s.view === "axon") {
      fibre = axonFibre(s);
      composition = [];
      engine = createEngine(gpu, [fibre], { histRows: HIST_ROWS, probeLen: PROBE_LEN, maxSteps: 64, capLen: 16 });
    } else {
      composition = nerveComposition(s.fibres);
      const fibres = composition.map(nerveFibre);
      fibre = fibres[0];
      engine = createEngine(gpu, fibres, { histRows: 1, probeLen: 16, maxSteps: 64, capLen: Math.ceil(NERVE_SWEEP_MS / DT) + 64 });
    }
    renderer = createRenderer(gpu, engine);
  };

  /** Starts a sweep from rest with the current stimuli. */
  const restart = () => {
    if (!engine || !settings) return;
    const s = settings;
    epoch++;
    engine.reset();
    debt = 0;
    finishing = false;
    live = null;
    traces = { t: [], v1: [], v2: [] };
    cap = { t: [], v: [] };
    if (s.view === "axon" && fibre) {
      const sweep = sweepFor(s, fibre);
      sweepMs = sweep.ms;
      histEvery = Math.max(1, Math.ceil(sweepMs / DT / HIST_ROWS));
      // Electrode B: the far end ("both ends"), or electrode A again for the second pulse of a pair.
      const b = sweep.b;
      engine.setStimulus(stimulusWeights([fibre], () => sweep.a), b ? stimulusWeights([fibre], () => b) : null, sweep.a.startMs, b ? b.startMs : -1, sweep.a.durationMs);
    } else {
      sweepMs = NERVE_SWEEP_MS;
      histEvery = 0;
      const fibres = engine.fibres;
      engine.setStimulus(stimulusWeights(fibres, (f) => nerveStimulus(f, s.nerveStim)), null, 0.5, -1, 0.2);
    }
    emit(true);
  };

  const emit = (force = false) => {
    if (!settings || !engine) return;
    if (!force && !live && !cap.t.length) return;
    if (settings.view === "axon") {
      onStats?.({
        view: "axon",
        time: engine.time,
        sweepMs,
        sweeps,
        live: live ?? axonReadout(fibre!, new Float32Array(fibre!.n).fill(-1), new Float32Array(fibre!.n), settings.e1, settings.e2),
        done: (done as AxonReadout | null) ?? null,
        traces,
      });
    } else {
      onStats?.({ view: "nerve", time: engine.time, sweepMs, sweeps, fibres: engine.fibres.length, cap, done: (done as CapReadout | null) ?? null });
    }
  };

  /** Reads the sweep's records back; `final` marks the end of a sweep. */
  const read = (final: boolean) => {
    if (!engine || !settings || reading) return;
    reading = true;
    lastRead = performance.now();
    const myEpoch = epoch;
    const e = engine;
    const s = settings;
    const steps = e.stepIndex;
    const L = e.layout;
    const job =
      s.view === "axon"
        ? Promise.all([e.readDyn(), e.readOut(L.probeOffset, 2 * PROBE_LEN)]).then(([d, p]) => {
            const n = fibre!.n;
            const arrival = new Float32Array(n);
            const crossings = new Float32Array(n);
            for (let i = 0; i < n; i++) {
              arrival[i] = d.data[i * 8 + 6];
              crossings[i] = d.data[i * 8 + 7];
            }
            const count = Math.min(steps, PROBE_LEN);
            const first = steps - count; // ring slot of step k is (k + 1) % len; samples steps first+1 … steps
            const sample = (k: number) => {
              const slot = (first + k + 1) % PROBE_LEN;
              return [p[2 * slot], p[2 * slot + 1]];
            };
            let peak1 = -Infinity;
            let peak2 = -Infinity;
            for (let k = 0; k < count; k++) {
              const [a, b] = sample(k);
              peak1 = Math.max(peak1, a);
              peak2 = Math.max(peak2, b);
            }
            const pts = decimate(count, TRACE_POINTS, (k) => [(first + k + 1) * DT, ...sample(k)]);
            return { readout: axonReadout(fibre!, arrival, crossings, s.e1, s.e2, count > 0 ? [peak1, peak2] : undefined), traces: { t: pts.map((q) => q[0]), v1: pts.map((q) => q[1]), v2: pts.map((q) => q[2]) } };
          })
        : e.readOut(L.capOffset, Math.min(steps, L.capLen)).then((c) => {
            const pts = decimate(c.length, 400, (k) => [(k + 1) * DT, (k + 1) * DT < BLANK_UNTIL ? 0 : c[k]]);
            const t = pts.map((q) => q[0]);
            const v = pts.map((q) => q[1]);
            return { cap: { t, v, ...capLatencies(t, v) } };
          });
    job.then(
      (r) => {
        reading = false;
        if (disposed || myEpoch !== epoch) return;
        if ("readout" in r) {
          live = r.readout;
          traces = r.traces;
          if (final) done = r.readout;
        } else {
          cap = r.cap;
          if (final) done = r.cap;
        }
        if (final) sweeps++;
        emit(true);
      },
      () => {
        reading = false;
      },
    );
  };

  const loop = frameLoop((frame, frameDt) => {
    if (!engine || !renderer || !settings) return;
    const now = performance.now();
    let steps = stepsRequested;
    stepsRequested = 0;
    if (playing && !finishing) {
      debt += frameDt * speed * (nerve() ? NERVE_SPEEDUP : 1);
      const due = Math.floor(debt / DT);
      steps += Math.min(due, MAX_STEPS_PER_FRAME);
      debt = due > MAX_STEPS_PER_FRAME ? 0 : debt - due * DT;
    }
    const left = Math.round((sweepMs - engine.time) / DT);
    steps = Math.min(steps, Math.max(0, left));
    if (steps > 0) {
      engine.step(steps, {
        celsius: settings.temp,
        dt: DT,
        histEvery: settings.view === "axon" ? histEvery : 0,
        probes: settings.view === "axon" ? probes() : undefined,
        electrode: settings.view === "nerve" ? { x: settings.distance, height: NERVE_HEIGHT } : undefined,
      });
    }
    if (engine.time >= sweepMs - DT / 2) {
      if (!finishing) {
        finishing = true;
        holdUntil = now + HOLD_MS;
        reading = false;
        read(true);
      } else if (playing && now >= holdUntil && !reading) {
        restart();
      }
    } else if (now - lastRead >= READ_INTERVAL_MS && (steps > 0 || !live)) {
      read(false);
    }

    let draw: (pass: FramePass) => void;
    if (settings.view === "axon" && fibre) {
      draw = renderer.axon(surface, { length: fibreLength(fibre), sweepMs, histEvery, rowsDone: Math.floor(engine.stepIndex / histEvery), dt: DT, myelinated: fibre.myelinated });
    } else {
      draw = renderer.nerve(surface, NERVE_LENGTH);
    }
    frame.pass({ target: surface, clear: [0.016, 0.02, 0.03, 1] }, draw);
    onFrame?.({ view: settings.view, length: settings.view === "axon" && fibre ? fibreLength(fibre) : NERVE_LENGTH, sweepMs, time: engine.time, composition: settings.view === "nerve" ? composition : undefined });
  });

  const handle: AxonHandle = {
    configure(next) {
      const g = geometryKey(next);
      const k = sweepKey(next);
      const changed = g !== geometry || k !== sweepSettings || next.view !== settings?.view;
      settings = next;
      if (g !== geometry) {
        geometry = g;
        rebuild(next);
      }
      sweepSettings = k;
      if (changed) {
        done = null;
        onStats?.(null);
        restart();
      }
    },
    setSpeed(next) {
      speed = next;
    },
    play: () => {
      playing = true;
    },
    pause: () => {
      playing = false;
    },
    step: () => {
      stepsRequested += Math.max(1, Math.round((nerve() ? 1 : 0.25) / DT));
    },
    reset: () => restart(),
    dispose() {
      disposed = true;
      loop.stop();
      engine?.dispose();
    },
  };
  return handle;
}
