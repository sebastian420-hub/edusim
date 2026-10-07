import { compute, storage } from "vgpu";
import type { Gpu, StorageBuffer } from "vgpu";
import type { FractalWindow, Integrator, PendulumParams } from "./pendulum";
import pendulumShader from "./pendulum.wgsl";
import seedShader from "./seed.wgsl";
import trailShader from "./trail.wgsl";
import { SEED_TILE, TRAIL_LEN, workgroupsFor } from "./sim-constants";

// GPUBufferUsage / GPUMapMode flags (the WebGPU globals are not in this project's type environment).
const USAGE_MAP_READ = 0x01;
const USAGE_COPY_DST = 0x08;
const MAP_MODE_READ = 0x01;

/** The public StorageBuffer type hides `destroy()`, which the runtime supports. */
type Destroyable = { destroy?: () => void };

export interface PendulumSnapshot {
  n: number;
  /** Simulation time of the snapshot (s). */
  time: number;
  /** (θ₁, θ₂, ω₁, ω₂) per pendulum. */
  state: Float32Array;
}

/**
 * Double-pendulum physics on the GPU, without drawing: a buffer of pendulums, the integrator kernel, fractal
 * seeding and the trail of pendulum 0. Shared by the simulation and the headless GPU tests.
 */
export function createEngine(gpu: Gpu, capacity: number) {
  const state = storage(gpu, capacity * 16);
  const flip = storage(gpu, capacity * 4);
  const trail = storage(gpu, TRAIL_LEN * 16);
  const integrate = compute(gpu, pendulumShader);
  const seed = compute(gpu, seedShader);
  const record = compute(gpu, trailShader);

  let n = 0;
  let time = 0;
  let params: PendulumParams = { m1: 1, m2: 1, l1: 1, l2: 1, g: 9.81 };
  let head = 0;
  let filled = 0;

  const engine = {
    capacity,
    buffers: { state, flip, trail } as { state: StorageBuffer; flip: StorageBuffer; trail: StorageBuffer },
    get n() {
      return n;
    },
    get time() {
      return time;
    },
    get trailHead() {
      return head;
    },
    get trailFilled() {
      return filled;
    },
    get params() {
      return params;
    },

    setParams(next: PendulumParams) {
      params = next;
    },

    /** Loads pendulums ((θ₁, θ₂, ω₁, ω₂) each) and restarts the clock, flips and trail. */
    upload(states: Float32Array, count: number) {
      if (count > capacity) throw new Error(`engine holds ${capacity} pendulums, got ${count}`);
      n = count;
      time = 0;
      state.write(states.subarray(0, n * 4) as Float32Array<ArrayBuffer>);
      flip.write(new Float32Array(n).fill(-1));
      engine.clearTrail();
    },

    /** Overwrites the states without touching the clock or the trail (the CPU-integrated pendulums, each frame). */
    setStates(states: Float32Array, count: number, at: number) {
      if (count > capacity) throw new Error(`engine holds ${capacity} pendulums, got ${count}`);
      n = count;
      time = at;
      state.write(states.subarray(0, n * 4) as Float32Array<ArrayBuffer>);
    },

    /** Fills a size×size grid, one pendulum per pixel of the window, released from rest (the fractal). */
    seedFractal(size: number, window: FractalWindow) {
      if (size * size > capacity) throw new Error(`engine holds ${capacity} pendulums, got ${size}²`);
      n = size * size;
      time = 0;
      seed.set({ state, flip, params: { n: size, pad: 0, cx: window.cx, cy: window.cy, span: window.span, pad1: 0, pad2: 0, pad3: 0 } });
      seed.dispatch(Math.ceil(size / SEED_TILE), Math.ceil(size / SEED_TILE), 1);
    },

    /** Advances every pendulum by `steps` steps of `dt` seconds. */
    step(dt: number, steps: number, integrator: Integrator = "rk4") {
      if (n === 0 || steps <= 0) return;
      const { m1, m2, l1, l2, g } = params;
      integrate
        .set({ state, flip, params: { n, steps, integrator: integrator === "rk4" ? 0 : 1, pad: 0, dt, time, m1, m2, l1, l2, g, pad2: 0 } })
        .dispatch(workgroupsFor(n));
      time += dt * steps;
    },

    /** Appends pendulum 0's lower bob to the trail. */
    recordTrail() {
      if (n === 0) return;
      record.set({ state, trail, params: { head, pad: 0, l1: params.l1, l2: params.l2 } }).dispatch(1);
      head = (head + 1) % TRAIL_LEN;
      filled = Math.min(filled + 1, TRAIL_LEN);
    },

    clearTrail() {
      trail.write(new Float32Array(TRAIL_LEN * 4));
      head = 0;
      filled = 0;
    },

    async read(): Promise<PendulumSnapshot> {
      const at = time;
      const count = n;
      const buffer = await state.read();
      return { n: count, time: at, state: new Float32Array(buffer, 0, count * 4) };
    },

    /** First-flip times (−1 = not yet), one per pendulum. */
    async readFlips(): Promise<Float32Array> {
      const count = n;
      return new Float32Array(await flip.read(), 0, count);
    },

    /** One pendulum's first-flip time (−1 = not yet), without reading the whole map back. */
    async readFlipAt(index: number): Promise<number> {
      const device = gpu.gpu;
      const staging = device.createBuffer({ size: 4, usage: USAGE_MAP_READ | USAGE_COPY_DST, label: "flip probe" });
      const encoder = device.createCommandEncoder({ label: "flip probe" });
      encoder.copyBufferToBuffer((flip as unknown as { gpu: unknown }).gpu, index * 4, staging, 0, 4);
      device.queue.submit([encoder.finish()]);
      try {
        await staging.mapAsync(MAP_MODE_READ);
        return new Float32Array(staging.getMappedRange().slice(0))[0];
      } finally {
        staging.destroy();
      }
    },

    dispose() {
      for (const b of [state, flip, trail]) (b as unknown as Destroyable).destroy?.();
    },
  };
  return engine;
}

export type Engine = ReturnType<typeof createEngine>;
