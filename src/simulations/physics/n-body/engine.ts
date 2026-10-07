import { compute, storage } from "vgpu";
import type { Gpu, StorageBuffer } from "vgpu";
import gravityShader from "./gravity.wgsl";
import integrateShader from "./integrate.wgsl";
import trailShader from "./trail.wgsl";
import type { Integrator } from "./nbody";
import { TRAIL_LEN, workgroupsFor } from "./sim-constants";

/** The public StorageBuffer type hides `destroy()`, which the runtime supports. */
type Destroyable = { destroy?: () => void };

export interface Snapshot {
  n: number;
  /** Simulation time when the snapshot was taken. */
  time: number;
  pos: Float32Array;
  vel: Float32Array;
  acc: Float32Array;
}

/**
 * The N-body physics on the GPU, without any drawing: body buffers, the force pass and the integrator.
 * Used by the simulation and, unchanged, by the headless GPU tests that compare it with the CPU twin.
 */
export function createEngine(gpu: Gpu, capacity: number, trailBodies: number) {
  const pos = storage(gpu, capacity * 16);
  const vel = storage(gpu, capacity * 16);
  const acc = storage(gpu, capacity * 16);
  const trail = storage(gpu, Math.max(1, trailBodies) * TRAIL_LEN * 16);
  const force = compute(gpu, gravityShader);
  const integrate = compute(gpu, integrateShader);
  const record = compute(gpu, trailShader);

  let n = 0;
  let G = 1;
  let eps2 = 1e-6;
  let time = 0;
  let head = 0;
  let filled = 0;

  const forcePass = () => {
    force.set({ pos, acc, params: { n, pad: 0, g: G, eps2 } }).dispatch(workgroupsFor(n));
  };
  const integratePass = (mode: number, dt: number) => {
    integrate.set({ pos, vel, acc, params: { n, mode, dt, pad: 0 } }).dispatch(workgroupsFor(n));
  };

  const engine = {
    capacity,
    trailBodies,
    buffers: { pos, vel, acc, trail } as { pos: StorageBuffer; vel: StorageBuffer; acc: StorageBuffer; trail: StorageBuffer },
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

    /** Loads bodies (vec4 per body, as in nbody.ts) and computes their accelerations. */
    upload(positions: Float32Array, velocities: Float32Array, count: number, gravity: number, softening2: number) {
      if (count > capacity) throw new Error(`engine holds ${capacity} bodies, got ${count}`);
      n = count;
      G = gravity;
      eps2 = softening2;
      time = 0;
      pos.write(positions.subarray(0, n * 4) as Float32Array<ArrayBuffer>);
      vel.write(velocities.subarray(0, n * 4) as Float32Array<ArrayBuffer>);
      forcePass();
      engine.clearTrails();
    },

    setSoftening(softening2: number) {
      eps2 = softening2;
      forcePass();
    },

    /** Advances `steps` steps of `dt`. Leapfrog: half-kick + drift, forces, half-kick. Euler: step, forces. */
    step(dt: number, steps = 1, integrator: Integrator = "leapfrog") {
      if (n === 0) return;
      for (let k = 0; k < steps; k++) {
        if (integrator === "leapfrog") {
          integratePass(0, dt);
          forcePass();
          integratePass(1, dt);
        } else {
          integratePass(2, dt);
          forcePass();
        }
        time += dt;
      }
    },

    /** Appends the current positions of the first `trailBodies` bodies to their trails. */
    recordTrail() {
      const count = Math.min(n, trailBodies);
      if (count === 0) return;
      record.set({ pos, trail, params: { count, len: TRAIL_LEN, head, pad: 0 } }).dispatch(workgroupsFor(count));
      head = (head + 1) % TRAIL_LEN;
      filled = Math.min(filled + 1, TRAIL_LEN);
    },

    clearTrails() {
      trail.write(new Float32Array(Math.max(1, trailBodies) * TRAIL_LEN * 4));
      head = 0;
      filled = 0;
    },

    /** Reads the current state back (all work queued so far is included). */
    async read(): Promise<Snapshot> {
      const at = time;
      const count = n;
      const [p, v, a] = await Promise.all([pos.read(), vel.read(), acc.read()]);
      return {
        n: count,
        time: at,
        pos: new Float32Array(p, 0, count * 4),
        vel: new Float32Array(v, 0, count * 4),
        acc: new Float32Array(a, 0, count * 4),
      };
    },

    dispose() {
      for (const b of [pos, vel, acc, trail]) (b as unknown as Destroyable).destroy?.();
    },
  };
  return engine;
}

export type Engine = ReturnType<typeof createEngine>;
