import { compute, storage } from "vgpu";
import type { Gpu, StorageBuffer } from "vgpu";
import { temperatureFactor } from "../hodgkin-huxley/hh";
import { REST_V, restingGates } from "./cable";
import type { Fibre, Stimulus } from "./cable";
import cableShader from "./cable.wgsl";
import capShader from "./cap.wgsl";

const WORKGROUP = 64;
// GPUBufferUsage / GPUMapMode flags (the WebGPU globals are not in this project's type environment).
const USAGE_MAP_READ = 0x01;
const USAGE_COPY_DST = 0x08;
const MAP_MODE_READ = 0x01;

type Destroyable = { destroy?: () => void };

export interface EngineOptions {
  /** Kymograph rows kept for fibre 0. */
  histRows?: number;
  /** Voltage-trace samples kept (two probes, one sample per step). */
  probeLen?: number;
  /** Steps per dispatch (the per-fibre signal buffer holds this many steps). */
  maxSteps?: number;
  /** Compound action potential samples kept (one per step). */
  capLen?: number;
}

/** Flat arrays for the GPU (and the tests): fibre table and per-compartment constants, as cable.wgsl reads them. */
export function packFibres(fibres: readonly Fibre[]): { table: Uint32Array; comps: Float32Array; total: number } {
  const total = fibres.reduce((a, f) => a + f.n, 0);
  const table = new Uint32Array(fibres.length * 4);
  const comps = new Float32Array(total * 8);
  let at = 0;
  fibres.forEach((f, k) => {
    table.set([at, f.n, 0, 0], k * 4);
    for (let i = 0; i < f.n; i++) {
      const A = f.area[i];
      comps.set([f.cm[i] * A, i < f.n - 1 ? f.axial[i] : 0, f.x[i], 0, f.gNa[i] * A, f.gK[i] * A, f.gL[i] * A, 0], (at + i) * 8);
    }
    at += f.n;
  });
  return { table, comps, total };
}

/** Per-compartment stimulus currents (µA) for all fibres concatenated, from one pulse per fibre (or none). */
export function stimulusWeights(fibres: readonly Fibre[], pulse: (f: Fibre, k: number) => Stimulus | null): Float32Array {
  const w = new Float32Array(fibres.reduce((a, f) => a + f.n, 0));
  let at = 0;
  fibres.forEach((f, k) => {
    const p = pulse(f, k);
    if (p) for (let i = p.first; i <= p.last; i++) w[at + i] = p.amplitude / (p.last - p.first + 1);
    at += f.n;
  });
  return w;
}

export interface StepOptions {
  celsius: number;
  dt: number;
  /** Kymograph row every this many steps (0 = off). */
  histEvery?: number;
  /** Fibre-0 compartments traced every step. */
  probes?: [number, number];
  /** Record the compound action potential at this electrode. */
  electrode?: { x: number; height: number };
}

/**
 * Hodgkin–Huxley cables on the GPU, without drawing: shared by the simulation and the headless GPU tests.
 * Rebuilt whenever the set of fibres changes (geometry, drugs, damage); stimuli can change without a rebuild.
 */
export function createEngine(gpu: Gpu, fibres: readonly Fibre[], options: EngineOptions = {}) {
  const histRows = options.histRows ?? 512;
  const probeLen = options.probeLen ?? 4096;
  const maxSteps = options.maxSteps ?? 64;
  const capLen = options.capLen ?? 8192;
  const F = fibres.length;
  const { table, comps, total } = packFibres(fibres);
  const n0 = fibres[0]?.n ?? 0;
  const histOffset = 0;
  const probeOffset = histOffset + histRows * n0;
  const sigOffset = probeOffset + 2 * probeLen;
  const capOffset = sigOffset + maxSteps * F;
  const outSize = capOffset + capLen;

  const fibreBuffer = storage(gpu, Math.max(16, table.byteLength));
  fibreBuffer.write(table as Uint32Array<ArrayBuffer>);
  const compBuffer = storage(gpu, Math.max(32, comps.byteLength));
  const dyn = storage(gpu, Math.max(32, total * 32));
  const out = storage(gpu, outSize * 4);
  const cable = compute(gpu, cableShader);
  const cap = compute(gpu, capShader);

  let time = 0;
  let stepIndex = 0;
  let capHead = 0;
  let capFilled = 0;
  let stim = { startA: -1, startB: -1, duration: 0.2 };

  const engine = {
    fibres,
    total,
    layout: { histRows, histOffset, probeLen, probeOffset, capLen, capOffset, n0 },
    buffers: { out, dyn, comps: compBuffer, fibres: fibreBuffer } as { out: StorageBuffer; dyn: StorageBuffer; comps: StorageBuffer; fibres: StorageBuffer },
    get time() {
      return time;
    },
    get stepIndex() {
      return stepIndex;
    },
    get capHead() {
      return capHead;
    },
    get capFilled() {
      return capFilled;
    },

    /**
     * Sets the stimulus currents per compartment (µA, all fibres concatenated) of electrodes A and B, and when they
     * fire (ms, negative = never). They stay off unless their start time lies ahead of the clock.
     */
    setStimulus(weightsA: Float32Array | null, weightsB: Float32Array | null, startA: number, startB: number, duration = 0.2) {
      for (let i = 0; i < total; i++) {
        comps[i * 8 + 3] = weightsA ? weightsA[i] : 0;
        comps[i * 8 + 7] = weightsB ? weightsB[i] : 0;
      }
      compBuffer.write(comps as Float32Array<ArrayBuffer>);
      stim = { startA, startB, duration };
    },

    /** Moves the stimulus times (e.g. "fire now"), keeping the currents. */
    fireAt(startA: number, startB: number) {
      stim = { ...stim, startA, startB };
    },

    /** Every fibre at rest, the clock at 0, records and traces cleared. */
    reset() {
      const g = restingGates();
      const d = new Float32Array(total * 8);
      for (let i = 0; i < total; i++) d.set([REST_V, g.m, g.h, g.n, 0, 0, -1, 0], i * 8);
      dyn.write(d);
      const o = new Float32Array(outSize);
      o.fill(REST_V, histOffset, sigOffset);
      out.write(o);
      time = 0;
      stepIndex = 0;
      capHead = 0;
      capFilled = 0;
    },

    step(steps: number, o: StepOptions) {
      const phi = temperatureFactor(o.celsius);
      while (steps > 0) {
        const k = Math.min(steps, maxSteps);
        const record = !!o.electrode;
        cable
          .set({
            params: {
              fibres: F,
              steps: k,
              histEvery: o.histEvery ?? 0,
              histRows,
              stepIndex,
              sigSteps: record ? k : 0,
              histOffset,
              sigOffset,
              time,
              dt: o.dt,
              phi,
              stimDuration: stim.duration,
              startA: stim.startA,
              startB: stim.startB,
              electrode: o.electrode?.x ?? 0,
              height: o.electrode?.height ?? 1,
              probeA: o.probes?.[0] ?? 0,
              probeB: o.probes?.[1] ?? 0,
              probeOffset,
              probeLen: o.probes ? probeLen : 0,
            },
            fibres: fibreBuffer,
            comps: compBuffer,
            dyn,
            out,
          })
          .dispatch(Math.max(1, Math.ceil(F / WORKGROUP)));
        if (record) {
          cap.set({ params: { fibres: F, steps: k, sigOffset, capOffset, head: capHead, len: capLen, pad0: 0, pad1: 0 }, out }).dispatch(Math.ceil(k / WORKGROUP));
          capHead = (capHead + k) % capLen;
          capFilled = Math.min(capLen, capFilled + k);
        }
        time += k * o.dt;
        stepIndex += k;
        steps -= k;
      }
    },

    /** (V, m, h, n, c′, d′, first arrival or −1, crossings) per compartment, all fibres concatenated. */
    async readDyn(): Promise<{ time: number; data: Float32Array }> {
      const at = time;
      return { time: at, data: new Float32Array(await dyn.read()) };
    },

    /** Copies a slice of the output buffer (offsets in floats) without reading the rest back. */
    async readOut(offset: number, count: number): Promise<Float32Array> {
      const device = gpu.gpu;
      const size = count * 4;
      const staging = device.createBuffer({ size, usage: USAGE_MAP_READ | USAGE_COPY_DST, label: "axon read-back" });
      const encoder = device.createCommandEncoder({ label: "axon read-back" });
      encoder.copyBufferToBuffer((out as unknown as { gpu: unknown }).gpu, offset * 4, staging, 0, size);
      device.queue.submit([encoder.finish()]);
      try {
        await staging.mapAsync(MAP_MODE_READ);
        return new Float32Array(staging.getMappedRange().slice(0));
      } finally {
        staging.destroy();
      }
    },

    dispose() {
      for (const b of [fibreBuffer, compBuffer, dyn, out]) (b as unknown as Destroyable).destroy?.();
    },
  };
  engine.reset();
  return engine;
}

export type Engine = ReturnType<typeof createEngine>;
