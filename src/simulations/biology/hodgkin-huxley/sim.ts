import { compute, effect, frameLoop, storage } from "vgpu";
import { frameDelta } from "@/lib/gpu/runtime";
import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import { DEFAULT_PARAMS, DT_MS, HISTORY_SAMPLES, REST_STATE, SAMPLE_EVERY } from "./hh";
import type { HHParams } from "./hh";
import neuronShader from "./neuron.wgsl";
import renderShader from "./render.wgsl";

export interface HHSimParams extends HHParams {
  /** Milliseconds of neuron time simulated per real-time second. */
  timeScale: number;
}

export const DEFAULT_SIM_PARAMS: HHSimParams = { ...DEFAULT_PARAMS, timeScale: 30 };

/** Upper bound on integration steps per frame (the integrator runs on a single GPU thread). */
const MAX_STEPS_PER_FRAME = 1500;

export interface HHHandle extends SimHandle {
  setParams(params: Partial<HHSimParams>): void;
  play(): void;
  pause(): void;
  reset(): void;
}

const initialState = () => new Float32Array([REST_STATE.V, REST_STATE.m, REST_STATE.h, REST_STATE.n]);

export function createHodgkinHuxley({ gpu, surface, clock }: SimContext): HHHandle {
  const stateBuffer = storage(gpu, 16);
  const historyBuffer = storage(gpu, HISTORY_SAMPLES * 16);
  const metaBuffer = storage(gpu, 16);
  const neuronPass = compute(gpu, neuronShader);
  const renderPass = effect(gpu, renderShader);

  let params: HHSimParams = { ...DEFAULT_SIM_PARAMS };
  let playing = true;
  let simTime = 0; // ms of neuron time
  let stepDebt = 0; // fractional integration steps carried between frames

  const resetBuffers = () => {
    stateBuffer.write(initialState());
    // Pre-fill the history with the resting state so the trace starts as a flat line.
    const rest = new Float32Array(HISTORY_SAMPLES * 4);
    for (let i = 0; i < HISTORY_SAMPLES; i++) rest.set(initialState(), i * 4);
    historyBuffer.write(rest);
    metaBuffer.write(new Uint32Array([0, 0, 0, 0]));
    simTime = 0;
    stepDebt = 0;
  };
  resetBuffers();

  const loop = frameLoop(gpu, (frame) => {
    if (playing) {
      stepDebt += (params.timeScale * frameDelta(clock)) / DT_MS;
      const whole = Math.floor(stepDebt);
      stepDebt -= whole;
      const steps = Math.min(whole, MAX_STEPS_PER_FRAME);

      if (steps > 0) {
        neuronPass
          .set({
            params: {
              g_Na: params.g_Na,
              g_K: params.g_K,
              g_L: params.g_L,
              E_Na: params.E_Na,
              E_K: params.E_K,
              E_L: params.E_L,
              C_m: params.C_m,
              I_inj: params.I_inj,
              temperature: params.temperature,
              dt: DT_MS,
              substeps: steps,
              pulse_mode: params.pulse_mode,
              time: simTime,
              sample_every: SAMPLE_EVERY,
              pad0: 0,
              pad1: 0,
            },
            state: stateBuffer,
            history: historyBuffer,
            histMeta: metaBuffer,
          })
          .dispatch(1);
        simTime += steps * DT_MS;
      }
    }

    renderPass.set({
      history: historyBuffer,
      histMeta: metaBuffer,
      params: { resolution: surface.size, time: simTime, pad: 0 },
    });
    frame.pass(surface, renderPass);
  });

  return {
    setParams(next) {
      params = { ...params, ...next };
    },
    play: () => {
      playing = true;
    },
    pause: () => {
      playing = false;
    },
    reset: resetBuffers,
    dispose: () => loop.stop(),
  };
}
