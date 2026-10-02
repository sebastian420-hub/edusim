import { effect, frameLoop } from "vgpu";
import { frameDelta } from "@/lib/gpu/runtime";
import type { SimContext, SimHandle } from "@/lib/gpu/runtime";
import waveShader from "./wave.wgsl";

export type WaveMode = "point" | "two-points" | "single-slit" | "double-slit";
export type WaveView = "amplitude" | "intensity" | "water";

export interface WaveParams {
  frequency: number;
  amplitude: number;
  damping: number;
  waveSpeed: number;
  mode: WaveMode;
  /** Distance between the two point sources / slit centres. */
  separation: number;
  slitWidth: number;
  view: WaveView;
}

export const DEFAULTS: WaveParams = {
  frequency: 3.0,
  amplitude: 1.0,
  damping: 0.02,
  waveSpeed: 1.0,
  mode: "double-slit",
  separation: 0.8,
  slitWidth: 0.15,
  view: "amplitude",
};

// Must match the numeric codes in wave.wgsl.
const MODE_CODES: Record<WaveMode, number> = { point: 0, "two-points": 1, "single-slit": 2, "double-slit": 3 };
const VIEW_CODES: Record<WaveView, number> = { amplitude: 0, intensity: 1, water: 2 };

export interface WaveHandle extends SimHandle {
  setParams(params: Partial<WaveParams>): void;
  play(): void;
  pause(): void;
  reset(): void;
}

export function createWaveSim({ gpu, surface, clock }: SimContext): WaveHandle {
  const pass = effect(gpu, waveShader);

  let params: WaveParams = { ...DEFAULTS };
  let playing = true;
  let time = 0;

  const paramUniforms = () => ({
    frequency: params.frequency,
    amplitude: params.amplitude,
    damping: params.damping,
    waveSpeed: params.waveSpeed,
    separation: params.separation,
    slitWidth: params.slitWidth,
    mode: MODE_CODES[params.mode],
    viewMode: VIEW_CODES[params.view],
  });

  // vgpu needs every member of a struct on its first set(); later calls may send any subset.
  pass.set({ u: { resolution: surface.size, time, ...paramUniforms() } });
  const stopResizeListener = surface.onResize(() => pass.set({ u: { resolution: surface.size } }));

  const loop = frameLoop(gpu, (frame) => {
    if (playing) {
      time += frameDelta(clock);
      pass.set({ u: { time } });
    }
    frame.pass(surface, pass);
  });

  return {
    setParams(next) {
      params = { ...params, ...next };
      pass.set({ u: paramUniforms() });
    },
    play: () => {
      playing = true;
    },
    pause: () => {
      playing = false;
    },
    reset: () => {
      time = 0;
      pass.set({ u: { time } });
    },
    dispose() {
      loop.stop();
      stopResizeListener();
    },
  };
}
