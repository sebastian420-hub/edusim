import { init, surface, effect, frameLoop } from "vgpu";
import { shaderSource } from "./shader.wgsl";

export interface WaveUniforms {
  frequency: number;
  amplitude: number;
  damping: number;
  waveSpeed: number;
  sourceCount: number;
  source1: [number, number];
  source2: [number, number];
  slitWidth: number;
  slitSeparation: number;
  viewMode: number;
}

export const DEFAULTS: WaveUniforms = {
  frequency: 3.0,
  amplitude: 1.0,
  damping: 0.02,
  waveSpeed: 1.0,
  sourceCount: 2,
  source1: [-1.5, 0.0],
  source2: [-1.5, 0.0],
  slitWidth: 0.15,
  slitSeparation: 0.6,
  viewMode: 0,
};

export async function createWaveRenderer(canvas: HTMLCanvasElement) {
  const gpu = await init();
  gpu.onError((err) => {
    console.error(">>> WAVE GPU ERROR:", err.message, (err as any).cause, (err as any).detail, err);
  });

  const target = surface(gpu, canvas, { dpr: window.devicePixelRatio || 1 });
  const pass = effect(gpu, shaderSource);
  
  let time = 0;
  let params = { ...DEFAULTS };
  
  const loop = frameLoop(gpu, (f) => {
    time += 0.016; 
    
    pass.set({
      u: {
        resolution: target.size,
        time: time,
        frequency: params.frequency,
        amplitude: params.amplitude,
        damping: params.damping,
        waveSpeed: params.waveSpeed,
        sourceCount: params.sourceCount,
        source1: params.source1,
        source2: params.source2,
        slitWidth: params.slitWidth,
        slitSeparation: params.slitSeparation,
        viewMode: params.viewMode,
        pad: 0
      }
    });
    
    f.pass(target, pass);
  });
  
  return {
    ready: true,
    setUniforms: (newParams: Partial<WaveUniforms>) => {
      params = { ...params, ...newParams };
    },
    dispose: () => {
      loop.stop();
      gpu.dispose();
    }
  };
}
