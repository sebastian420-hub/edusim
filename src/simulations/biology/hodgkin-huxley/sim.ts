import { init, target, surface, effect, compute, storage, frameLoop } from 'vgpu';

import { shaderSource as neuronWgsl } from './neuron.wgsl';
import { shaderSource as renderWgsl } from './render.wgsl';

export interface SimState {
  g_Na: number;
  g_K: number;
  g_L: number;
  E_Na: number;
  E_K: number;
  E_L: number;
  C_m: number;
  I_inj: number;
  temperature: number;
  pulse_mode: number;
}

export const createSim = async (canvas: HTMLCanvasElement) => {
  const gpu = await init();
  gpu.onError((err) => {
    console.error(">>> BIOLOGY GPU ERROR:", err.message, (err as any).cause, (err as any).detail, err);
  });

  const targetSurface = surface(gpu, canvas, { dpr: window.devicePixelRatio || 1 });
  
  const stateBuffer = storage(gpu, 16);
  const initState = new Float32Array([-65.0, 0.05, 0.6, 0.31]);
  stateBuffer.write(initState);
  
  const historyBuffer = storage(gpu, 32768);
  const metaBuffer = storage(gpu, 16);
  metaBuffer.write(new Uint32Array([0, 0, 0, 0]));

  const neuronPass = compute(gpu, neuronWgsl);
  const renderPass = effect(gpu, renderWgsl);
  
  let time = 0.0;
  let isPlaying = true;
  let state: SimState = {
    g_Na: 120.0,
    g_K: 36.0,
    g_L: 0.3,
    E_Na: 50.0,
    E_K: -77.0,
    E_L: -54.387,
    C_m: 1.0,
    I_inj: 10.0,
    temperature: 6.3,
    pulse_mode: 0,
  };

  const loop = frameLoop(gpu, (f) => {
    if (isPlaying) {
      const dt = 0.01;
      const substeps = 10;
      
      neuronPass.set({
        params: {
          g_Na: state.g_Na,
          g_K: state.g_K,
          g_L: state.g_L,
          E_Na: state.E_Na,
          E_K: state.E_K,
          E_L: state.E_L,
          C_m: state.C_m,
          I_inj: state.I_inj,
          temperature: state.temperature,
          dt: dt,
          substeps: substeps,
          pulse_mode: state.pulse_mode,
          time: time,
          pad0: 0, pad1: 0, pad2: 0
        },
        state: stateBuffer,
        history: historyBuffer,
        meta: metaBuffer
      }).dispatch(1);
        
      time += dt * substeps;
    }

    renderPass.set({
      history: historyBuffer,
      meta: metaBuffer,
      params: {
        resolution: targetSurface.size,
        time: time,
        pad: 0
      }
    });
    
    f.pass(targetSurface, renderPass);
  });

  return {
    destroy: () => {
      loop.stop();
      gpu.dispose();
    },
    setState: (newState: Partial<SimState>) => {
      state = { ...state, ...newState };
    },
    play: () => { isPlaying = true; },
    pause: () => { isPlaying = false; },
    reset: () => {
      time = 0;
      stateBuffer.write(initState);
      metaBuffer.write(new Uint32Array([0, 0, 0, 0]));
    }
  };
};
