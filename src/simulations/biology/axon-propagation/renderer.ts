import { effect } from "vgpu";
import type { FramePass, Gpu } from "vgpu";
import axonShader from "./axon.wgsl";
import nerveShader from "./nerve.wgsl";
import type { Engine } from "./engine";

/** Bands of the Axon view (fractions of the canvas height) and the horizontal margin. */
export const AXON_LAYOUT = { tubeTop: 0.1, tubeBottom: 0.24, kymoTop: 0.32, kymoBottom: 0.95, margin: 0.05 };
export const NERVE_LAYOUT = { top: 0.06, bottom: 0.62, margin: 0.05 };

export interface AxonFrame {
  length: number;
  sweepMs: number;
  histEvery: number;
  rowsDone: number;
  dt: number;
  myelinated: boolean;
}

/** Draws the Axon view (tube + kymograph of fibre 0) or the Nerve view (every fibre, a row each). */
export function createRenderer(gpu: Gpu, engine: Engine) {
  const axon = effect(gpu, axonShader);
  const nerve = effect(gpu, nerveShader);
  const fibreTable = engine.buffers.fibres;
  return {
    axon(target: { readonly size: readonly [number, number] }, f: AxonFrame) {
      const L = engine.layout;
      axon.set({
        view: {
          resolution: [target.size[0], target.size[1]],
          length: f.length,
          sweepMs: f.sweepMs,
          n: L.n0,
          histEvery: f.histEvery,
          histRows: L.histRows,
          rowsDone: f.rowsDone,
          dt: f.dt,
          myelinated: f.myelinated ? 1 : 0,
          ...AXON_LAYOUT,
          pad: 0,
        },
        comps: engine.buffers.comps,
        dyn: engine.buffers.dyn,
        out: engine.buffers.out,
      });
      return (pass: FramePass) => pass.draw(axon);
    },
    nerve(target: { readonly size: readonly [number, number] }, length: number) {
      nerve.set({
        view: { resolution: [target.size[0], target.size[1]], length, fibres: engine.fibres.length, ...NERVE_LAYOUT, pad: 0 },
        fibres: fibreTable,
        comps: engine.buffers.comps,
        dyn: engine.buffers.dyn,
      });
      return (pass: FramePass) => pass.draw(nerve);
    },
  };
}

export type Renderer = ReturnType<typeof createRenderer>;
