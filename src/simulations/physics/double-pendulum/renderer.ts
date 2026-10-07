import { draw, effect, geometry } from "vgpu";
import type { FramePass, Gpu, StorageBuffer } from "vgpu";
import armsShader from "./arms.wgsl";
import fractalShader from "./fractal.wgsl";
import trailShader from "./trail-draw.wgsl";
import type { Engine } from "./engine";
import { flipBoundary } from "./pendulum";
import type { FractalWindow } from "./pendulum";
import { TRAIL_LEN } from "./sim-constants";

/** Where the camera looks (world units = metres, y up, pivot at the origin). */
export interface View {
  center: [number, number];
  pxPerUnit: number;
}

const rawBuffer = (b: StorageBuffer): unknown => (b as unknown as { gpu: unknown }).gpu;
// GPUBufferUsage flags (the WebGPU globals are not in this project's type environment).
const USAGE_COPY_DST = 0x08;
const USAGE_VERTEX = 0x20;

export type DrawMode = { kind: "pendulums"; crowd: boolean; color?: [number, number, number, number] } | { kind: "fractal"; size: number; window: FractalWindow; tMax: number; boundary: boolean };

/**
 * Draws pendulums (rods and bobs, one instance each; a trail for pendulum 0) or the fractal map.
 * Pendulum states are copied GPU-to-GPU into a vertex buffer each frame (vertex shaders may not read storage
 * buffers on every device); the fractal is a fullscreen fragment shader, which may.
 */
export function createRenderer(gpu: Gpu, engine: Engine) {
  const device = gpu.gpu;
  const usage = USAGE_VERTEX | USAGE_COPY_DST;
  const stateVertices = device.createBuffer({ size: engine.capacity * 16, usage, label: "pendulum states (vertex)" });
  const trailVertices = device.createBuffer({ size: TRAIL_LEN * 16, usage, label: "pendulum trail (vertex)" });

  const armGeometry = geometry(gpu, {
    buffers: [{ attributes: { state: "float32x4" }, buffer: stateVertices, stride: 16, stepMode: "instance" }],
    vertexCount: 30,
    instanceCount: engine.capacity,
  });
  const trailGeometry = geometry(gpu, {
    buffers: [{ attributes: { point: "float32x4" }, buffer: trailVertices, stride: 16, stepMode: "vertex" }],
    topology: "line-strip",
    vertexCount: TRAIL_LEN,
  });
  const solidArms = draw(gpu, { shader: armsShader, geometry: armGeometry, blend: "premultiplied" });
  const crowdArms = draw(gpu, { shader: armsShader, geometry: armGeometry, blend: "additive" });
  const trail = draw(gpu, { shader: trailShader, geometry: trailGeometry, blend: "alpha" });
  const fractal = effect(gpu, fractalShader);

  return {
    /** Encodes the copies, then returns the body of a frame pass. */
    prepare(target: { readonly size: readonly [number, number] }, view: View, mode: DrawMode, showTrail: boolean) {
      const resolution = [target.size[0], target.size[1]] as [number, number];
      if (mode.kind === "fractal") {
        const { a, b, c } = flipBoundary(engine.params);
        fractal.set({
          flip: engine.buffers.flip,
          params: { resolution, n: mode.size, showBoundary: mode.boundary ? 1 : 0, cx: mode.window.cx, cy: mode.window.cy, span: mode.window.span, tMax: mode.tMax, a, b, c, pad: 0 },
        });
        return (pass: FramePass) => pass.draw(fractal);
      }
      const n = engine.n;
      const encoder = device.createCommandEncoder({ label: "pendulum vertex copies" });
      if (n > 0) encoder.copyBufferToBuffer(rawBuffer(engine.buffers.state), 0, stateVertices, 0, n * 16);
      if (showTrail) encoder.copyBufferToBuffer(rawBuffer(engine.buffers.trail), 0, trailVertices, 0, TRAIL_LEN * 16);
      device.queue.submit([encoder.finish()]);

      const { l1, l2 } = engine.params;
      const scale = view.pxPerUnit;
      const crowd = mode.crowd;
      const arms = crowd ? crowdArms : solidArms;
      arms.set({
        view: {
          center: view.center,
          resolution,
          pxPerUnit: scale,
          l1,
          l2,
          rodPx: crowd ? 1.6 : Math.max(2, scale * 0.025),
          bobPx: crowd ? 3.5 : Math.max(5, scale * 0.06),
          alpha: crowd ? Math.min(0.6, Math.max(0.03, 4 / Math.sqrt(n))) : 1,
          colorMode: crowd ? 1 : 0,
          count: n,
          color: mode.color ?? [0.85, 0.9, 1, 1],
        },
      });
      const head = engine.trailHead;
      const filled = engine.trailFilled;
      trail.set({ view: { center: view.center, resolution, pxPerUnit: scale, pad0: 0, len: TRAIL_LEN, head, filled, pad1: 0, color: [0.35, 0.75, 1, 0.9] } });

      return (pass: FramePass) => {
        if (showTrail && filled > 1) {
          if (filled === TRAIL_LEN && head < TRAIL_LEN - 1) pass.draw(trail, { firstVertex: head, vertices: TRAIL_LEN - head });
          if (head > 1) pass.draw(trail, { firstVertex: 0, vertices: head });
        }
        if (n > 0) pass.draw(arms, { instances: n, vertices: crowd ? 24 : 30 }); // a crowd shares one pivot: not drawn
      };
    },

    dispose() {
      stateVertices.destroy();
      trailVertices.destroy();
    },
  };
}

export type Renderer = ReturnType<typeof createRenderer>;
