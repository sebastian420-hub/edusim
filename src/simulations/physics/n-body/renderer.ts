import { draw, geometry } from "vgpu";
import type { FramePass, Gpu, StorageBuffer } from "vgpu";
import bodiesShader from "./bodies.wgsl";
import trailsShader from "./trails.wgsl";
import type { Engine } from "./engine";
import { TRAIL_LEN } from "./sim-constants";

/** Where the camera looks: world point at the centre of the target, and its zoom. */
export interface View {
  center: [number, number];
  pxPerUnit: number;
}

/** The raw GPUBuffer behind a vgpu storage buffer (the public type hides it). */
const rawBuffer = (b: StorageBuffer): unknown => (b as unknown as { gpu: unknown }).gpu;

// GPUBufferUsage flags (spelled out: the WebGPU globals are not in this project's type environment).
const USAGE_COPY_DST = 0x08;
const USAGE_VERTEX = 0x20;

/**
 * Draws an engine's bodies (glowing instanced discs) and orbit trails (fading line strips).
 *
 * Vertex shaders may not read storage buffers on every device (compatibility-mode adapters allow none), so the
 * positions and trails are copied GPU-to-GPU into vertex buffers each frame and read as vertex attributes.
 */
export function createRenderer(gpu: Gpu, engine: Engine) {
  const device = gpu.gpu;
  const usage = USAGE_VERTEX | USAGE_COPY_DST;
  const bodyVertices = device.createBuffer({ size: engine.capacity * 16, usage, label: "n-body positions (vertex)" });
  const trailVertices = device.createBuffer({ size: Math.max(1, engine.trailBodies) * TRAIL_LEN * 16, usage, label: "n-body trails (vertex)" });
  const lookBuffer = device.createBuffer({ size: engine.capacity * 16, usage, label: "n-body looks" });

  const bodyGeometry = geometry(gpu, {
    buffers: [
      { attributes: { body: "float32x4" }, buffer: bodyVertices, stride: 16, stepMode: "instance" },
      { attributes: { look: "float32x4" }, buffer: lookBuffer, stride: 16, stepMode: "instance" },
    ],
    vertexCount: 6,
    instanceCount: engine.capacity,
  });
  const trailGeometry = geometry(gpu, {
    buffers: [
      { attributes: { point: "float32x4" }, buffer: trailVertices, stride: 16, stepMode: "vertex" },
      { attributes: { look: "float32x4" }, buffer: lookBuffer, stride: 16, stepMode: "instance" },
    ],
    topology: "line-strip",
    vertexCount: Math.max(1, engine.trailBodies) * TRAIL_LEN,
    instanceCount: engine.capacity,
  });
  const bodies = draw(gpu, { shader: bodiesShader, geometry: bodyGeometry, blend: "additive" });
  const trails = draw(gpu, { shader: trailsShader, geometry: trailGeometry, blend: "alpha" });

  let glow = 0.5;
  let showTrails = true;

  return {
    /** Per body (r, g, b, radius in pixels). */
    setLooks(looks: Float32Array) {
      device.queue.writeBuffer(lookBuffer, 0, looks.buffer, looks.byteOffset, Math.min(looks.byteLength, engine.capacity * 16));
    },
    setGlow(value: number) {
      glow = value;
    },
    setTrails(on: boolean) {
      showTrails = on;
    },

    /** Encodes the copies, then returns the body of a frame pass that draws trails and bodies into `target`. */
    prepare(target: { readonly size: readonly [number, number] }, view: View) {
      const n = engine.n;
      const trailCount = Math.min(n, engine.trailBodies);
      const encoder = device.createCommandEncoder({ label: "n-body vertex copies" });
      if (n > 0) encoder.copyBufferToBuffer(rawBuffer(engine.buffers.pos), 0, bodyVertices, 0, n * 16);
      if (showTrails && trailCount > 0) encoder.copyBufferToBuffer(rawBuffer(engine.buffers.trail), 0, trailVertices, 0, trailCount * TRAIL_LEN * 16);
      device.queue.submit([encoder.finish()]);

      const resolution = [target.size[0], target.size[1]] as [number, number];
      bodies.set({ view: { center: view.center, resolution, pxPerUnit: view.pxPerUnit, glow, pad0: 0, pad1: 0 } });
      const head = engine.trailHead;
      const filled = engine.trailFilled;
      trails.set({ view: { center: view.center, resolution, pxPerUnit: view.pxPerUnit, pad0: 0, len: TRAIL_LEN, head, filled, pad1: 0, pad2: 0, pad3: 0 } });

      return (pass: FramePass) => {
        if (showTrails && filled > 1) {
          for (let b = 0; b < trailCount; b++) {
            // Oldest to newest: [head, len) then [0, head).
            const base = b * TRAIL_LEN;
            if (filled === TRAIL_LEN && head < TRAIL_LEN - 1) pass.draw(trails, { firstVertex: base + head, vertices: TRAIL_LEN - head, firstInstance: b, instances: 1 });
            if (head > 1) pass.draw(trails, { firstVertex: base, vertices: head, firstInstance: b, instances: 1 });
          }
        }
        if (n > 0) pass.draw(bodies, { instances: n, vertices: 6 });
      };
    },

    dispose() {
      bodyVertices.destroy();
      trailVertices.destroy();
      lookBuffer.destroy();
    },
  };
}

export type Renderer = ReturnType<typeof createRenderer>;
