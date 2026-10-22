import { draw, effect, geometry } from "vgpu";
import type { FramePass, Gpu, StorageBuffer } from "vgpu";
import { ACTIVATION_CODE } from "./engine";
import type { Engine } from "./engine";
import linksShader from "./links.wgsl";
import { FEATURES } from "./net";
import playgroundShader from "./playground.wgsl";
import pointsShader from "./points.wgsl";
import type { SceneLayout } from "./scene";

const rawBuffer = (b: StorageBuffer): unknown => (b as unknown as { gpu: unknown }).gpu;
// GPUBufferUsage flags (the WebGPU globals are not in this project's type environment).
const USAGE_COPY_DST = 0x08;
const USAGE_VERTEX = 0x20;

const sizesUniform = (columns: number[]) => [0, 1].map((k) => [0, 1, 2, 3].map((i) => columns[4 * k + i] ?? 0));

/**
 * Draws the Playground: the decision map and the network diagram (playground.wgsl), the weights as lines
 * (links.wgsl) and the data points (points.wgsl). Weights are copied GPU-to-GPU into a vertex buffer each frame
 * (vertex shaders may not read storage buffers on every device).
 */
export function createPlaygroundRenderer(gpu: Gpu, engine: Engine, featureMask: number) {
  const device = gpu.gpu;
  const lay = engine.layout;
  const weightCount = lay.layers.reduce((a, L) => a + L.nIn * L.nOut, 0);
  const weightVertices = device.createBuffer({ size: Math.max(4, weightCount * 4), usage: USAGE_VERTEX | USAGE_COPY_DST, label: "network weights (vertex)" });
  let pointVertices = device.createBuffer({ size: 16, usage: USAGE_VERTEX | USAGE_COPY_DST, label: "data points (vertex)" });
  let pointCount = 0;

  const background = effect(gpu, playgroundShader);
  const linkGeometry = geometry(gpu, {
    buffers: [{ attributes: { weight: "float32" }, buffer: weightVertices, stride: 4, stepMode: "instance" }],
    vertexCount: 6,
    instanceCount: Math.max(1, weightCount),
  });
  const links = draw(gpu, { shader: linksShader, geometry: linkGeometry, blend: "premultiplied" });
  let pointGeometry = geometry(gpu, { buffers: [{ attributes: { point: "float32x4" }, buffer: pointVertices, stride: 16, stepMode: "instance" }], vertexCount: 6, instanceCount: 1 });
  let pointDraw = draw(gpu, { shader: pointsShader, geometry: pointGeometry, blend: "premultiplied" });

  const featureIds: number[] = [];
  FEATURES.forEach((_, f) => {
    if ((featureMask >> f) & 1) featureIds.push(f);
  });

  return {
    /** The data points: x, y (plane), label, 1 for test points. */
    setPoints(data: Float32Array) {
      pointCount = data.length / 4;
      pointVertices.destroy();
      pointVertices = device.createBuffer({ size: Math.max(16, data.byteLength), usage: USAGE_VERTEX | USAGE_COPY_DST, label: "data points (vertex)" });
      device.queue.writeBuffer(pointVertices, 0, data);
      pointGeometry = geometry(gpu, { buffers: [{ attributes: { point: "float32x4" }, buffer: pointVertices, stride: 16, stepMode: "instance" }], vertexCount: 6, instanceCount: Math.max(1, pointCount) });
      pointDraw = draw(gpu, { shader: pointsShader, geometry: pointGeometry, blend: "premultiplied" });
    },

    /** Encodes the weight copy, then returns the body of a frame pass. */
    prepare(target: { readonly size: readonly [number, number] }, scene: SceneLayout, opts: { showTest: boolean; hasMap: boolean; dpr: number }) {
      const resolution = [target.size[0], target.size[1]] as [number, number];
      const encoder = device.createCommandEncoder({ label: "network weight copies" });
      let at = 0;
      for (const L of lay.layers) {
        const bytes = L.nIn * L.nOut * 4;
        if (bytes > 0) encoder.copyBufferToBuffer(rawBuffer(engine.buffers.params), L.w * 4, weightVertices, at, bytes);
        at += bytes;
      }
      device.queue.submit([encoder.finish()]);

      const rect = (r: { x: number; y: number; w: number; h: number }) => [r.x, r.y, r.w, r.h];
      const sizes = sizesUniform(scene.columns);
      background.set({
        map: engine.buffers.map,
        scene: {
          resolution,
          spacing: scene.spacing,
          node: scene.node,
          plot: rect(scene.plot),
          diagram: rect(scene.diagram),
          columns: scene.columns.length,
          mapSize: engine.mapSize,
          thumb: engine.thumb,
          act: ACTIVATION_CODE[engine.spec.activation],
          sizes,
          features: [0, 1].map((k) => [0, 1, 2, 3].map((i) => featureIds[4 * k + i] ?? 0)),
          hasMap: opts.hasMap ? 1 : 0,
          pad0: 0,
          pad1: 0,
          pad2: 0,
        },
      });
      links.set({ links: { resolution, spacing: scene.spacing, node: scene.node, diagram: rect(scene.diagram), columns: scene.columns.length, pad0: 0, pad1: 0, pad2: 0, sizes } });
      const radius = Math.max(2.5, Math.min(5, scene.plot.w / 120)) * Math.max(1, opts.dpr * 0.8);
      pointDraw.set({ points: { resolution, radius, showTest: opts.showTest ? 1 : 0, plot: rect(scene.plot) } });

      return (pass: FramePass) => {
        pass.draw(background);
        if (weightCount > 0) pass.draw(links, { instances: weightCount, vertices: 6 });
        if (pointCount > 0) pass.draw(pointDraw, { instances: pointCount, vertices: 6 });
      };
    },

    dispose() {
      weightVertices.destroy();
      pointVertices.destroy();
    },
  };
}

export type PlaygroundRenderer = ReturnType<typeof createPlaygroundRenderer>;
