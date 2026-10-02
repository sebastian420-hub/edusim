import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { compute, effect, pingPongStorage, storage, target } from "vgpu/node";
import { tryInitGpu } from "@/test/gpu";
import type { NodeGpu } from "@/test/gpu";
import computeShader from "./cs/cellular-automata/compute.wgsl";
import { workgroupsFor } from "./cs/cellular-automata/workgroup";
import neuronShader from "./biology/hodgkin-huxley/neuron.wgsl";
import waveShader from "./physics/wave-interference/wave.wgsl";
import { DEFAULTS as WAVE_DEFAULTS, phasor as wavePhasor } from "./physics/wave-interference/wave";
import type { WaveMode } from "./physics/wave-interference/wave";
import { DEFAULT_PARAMS, DT_MS, HISTORY_SAMPLES, REST_STATE, SAMPLE_EVERY, simulate } from "./biology/hodgkin-huxley/hh";

let gpu: NodeGpu | null = null;

beforeAll(async () => {
  gpu = await tryInitGpu();
});
afterAll(() => gpu?.dispose());

const needsGpu = (name: string, fn: (gpu: NodeGpu) => Promise<void>) =>
  it(name, async (ctx) => {
    if (!gpu) return ctx.skip();
    await fn(gpu);
  });

describe("vgpu conventions the shaders rely on", () => {
  needsGpu("effect uv has its origin at the top-left with y pointing down", async (gpu) => {
    const t = target(gpu, { size: [4, 4] });
    effect(gpu, `@fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f { return vec4f(uv.x, uv.y, 0.0, 1.0); }`).draw(t);
    const px = await t.color.read({ mipLevel: 0, region: "all" });
    const at = (x: number, y: number) => px.slice((y * 4 + x) * 4, (y * 4 + x) * 4 + 2);
    expect(at(0, 0)[1]).toBeLessThan(64); // y small at top
    expect(at(0, 3)[1]).toBeGreaterThan(190); // y large at bottom
    expect(at(3, 0)[0]).toBeGreaterThan(190); // x large at right
  });
});

describe("Hodgkin-Huxley compute shader", () => {
  needsGpu("matches the CPU reference model", async (gpu) => {
    const stateBuf = storage(gpu, 16);
    const historyBuf = storage(gpu, HISTORY_SAMPLES * 16);
    const metaBuf = storage(gpu, 16);
    stateBuf.write(new Float32Array([REST_STATE.V, REST_STATE.m, REST_STATE.h, REST_STATE.n]));
    metaBuf.write(new Uint32Array([0, 0, 0, 0]));

    const durationMs = 30;
    const steps = Math.round(durationMs / DT_MS);
    const p = { ...DEFAULT_PARAMS, I_inj: 10, pulse_mode: 0 as const };
    compute(gpu, neuronShader)
      .set({
        params: { ...p, dt: DT_MS, substeps: steps, time: 0, sample_every: SAMPLE_EVERY, pad0: 0, pad1: 0 },
        state: stateBuf,
        history: historyBuf,
        histMeta: metaBuf,
      })
      .dispatch(1);

    const history = new Float32Array(await historyBuf.read());
    const cpu = simulate(p, durationMs);
    const gpuV = Array.from({ length: cpu.V.length }, (_, i) => history[i * 4]);

    const maxErr = Math.max(...cpu.V.map((v, i) => Math.abs(v - gpuV[i])));
    expect(Math.max(...gpuV)).toBeGreaterThan(30); // it spiked
    expect(maxErr).toBeLessThan(2); // mV, float32 vs float64

    const meta = new Uint32Array(await metaBuf.read());
    expect(meta[0]).toBe(cpu.V.length); // ring-buffer head advanced by the number of samples
    expect(meta[1]).toBe(steps);
  });
});

describe("Cellular automata compute shader", () => {
  needsGpu("advances a glider one cell diagonally every 4 generations (torus wrap)", async (gpu) => {
    const size = 16;
    const cells = pingPongStorage(gpu, size * size * 4);
    const initial = new Uint32Array(size * size);
    const glider: [number, number][] = [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]];
    for (const [x, y] of glider) initial[(y + 6) * size + (x + 6)] = 1;
    cells.read.write(initial);

    const pass = compute(gpu, computeShader);
    for (let g = 0; g < 4; g++) {
      pass
        .set({
          cellsIn: cells.read,
          cellsOut: cells.write,
          params: { width: size, height: size, rule_birth: 0b1000, rule_survive: 0b1100 },
        })
        .dispatch(workgroupsFor(size), workgroupsFor(size), 1);
      cells.swap();
    }

    const out = new Uint32Array(await cells.read.read());
    const expected = new Uint32Array(size * size);
    for (const [x, y] of glider) expected[(y + 7) * size + (x + 7)] = 1;
    expect(Array.from(out)).toEqual(Array.from(expected));
  });
});

describe("Wave interference shader", () => {
  const SIZE = 128;
  const render = async (gpu: NodeGpu, patch: Record<string, number>) => {
    const t = target(gpu, { size: [SIZE, SIZE] });
    const fx = effect(gpu, waveShader);
    fx.set({
      u: {
        resolution: [SIZE, SIZE],
        time: 0.3,
        frequency: 3,
        amplitude: 1,
        damping: 0.02,
        waveSpeed: 1,
        separation: 0.8,
        slitWidth: 0.15,
        mode: 3,
        viewMode: 1,
        ...patch,
      },
    });
    fx.draw(t);
    return t.color.read({ mipLevel: 0, region: "all" });
  };
  const channel = (px: Uint8Array, x: number, y: number, c = 0) => px[(y * SIZE + x) * 4 + c];
  const red = (px: Uint8Array, x: number, y: number) => channel(px, x, y, 0);

  needsGpu("double slit: symmetric intensity fringes right of the barrier", async (gpu) => {
    const px = await render(gpu, {});
    // Column well to the right of the barrier (x = -1.5), where the fringes are fully formed.
    const x = Math.floor(SIZE * 0.85);
    const column = Array.from({ length: SIZE }, (_, y) => channel(px, x, y, 1)); // green has the most contrast
    expect(Math.max(...column) - Math.min(...column)).toBeGreaterThan(40); // visible fringes
    for (let y = 0; y < SIZE / 2; y++) {
      expect(Math.abs(column[y] - column[SIZE - 1 - y])).toBeLessThanOrEqual(2); // mirror symmetry
    }
  });

  needsGpu("barrier pixels are drawn grey and the slits are open", async (gpu) => {
    const px = await render(gpu, {});
    // Barrier at x = -1.5 -> uv.x = 0.5 + (-1.5 / (4 * aspect)) with aspect 1.
    const bx = Math.round((0.5 - 1.5 / 4) * SIZE);
    const top = [red(px, bx, 2), red(px, bx, SIZE - 3)];
    expect(top.every((v) => Math.abs(v - 0.35 * 255) < 3)).toBe(true);
    // Slit centre at y = +0.4 -> uv.y = 0.5 - 0.4 / 4 = 0.4.
    const slitY = Math.round(0.4 * SIZE);
    expect(red(px, bx, slitY)).not.toBeCloseTo(0.35 * 255, -1);
  });

  needsGpu("every mode and view renders opaque, varying pixels", async (gpu) => {
    for (const mode of [0, 1, 2, 3]) {
      for (const viewMode of [0, 1, 2]) {
        const px = await render(gpu, { mode, viewMode });
        const values = new Set<number>();
        for (let i = 0; i < px.length; i += 4) {
          expect(px[i + 3]).toBe(255);
          values.add(px[i]);
        }
        expect(values.size, `mode ${mode} view ${viewMode}`).toBeGreaterThan(8);
      }
    }
  });
});

describe("Wave shader vs CPU twin", () => {
  // Test-only compute entry appended to the real shader source so the fragment shader's own
  // phasor() maths is evaluated at arbitrary points and read back.
  const probeSource =
    waveShader.wgsl +
    `
@group(0) @binding(1) var<storage, read_write> samples: array<vec4f>;
@compute @workgroup_size(8)
fn probe(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&samples)) { return; }
  let ph = phasor(samples[id.x].xy);
  samples[id.x] = vec4f(ph, 0.0, 0.0);
}`;
  const MODE_CODES: Record<WaveMode, number> = { point: 0, "two-points": 1, "single-slit": 2, "double-slit": 3 };
  const points: [number, number][] = [];
  for (const x of [-3, -1.6, -1.4, -0.2, 0.9, 2.5]) for (const y of [-1.8, -0.55, 0, 0.3, 1.4]) points.push([x, y]);

  for (const mode of Object.keys(MODE_CODES) as WaveMode[]) {
    needsGpu(`phasor matches the twin in ${mode} mode`, async (gpu) => {
      const p = { ...WAVE_DEFAULTS, mode, frequency: 4.5, damping: 0.03, separation: 0.9, slitWidth: 0.2 };
      const input = new Float32Array(points.length * 4);
      points.forEach(([x, y], i) => input.set([x, y, 0, 0], i * 4));
      const buffer = storage(gpu, input.byteLength);
      buffer.write(input);
      compute(gpu, probeSource, { entry: "probe" })
        .set({
          u: {
            resolution: [100, 100],
            time: 0,
            frequency: p.frequency,
            amplitude: p.amplitude,
            damping: p.damping,
            waveSpeed: p.waveSpeed,
            separation: p.separation,
            slitWidth: p.slitWidth,
            mode: MODE_CODES[mode],
            viewMode: 0,
          },
          samples: buffer,
        })
        .dispatch(Math.ceil(points.length / 8));
      const out = new Float32Array(await buffer.read());
      points.forEach(([x, y], i) => {
        const [c, s] = wavePhasor(p, x, y);
        expect(out[i * 4], `C at ${x},${y}`).toBeCloseTo(c, 3);
        expect(out[i * 4 + 1], `S at ${x},${y}`).toBeCloseTo(s, 3);
      });
    });
  }
});
