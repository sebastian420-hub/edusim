import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { compute, effect, frame, pingPongStorage, storage, target } from "vgpu/node";
import { tryInitGpu } from "@/test/gpu";
import type { NodeGpu } from "@/test/gpu";
import computeShader from "./cs/cellular-automata/compute.wgsl";
import countShader from "./cs/cellular-automata/count.wgsl";
import { countAlive, stateHash, stepLife } from "./cs/cellular-automata/life";
import { STATS_SLOTS } from "./cs/cellular-automata/sim-constants";
import { workgroupsFor } from "./cs/cellular-automata/workgroup";
import neuronShader from "./biology/hodgkin-huxley/neuron.wgsl";
import waveShader from "./physics/wave-interference/wave.wgsl";
import { DEFAULTS as WAVE_DEFAULTS, phasor as wavePhasor } from "./physics/wave-interference/wave";
import type { WaveMode } from "./physics/wave-interference/wave";
import { createEngine as createNBodyEngine } from "./physics/n-body/engine";
import { createRenderer as createNBodyRenderer } from "./physics/n-body/renderer";
import { computeAccelerations, G_ORBIT, rng as nbodyRng, stateFromBodies, step } from "./physics/n-body/nbody";
import { orbitPreset as nbodyPreset } from "./physics/n-body/presets";
import { TRAIL_LEN as NBODY_TRAIL_LEN } from "./physics/n-body/sim-constants";
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

describe("Cellular automata count/fingerprint shader vs CPU twin", () => {
  needsGpu("matches the CPU count and fingerprint for random soups, several slots, on a non-multiple-of-8 grid", async (gpu) => {
    // 70 is not a multiple of the 8x8 workgroup, so the bounds check and partial workgroups are exercised.
    const size = 70;
    const cells = storage(gpu, size * size * 4);
    const stats = storage(gpu, STATS_SLOTS * 8);
    const pass = compute(gpu, countShader);
    let seed = 12345;
    const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);

    for (const slot of [0, 7, STATS_SLOTS - 1]) {
      const grid = new Uint32Array(size * size);
      for (let i = 0; i < grid.length; i++) grid[i] = random() < 0.3 ? 1 : 0;
      cells.write(grid);
      (stats as unknown as { write(d: BufferSource, offset: number): void }).write(new Uint32Array([0, 0]), slot * 8);
      pass.set({ cells, stats, params: { width: size, height: size, slot, pad: 0 } }).dispatch(workgroupsFor(size), workgroupsFor(size), 1);
      const ring = new Uint32Array(await stats.read());
      expect(ring[slot * 2], `count in slot ${slot}`).toBe(countAlive(grid));
      expect(ring[slot * 2 + 1], `fingerprint in slot ${slot}`).toBe(stateHash(grid));
    }
  });

  needsGpu("fingerprints agree with the CPU after real generations (compute shader + count shader together)", async (gpu) => {
    const size = 32;
    const rule = { birth: 0b1000, survive: 0b1100 };
    const cells = pingPongStorage(gpu, size * size * 4);
    const stats = storage(gpu, STATS_SLOTS * 8);
    const step = compute(gpu, computeShader);
    const count = compute(gpu, countShader);
    let grid = new Uint32Array(size * size);
    for (const [x, y] of [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2], [10, 10], [11, 10], [12, 10]]) grid[y * size + x] = 1; // glider + blinker
    cells.read.write(grid);
    for (let g = 1; g <= 6; g++) {
      step.set({ cellsIn: cells.read, cellsOut: cells.write, params: { width: size, height: size, rule_birth: rule.birth, rule_survive: rule.survive } }).dispatch(workgroupsFor(size), workgroupsFor(size), 1);
      cells.swap();
      grid = stepLife(grid, size, size, rule.birth, rule.survive);
      (stats as unknown as { write(d: BufferSource, offset: number): void }).write(new Uint32Array([0, 0]), g * 8);
      count.set({ cells: cells.read, stats, params: { width: size, height: size, slot: g, pad: 0 } }).dispatch(workgroupsFor(size), workgroupsFor(size), 1);
      const ring = new Uint32Array(await stats.read());
      expect(ring[g * 2], `count after generation ${g}`).toBe(countAlive(grid));
      expect(ring[g * 2 + 1], `fingerprint after generation ${g}`).toBe(stateHash(grid));
    }
  });
});

describe("N-body engine (gravity + integrate shaders) vs CPU twin", () => {
  const random = nbodyRng(99);
  const randomState = (n: number) => {
    const s = { n, pos: new Float64Array(n * 4), vel: new Float64Array(n * 4), acc: new Float64Array(n * 4) };
    for (let i = 0; i < n; i++) {
      s.pos.set([random() * 4 - 2, random() * 4 - 2, random() * 0.2 - 0.1, 0.1 + random()], i * 4);
      s.vel.set([random() - 0.5, random() - 0.5, 0, 0], i * 4);
    }
    return s;
  };

  needsGpu("forces and potentials match for body counts around the 64-wide tiles (1, 2, 63, 64, 65, 1000)", async (gpu) => {
    const engine = createNBodyEngine(gpu, 1000, 0);
    for (const n of [1, 2, 63, 64, 65, 1000]) {
      const s = randomState(n);
      computeAccelerations(s, 1, 0.01);
      engine.upload(new Float32Array(s.pos), new Float32Array(s.vel), n, 1, 0.01);
      const snap = await engine.read();
      const scale = Math.max(1e-6, ...s.acc.map(Math.abs));
      for (let k = 0; k < n * 4; k++) expect(Math.abs(snap.acc[k] - s.acc[k]) / scale, `n=${n}, component ${k}`).toBeLessThan(2e-5);
    }
    engine.dispose();
  });

  needsGpu("one year of Earth's orbit (1000 leapfrog steps) lands where the twin does, in f32", async (gpu) => {
    const bodies = nbodyPreset("sun-earth").bodies;
    const twin = stateFromBodies(bodies);
    computeAccelerations(twin, G_ORBIT, 1e-8);
    const engine = createNBodyEngine(gpu, 8, 2);
    engine.upload(new Float32Array(twin.pos), new Float32Array(twin.vel), twin.n, G_ORBIT, 1e-8);
    for (let k = 0; k < 1000; k++) step(twin, 1e-3, G_ORBIT, 1e-8);
    engine.step(1e-3, 1000);
    const snap = await engine.read();
    expect(snap.time).toBeCloseTo(1, 9);
    expect(Math.hypot(snap.pos[4] - twin.pos[4], snap.pos[5] - twin.pos[5])).toBeLessThan(1e-3);
    expect(Math.hypot(snap.pos[4], snap.pos[5])).toBeCloseTo(1, 3);
    engine.dispose();
  });

  needsGpu("explicit Euler on the GPU drifts outwards exactly like the twin", async (gpu) => {
    const bodies = nbodyPreset("sun-earth").bodies;
    const twin = stateFromBodies(bodies);
    computeAccelerations(twin, G_ORBIT, 1e-8);
    const engine = createNBodyEngine(gpu, 8, 0);
    engine.upload(new Float32Array(twin.pos), new Float32Array(twin.vel), twin.n, G_ORBIT, 1e-8);
    for (let k = 0; k < 500; k++) step(twin, 2e-3, G_ORBIT, 1e-8, "euler");
    engine.step(2e-3, 500, "euler");
    const snap = await engine.read();
    expect(Math.hypot(snap.pos[4] - twin.pos[4], snap.pos[5] - twin.pos[5])).toBeLessThan(1e-3);
    expect(Math.hypot(snap.pos[4], snap.pos[5])).toBeGreaterThan(1.02);
    engine.dispose();
  });

  needsGpu("trail recording writes the newest position into the ring slot", async (gpu) => {
    const engine = createNBodyEngine(gpu, 8, 2);
    const s = randomState(3);
    engine.upload(new Float32Array(s.pos), new Float32Array(s.vel), 3, 1, 0.01);
    engine.recordTrail();
    engine.step(0.01, 3);
    engine.recordTrail();
    const trail = new Float32Array(await engine.buffers.trail.read());
    const snap = await engine.read();
    expect(engine.trailHead).toBe(2);
    expect(engine.trailFilled).toBe(2);
    // Body 1, slot 1 = its position now; body 2 has no trail (only 2 trail bodies).
    expect(trail[(1 * NBODY_TRAIL_LEN + 1) * 4]).toBeCloseTo(snap.pos[4], 6);
    expect(trail[(1 * NBODY_TRAIL_LEN + 1) * 4 + 3]).toBe(1);
    expect(trail[(1 * NBODY_TRAIL_LEN + 2) * 4 + 3]).toBe(0);
    engine.dispose();
  });

  needsGpu("instanced draw: each body appears as a disc at its position (world y up), trails as fading lines", async (gpu) => {
    const t = target(gpu, { size: [64, 64] });
    const engine = createNBodyEngine(gpu, 4, 3);
    // Three bodies at rest far apart (tiny G: they do not move noticeably).
    engine.upload(new Float32Array([0, 0, 0, 1, 10, 10, 0, 1, -10, 10, 0, 1]), new Float32Array(12), 3, 1e-9, 1e-6);
    const renderer = createNBodyRenderer(gpu, engine);
    renderer.setLooks(new Float32Array([1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 3]));
    renderer.setGlow(0);
    frame(gpu, (f) => f.pass({ target: t, clear: [0, 0, 0, 1] }, renderer.prepare(t, { center: [0, 0], pxPerUnit: 1 })));
    const px = await t.color.read({ mipLevel: 0, region: "all" });
    const at = (x: number, y: number) => [...px.slice((y * 64 + x) * 4, (y * 64 + x) * 4 + 3)];
    expect(at(32, 32)[0]).toBeGreaterThan(200); // red body at the centre
    expect(at(42, 21)[1]).toBeGreaterThan(200); // green: 10 right, 10 up
    expect(at(21, 21)[2]).toBeGreaterThan(200); // blue: 10 left, 10 up
    expect(at(32, 50)).toEqual([0, 0, 0]); // empty space stays black
    expect(at(5, 5)).toEqual([0, 0, 0]);

    // A trail: one body moving right, recorded every step, drawn as a line that fades towards the oldest end.
    const t2 = target(gpu, { size: [64, 64] });
    const e2 = createNBodyEngine(gpu, 2, 1);
    e2.upload(new Float32Array([-20, 0, 0, 1]), new Float32Array([1, 0, 0, 0]), 1, 1e-9, 1e-6);
    for (let k = 0; k < 10; k++) {
      e2.recordTrail();
      e2.step(2, 1);
    }
    const r2 = createNBodyRenderer(gpu, e2);
    r2.setLooks(new Float32Array([0.1, 1, 0.1, 0.01])); // a green body: its trail must be green
    frame(gpu, (f) => f.pass({ target: t2, clear: [0, 0, 0, 1] }, r2.prepare(t2, { center: [0, 0], pxPerUnit: 1 })));
    const p2 = await t2.color.read({ mipLevel: 0, region: "all" });
    const row = (x: number) => Math.max(p2[(31 * 64 + x) * 4 + 1], p2[(32 * 64 + x) * 4 + 1]);
    // Recorded at x = −20 … −2, i.e. pixels 12 … 30: brightest at the newest end.
    expect(row(28)).toBeGreaterThan(row(14));
    expect(row(14)).toBeGreaterThan(0);
    expect(row(8)).toBe(0); // nothing before the oldest point
    expect(p2[(31 * 64 + 28) * 4 + 1] + p2[(32 * 64 + 28) * 4 + 1]).toBeGreaterThan(3 * (p2[(31 * 64 + 28) * 4] + p2[(32 * 64 + 28) * 4]));
    expect(row(40)).toBe(0); // nothing beyond the body
    renderer.dispose();
    r2.dispose();
    engine.dispose();
    e2.dispose();
  });
});
