/**
 * A language model on the GPU: the plan's four buffers and its programs (plan.ts), each program run as one compute
 * pass in one command buffer. vgpu's compute() submits every dispatch on its own, which is too slow for the ~60–110
 * kernels of a transformer step, so the engine drives WebGPU directly: one explicit bind group layout shared by every
 * kernel of lm.wgsl, and each program's op records (model.ts) written once into a uniform buffer read at dynamic
 * offsets. Shared by the simulation and the headless GPU tests; the CPU twin (twin.ts) has the same operations.
 */
import type { Gpu } from "vgpu";
import lmShader from "./lm.wgsl";
import { invocations, KERNELS, OP_F32_FIELDS, OP_U32_FIELDS, stepValues } from "./model";
import type { Kernel, ModelSpec, Op, Program } from "./model";
import { lineStart, makePlan, predictOf, writeOf } from "./plan";
import type { Plan, PlanOptions } from "./plan";
import { CONTEXT_MARGIN } from "./text";
import { MAP_READ, STAGE_COMPUTE, USAGE } from "./webgpu";
import type { BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device } from "./webgpu";

const SLOT = 256; // bytes per op record (the uniform offset alignment WebGPU guarantees)
const MAX_GROUPS = 65535;

interface Kernels {
  layout: BindGroupLayout;
  pipelines: Record<Kernel, ComputePipeline>;
}

// Pipelines are per device and shared by every engine on it (an engine is rebuilt whenever the model changes).
const kernelCache = new WeakMap<Device, Kernels>();

function kernelsFor(device: Device): Kernels {
  const cached = kernelCache.get(device);
  if (cached) return cached;
  const storageEntry = (binding: number): Parameters<Device["createBindGroupLayout"]>[0]["entries"][number] => ({ binding, visibility: STAGE_COMPUTE, buffer: { type: "storage" } });
  const layout = device.createBindGroupLayout({
    label: "lm",
    entries: [storageEntry(0), storageEntry(1), storageEntry(2), storageEntry(3), { binding: 4, visibility: STAGE_COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: SLOT } }],
  });
  const pipelineLayout = device.createPipelineLayout({ label: "lm", bindGroupLayouts: [layout] });
  const shaderModule = device.createShaderModule({ label: "lm.wgsl", code: lmShader.wgsl });
  const pipelines = Object.fromEntries(KERNELS.map((k) => [k, device.createComputePipeline({ label: `lm.${k}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint: k } })])) as Record<Kernel, ComputePipeline>;
  const kernels = { layout, pipelines };
  kernelCache.set(device, kernels);
  return kernels;
}

/** Workgroups for an op: matmul tiles in 2D, everything else 64 invocations per group, spilling into y. */
export function groupsOf(o: Op): [number, number] {
  if (o.kernel === "matmul") return [Math.ceil(o.N / 32), Math.ceil(o.M / 32)];
  const groups = Math.max(1, Math.ceil(invocations(o) / 64));
  const x = Math.min(groups, MAX_GROUPS);
  return [x, Math.ceil(groups / x)];
}

/** The op records of a program, one 256-byte slot each, as the shader's `Op` struct. */
export function encodeOps(ops: readonly Op[]): ArrayBuffer {
  const bytes = new ArrayBuffer(Math.max(1, ops.length) * SLOT);
  ops.forEach((o, k) => {
    const u32 = new Uint32Array(bytes, k * SLOT, OP_U32_FIELDS.length);
    OP_U32_FIELDS.forEach((f, i) => {
      const v = o[f];
      if (!Number.isInteger(v) || v < 0 || v > 0xffffffff) throw new Error(`op ${o.kernel}: field ${f} = ${v} is not a u32`);
      u32[i] = v;
    });
    const f32 = new Float32Array(bytes, k * SLOT + 4 * OP_U32_FIELDS.length, OP_F32_FIELDS.length);
    OP_F32_FIELDS.forEach((f, i) => (f32[i] = o[f]));
  });
  return bytes;
}

export function createEngine(gpu: Gpu, spec: ModelSpec, text: Uint8Array, opts: PlanOptions) {
  const plan: Plan = makePlan(spec, text.length, opts);
  const device = gpu.gpu as unknown as Device;
  const { layout, pipelines } = kernelsFor(device);
  const usage = USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC;
  const buffer = (label: string, floats: number) => device.createBuffer({ label, size: Math.max(16, Math.ceil((floats * 4) / 16) * 16), usage });
  const params = buffer("lm.params", plan.lay.P);
  const work = buffer("lm.work", plan.work.size);
  const opt = buffer("lm.opt", plan.opt.size);
  const tokens = buffer("lm.tokens", plan.tokens.size);
  const owned: Buffer[] = [params, work, opt, tokens];
  device.queue.writeBuffer(tokens, 0, Uint32Array.from(text));

  interface Compiled {
    ops: readonly Op[];
    uniform: Buffer;
    group: BindGroup;
    groups: [number, number][];
  }
  const compile = (program: Program | readonly Op[]): Compiled => {
    const ops = "ops" in program ? program.ops : program;
    const data = encodeOps(ops);
    const uniform = device.createBuffer({ label: "lm.ops", size: data.byteLength, usage: USAGE.UNIFORM | USAGE.COPY_DST });
    device.queue.writeBuffer(uniform, 0, data);
    owned.push(uniform);
    const group = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: work } },
        { binding: 2, resource: { buffer: opt } },
        { binding: 3, resource: { buffer: tokens } },
        { binding: 4, resource: { buffer: uniform, size: SLOT } },
      ],
    });
    return { ops, uniform, group, groups: ops.map(groupsOf) };
  };
  const release = (c: Compiled) => {
    c.uniform.destroy();
    owned.splice(owned.indexOf(c.uniform), 1);
  };
  /** Encodes compiled programs (each `times` over) into one compute pass and submits it. */
  const run = (...list: [Compiled, number][]) => {
    const encoder = device.createCommandEncoder({ label: "lm" });
    const pass = encoder.beginComputePass({ label: "lm" });
    for (const [c, times] of list) {
      for (let n = 0; n < times; n++) {
        c.ops.forEach((o, k) => {
          pass.setPipeline(pipelines[o.kernel]);
          pass.setBindGroup(0, c.group, [k * SLOT]);
          pass.dispatchWorkgroups(c.groups[k][0], c.groups[k][1]);
        });
      }
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  };
  /**
   * Copies ranges of 32-bit words ([from, count] each) out of a buffer. The copies are submitted at once, before
   * anything else can be queued: later programs reuse the same memory, so a copy must not wait for an await.
   */
  const readRanges = async (src: Buffer, ranges: readonly [number, number][]): Promise<ArrayBuffer[]> => {
    const total = ranges.reduce((a, [, n]) => a + Math.max(1, n), 0);
    const staging = device.createBuffer({ label: "lm.read", size: total * 4, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    const encoder = device.createCommandEncoder();
    let at = 0;
    for (const [from, n] of ranges) {
      if (n > 0) encoder.copyBufferToBuffer(src, from * 4, staging, at * 4, n * 4);
      at += Math.max(1, n);
    }
    device.queue.submit([encoder.finish()]);
    try {
      await staging.mapAsync(MAP_READ);
      const all = staging.getMappedRange();
      const out: ArrayBuffer[] = [];
      at = 0;
      for (const [, n] of ranges) {
        out.push(all.slice(at * 4, at * 4 + n * 4));
        at += Math.max(1, n);
      }
      return out;
    } finally {
      staging.destroy();
    }
  };
  const read = async (src: Buffer, from: number, count: number): Promise<ArrayBuffer> => (await readRanges(src, [[from, count]]))[0];

  const train = compile(plan.train);
  const evals = plan.evalChunks.map(compile);
  let writer: Compiled | null = null;
  let predictors = new Map<number, { compiled: Compiled; program: Program }>();
  let steps = 0;
  let writtenCount = 0;
  let disposed = false;

  return {
    spec,
    plan,
    /** Training steps since the parameters were loaded. */
    get steps() {
      return steps;
    },

    /** Loads parameters and clears the optimiser's state. */
    load(p: ArrayLike<number>) {
      if (p.length !== plan.lay.P) throw new Error(`expected ${plan.lay.P} parameters, got ${p.length}`);
      device.queue.writeBuffer(params, 0, Float32Array.from(p as ArrayLike<number>));
      device.queue.writeBuffer(opt, 0, new Float32Array(plan.opt.size));
      steps = 0;
    },

    /** One AdamW step on the B windows starting at `starts`. */
    step(starts: ArrayLike<number>, lr: number) {
      if (starts.length !== plan.B) throw new Error(`a step takes ${plan.B} windows`);
      device.queue.writeBuffer(tokens, plan.tokens.startsTrain * 4, Uint32Array.from(starts));
      device.queue.writeBuffer(opt, plan.opt.step * 4, Float32Array.from(stepValues(steps, lr)));
      run([train, 1]);
      steps++;
    },

    /** Per-row losses of the E evaluation windows (window-major, T per window). */
    async evaluate(starts: ArrayLike<number>): Promise<Float32Array> {
      this.evaluateChunks(starts, 0, evals.length);
      return this.readLosses();
    },
    /** Evaluation chunks: how many, and running chunks [from, to) (writing the window starts when from = 0). */
    get evalChunks() {
      return evals.length;
    },
    evaluateChunks(starts: ArrayLike<number>, from: number, to: number) {
      if (starts.length !== plan.E) throw new Error(`evaluation takes ${plan.E} windows`);
      if (from === 0) device.queue.writeBuffer(tokens, plan.tokens.startsEval * 4, Uint32Array.from(starts));
      run(...evals.slice(from, to).map((c): [Compiled, number] => [c, 1]));
    },
    /** The losses of the last evaluation (window-major, T per window). */
    async readLosses(): Promise<Float32Array> {
      return new Float32Array(await read(work, plan.work.losses, plan.E * spec.T));
    },

    /** Starts G lines of writing from their contexts (margin + T letters each). */
    startWriting(contexts: readonly ArrayLike<number>[], temperature: number, seed: number) {
      for (let g = 0; g < plan.G; g++) device.queue.writeBuffer(tokens, lineStart(plan, g) * 4, Uint32Array.from(contexts[g]));
      device.queue.writeBuffer(tokens, plan.tokens.startsWrite * 4, Uint32Array.from({ length: plan.G }, (_, g) => lineStart(plan, g) + CONTEXT_MARGIN));
      device.queue.writeBuffer(tokens, plan.tokens.counter * 4, new Uint32Array([0]));
      writtenCount = 0;
      this.setWriting(temperature, seed);
    },
    /** Changes the temperature (or the random draws) of the writing in progress. */
    setWriting(temperature: number, seed: number) {
      if (writer) release(writer);
      writer = compile(writeOf(plan, temperature, seed));
    },
    /** Letters written on each line so far (at most W). */
    get writtenCount() {
      return writtenCount;
    },
    /** Writes n more letters on every line (stopping at W), without leaving the GPU. */
    write(n: number) {
      const k = Math.min(n, plan.W - writtenCount);
      if (writer && k > 0) run([writer, k]);
      writtenCount += Math.max(0, k);
    },
    /** The letters written so far on each line. */
    async written(): Promise<Uint32Array[]> {
      const all = new Uint32Array(await read(tokens, plan.tokens.write, plan.tokens.counter + 1 - plan.tokens.write));
      const n = all[plan.tokens.counter - plan.tokens.write];
      return Array.from({ length: plan.G }, (_, g) => {
        const from = lineStart(plan, g) - plan.tokens.write + CONTEXT_MARGIN + spec.T;
        return all.slice(from, from + Math.min(n, plan.W));
      });
    },

    /** The model's surprise (−ln p, nats) at each letter it wrote, per line (W slots; the first `written` are set). */
    async surprise(): Promise<Float32Array[]> {
      const all = new Float32Array(await read(work, plan.work.surprise, plan.G * plan.W));
      return Array.from({ length: plan.G }, (_, g) => all.slice(g * plan.W, (g + 1) * plan.W));
    },

    /** The scores for the letter after `prompt` (its last ≤ T letters), and each layer's attention [H][T][T]. */
    async predict(prompt: readonly number[], pad: number): Promise<{ logits: Float32Array; attention: Float32Array[]; T: number }> {
      const T = Math.max(1, Math.min(spec.T, prompt.length));
      const window = prompt.length ? prompt.slice(-T) : [pad];
      const region = plan.tokens.prompt;
      const data = new Uint32Array(CONTEXT_MARGIN + T).fill(pad);
      data.set(window, CONTEXT_MARGIN);
      device.queue.writeBuffer(tokens, region * 4, data);
      device.queue.writeBuffer(tokens, plan.tokens.startsPredict * 4, new Uint32Array([region + CONTEXT_MARGIN]));
      let entry = predictors.get(T);
      if (!entry) {
        const program = predictOf(plan, T);
        entry = { program, compiled: compile(program) };
        predictors.set(T, entry);
      }
      run([entry.compiled, 1]);
      const V = spec.V;
      const [logits, ...attention] = await readRanges(work, [[entry.program.acts.logits + (T - 1) * V, V], ...entry.program.acts.attention.map((p): [number, number] => [p, spec.H * T * T])]);
      return { logits: new Float32Array(logits), attention: attention.map((a) => new Float32Array(a)), T };
    },

    async readParams(): Promise<Float32Array> {
      return new Float32Array(await read(params, 0, plan.lay.P));
    },
    /** The whole optimiser buffer (gradient, Adam's moments, norm scratch). */
    async readOpt(): Promise<Float32Array> {
      return new Float32Array(await read(opt, 0, plan.opt.size));
    },
    async readWork(from: number, count: number): Promise<Float32Array> {
      return new Float32Array(await read(work, from, count));
    },
    async readTokens(from: number, count: number): Promise<Uint32Array> {
      return new Uint32Array(await read(tokens, from, count));
    },
    /** Runs a raw op list once (tests: one kernel at a time). */
    runOps(ops: readonly Op[]) {
      const c = compile(ops);
      run([c, 1]);
      release(c);
    },
    writeBuffer(which: "params" | "work" | "opt" | "tokens", data: Float32Array | Uint32Array, at = 0) {
      device.queue.writeBuffer({ params, work, opt, tokens }[which], at * 4, data as Float32Array<ArrayBuffer>);
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      for (const b of owned) b.destroy();
      owned.length = 0;
      predictors = new Map();
      writer = null;
    },
  };
}

export type Engine = ReturnType<typeof createEngine>;
