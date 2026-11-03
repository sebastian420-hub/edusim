/**
 * The slice of the WebGPU API the engine drives directly, typed structurally (the WebGPU type package is not part
 * of this project's type environment). Values come from vgpu's device (`gpu.gpu`), in the browser and under Node.
 */
export interface Buffer {
  destroy(): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
}
export type BindGroupLayout = object;
export type BindGroup = object;
export type ComputePipeline = object;

export interface ComputePass {
  setPipeline(p: ComputePipeline): void;
  setBindGroup(index: number, group: BindGroup, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

export interface CommandEncoder {
  beginComputePass(desc?: { label?: string }): ComputePass;
  copyBufferToBuffer(src: Buffer, srcOffset: number, dst: Buffer, dstOffset: number, size: number): void;
  finish(): object;
}

export interface Device {
  createBuffer(desc: { label?: string; size: number; usage: number }): Buffer;
  createBindGroupLayout(desc: { label?: string; entries: { binding: number; visibility: number; buffer: { type: "storage" | "uniform" | "read-only-storage"; hasDynamicOffset?: boolean; minBindingSize?: number } }[] }): BindGroupLayout;
  createPipelineLayout(desc: { label?: string; bindGroupLayouts: BindGroupLayout[] }): object;
  createShaderModule(desc: { label?: string; code: string }): object;
  createComputePipeline(desc: { label?: string; layout: object; compute: { module: object; entryPoint: string } }): ComputePipeline;
  createBindGroup(desc: { layout: BindGroupLayout; entries: { binding: number; resource: { buffer: Buffer; offset?: number; size?: number } }[] }): BindGroup;
  createCommandEncoder(desc?: { label?: string }): CommandEncoder;
  queue: {
    writeBuffer(buffer: Buffer, offset: number, data: ArrayBufferView | ArrayBuffer): void;
    submit(buffers: object[]): void;
  };
}

/** WebGPU flag values. */
export const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;
export const STAGE_COMPUTE = 0x4;
export const MAP_READ = 0x1;
