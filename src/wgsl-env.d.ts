// `.wgsl` imports resolve (via vgpu's bundler loader, see next.config.ts) to a ShaderSource object.
declare module "*.wgsl" {
  import type { ShaderSource } from "vgpu";
  const source: ShaderSource;
  export default source;
}
