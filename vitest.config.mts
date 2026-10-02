import { wgslVitePlugin } from "@vgpu/wgsl/loader-vite";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [wgslVitePlugin()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    // The headless GPU tests drive a native (Dawn) device, which is safest in forked processes.
    pool: "forks",
    fileParallelism: false,
    testTimeout: 60_000,
  },
});
