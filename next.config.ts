import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `pnpm export` (EXPORT=1) writes a fully static site to out/ that runs from any static file server,
  // with no Node and no internet — for classrooms, USB sticks and school networks.
  ...(process.env.EXPORT === "1" ? { output: "export" as const, trailingSlash: true } : {}),
  // Load shaders from real .wgsl files (resolved at build time by vgpu's loader).
  turbopack: {
    rules: {
      "*.wgsl": {
        loaders: ["@vgpu/wgsl/loader-webpack"],
        as: "*.js",
      },
    },
  },
};

export default nextConfig;
