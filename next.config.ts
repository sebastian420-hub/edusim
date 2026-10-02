import type { NextConfig } from "next";

const nextConfig: NextConfig = {
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
