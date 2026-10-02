# EduSim

GPU-powered interactive science simulations, running in the browser on WebGPU.
Built with Next.js 16 (App Router, Turbopack), React 19, Tailwind 4 and [vgpu](https://vgpu.sh).

| Simulation | Subject | GPU technique |
|---|---|---|
| Wave Interference & Diffraction | Physics | Fragment shader; Huygens–Fresnel slits as phasor sums |
| Hodgkin–Huxley Neuron | Biology | Compute shader integrates the ODEs; fragment shader plots the traces |
| Cellular Automata | Computer Science | Ping-pong compute shader over grids up to 2048², pan/zoom renderer |

The catalog in `src/lib/subjects.ts` lists many more planned simulations (marked "Coming Soon").
See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the improvement plan.

## Getting started

```bash
pnpm install
pnpm dev          # http://localhost:3000
```

You need a browser with WebGPU (recent Chrome, Edge or Safari). Without it, simulation pages show an
explanatory message instead of a blank canvas.

> This repo uses a newer Next.js than most training data covers — see `AGENTS.md` and read
> `node_modules/next/dist/docs/` before changing framework-level code.

## Scripts

| Command | What it does |
|---|---|
| `pnpm lint` | ESLint |
| `pnpm typecheck` | `next typegen` (route types) + `tsc --noEmit` |
| `pnpm test` | Vitest: pure logic **and** headless-GPU tests of every shader |
| `pnpm check:wgsl` | Validates every `.wgsl` file against a real WebGPU device (`next build` does not) |
| `pnpm build` | Production build (all simulation routes are statically generated) |
| `pnpm smoke` | Browser smoke test of a running build (`BASE_URL=http://localhost:3000`) |
| `pnpm verify` | lint → typecheck → test → check:wgsl → build |

### Running the GPU tests without a GPU

`pnpm test` and `pnpm check:wgsl` need a WebGPU adapter. On a machine without one (containers, CI):

```bash
npx vgpu install-software-renderer   # one-off: downloads a CPU Vulkan renderer
REQUIRE_GPU=1 pnpm test              # REQUIRE_GPU makes a missing adapter a failure instead of a skip
```

## Project layout

```
src/
  app/
    page.tsx                    landing page, grouped by subject
    [subject]/[sim]/page.tsx    the single simulation route (generateStaticParams, 404 for the rest)
  components/                   SimLayout, ParameterSlider, SimCard, GpuStatusOverlay
  lib/
    subjects.ts                 simulation catalog (single source of truth)
    gpu/runtime.ts              WebGPU lifecycle: init, surface, clock, safe teardown
    gpu/useGpuSim.ts            React hook wrapping the runtime
  simulations/
    loaders.tsx                 sim id -> lazily loaded, client-only component
    <subject>/<sim>/
      sim.ts                    creates the GPU resources and the frame loop; returns a handle
      controls.tsx              React UI built on SimLayout + useGpuSim
      *.wgsl                    shaders, loaded through vgpu's bundler loader
```

## Adding a simulation

1. Create `src/simulations/<subject>/<id>/` with `*.wgsl`, `sim.ts` (a `SimFactory` that returns a handle with
   `dispose()`), and `controls.tsx` (default export; use `useGpuSim(factory)` and `SimLayout`).
2. Set `implemented: true` for it in `src/lib/subjects.ts`.
3. Register it in `src/simulations/loaders.tsx`.

`src/lib/subjects.test.ts` fails if the catalog and the loaders disagree. Put pure logic in its own module
(see `hh.ts`, `rules.ts`) so it can be unit-tested, and add a headless-GPU test for any new shader in
`src/simulations/gpu-shaders.test.ts`.

## Conventions worth knowing

- **Lifecycle:** never create a GPU device outside `launchSim` / `useGpuSim`. Teardown must be synchronous and
  safe at any point, which is what makes React strict mode (double mount) leak-free.
- **Time:** use the frame clock (`frameDelta(ctx.clock)`), never a fixed per-frame increment, so speed does not
  depend on the display refresh rate.
- **vgpu `set()`:** the first call for a struct must include every member; later calls may send a subset.
- **Workgroups:** keep workgroup sizes small (≤ 64 invocations) so shaders are valid on every adapter.
- **Reserved words:** WGSL reserves words such as `meta`; `pnpm check:wgsl` will tell you.
