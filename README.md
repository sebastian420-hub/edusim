# EduSim

GPU-powered interactive science simulations, running in the browser on WebGPU.
Built with Next.js 16 (App Router, Turbopack), React 19, Tailwind 4 and [vgpu](https://vgpu.sh).

| Simulation | Subject | GPU technique |
|---|---|---|
| Wave Interference & Diffraction | Physics | Fragment shader; Huygens–Fresnel slits as phasor sums |
| N-Body Orbital Mechanics | Physics | Tiled all-pairs gravity compute shader (up to 16 384 bodies), leapfrog integrator, instanced rendering |
| Hodgkin–Huxley Neuron | Biology | Compute shader integrates the ODEs; fragment shader plots the traces |
| Cellular Automata | Computer Science | Ping-pong compute shader over grids up to 2048²; a reduction shader counts and fingerprints every generation; pan/zoom renderer |

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
| `pnpm smoke` | Browser smoke test of a running server (`BASE_URL=http://localhost:3000`) |
| `pnpm e2e` | Full browser suite (53 scenarios: every sim, URL state, measurement tools, challenges, shortcuts, no-WebGPU, phone) against `BASE_URL`. Works on `pnpm start`, `pnpm dev` with `E2E_DEV=1` (React strict mode; skips size budgets) and a static export. `E2E_GREP=<regex>` runs only the scenarios whose name matches |
| `pnpm smoke:local` | Builds, serves on a spare port, runs the smoke test, stops the server |
| `pnpm export` | Fully static offline build in `out/` — serve with any static file server, no Node or internet |
| `pnpm verify` | lint → typecheck → test → check:wgsl → build |

### Git hooks

`pnpm install` enables `.githooks/pre-push`, which runs `pnpm verify` before every push so a broken build
never reaches GitHub (GitHub Actions may be unavailable). Bypass in an emergency with `git push --no-verify`.

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

## Shared features for simulations

- **Shareable state:** `usePersistedParams(simId, schema, defaults)` keeps settings in the URL (and, as a fallback,
  localStorage). Schemas in `lib/urlState.ts` clamp/validate everything, so a bad link can't break a sim.
- **`SimLayout`:** copy link, save image, reset to defaults, keyboard shortcuts (Space, `.`, `R`, `?`) and an
  optional Challenges tab.
- **Challenges:** data in `lib/challenges.ts` format (predict → set up → goal check → explanation), rendered by
  `ChallengesPanel`. See `challenges.ts` next to each simulation. Each has a test that its setup does not already
  solve it, that the intended answer does, and that obvious shortcuts do not.
- **Measurement:** sims get a TypeScript twin of the shader (`wave.ts`, `hh.ts`, `life.ts`) used for graphs,
  readouts and tests, and a GPU test proving the two agree. `MiniChart` is a small SVG chart for readouts like the
  firing-rate curve and the population graph.
- **N-body:** `gravity.wgsl` sums softened gravity over all pairs in 64-body tiles of workgroup memory, `integrate.wgsl`
  does leapfrog (or Euler, for the numerics lesson). `engine.ts` is the GPU physics without drawing, shared by the sim
  and the GPU tests; `renderer.ts` copies positions into vertex buffers (vertex shaders may not read storage buffers
  on compatibility-mode devices) and draws instanced discs and fading trails. Energies, orbital elements and measured
  periods come from small read-backs, sampled every ≤ 16 steps so fast planets are never aliased. The orbit lab's
  units are AU, years and solar masses (G = 4π²), so Earth's period is exactly 1 year.
- **Cellular automata measurement:** `count.wgsl` reduces the grid to a live-cell count and an order-independent
  fingerprint every generation (into a 1024-slot ring buffer); `sim.ts` reads it back at most every 100 ms and
  `tracker.ts` classifies the history (extinct / still life / oscillator with period / moving pattern). It only runs
  when a host passes `onStats`, so the home-page preview pays nothing. A shared link reproduces the rules, grid,
  speed, colours and the *starting pattern* (a library pattern, random soup or empty grid) — not hand-drawn cells.

## The home page

A compact editorial layout (about one screen on a desktop, 1.35 on a phone): numbered *plates* for the working
simulations and a typographic index of the planned ones. Design tokens (colours, type scale, motion) live in
`src/app/globals.css`; components in `src/components/home/`. The plan and measurements are in
[`docs/HOMEPAGE-PLAN.md`](docs/HOMEPAGE-PLAN.md).

- **Posters.** Each plate shows a poster captured from the *real* simulation: `pnpm posters` builds, serves, drives
  each simulation from fixed settings, reads only the canvas pixels, and writes `public/plates/<id>.webp` plus the
  social-preview image `src/app/opengraph-image.jpg`. Re-run it after changing how a simulation looks.
- **Live on intent.** Hovering or keyboard-focusing a plate runs the real simulation inside it
  (`simulations/previews.ts`, loaded only then). Rules, in `components/home/intent.ts`: one live plate at a time;
  never on touch, with `prefers-reduced-motion` or without WebGPU; stops when scrolled out of view or the tab is
  hidden; a GPU failure leaves the poster. The home page uses no GPU until the visitor shows intent.
- **Budgets** (enforced by `pnpm e2e`): ≤ 1.15 screens tall on desktop, ≤ 1.4 on a phone, no horizontal overflow,
  JavaScript ≤ 470 KB (the React + Next runtime is ~457 KB of that), HTML ≤ 14 KB gzipped, zero axe violations.

## Adaptive quality

Simulations run through a paced render loop (`ctx.loop`): it skips ticks while the GPU is still busy, so a slow
device never builds a backlog (Pause/sliders stay responsive), and a governor (`lib/gpu/pacing.ts`) lowers the
canvas resolution — down to 35% — when frames cost too much, raising it again only when the predicted cost fits.
A badge shows when resolution is reduced. To pin full resolution (projector, screenshots), run
`localStorage.setItem("edusim:quality", "full")` in the browser console.

## Adding a simulation

1. Create `src/simulations/<subject>/<id>/` with `*.wgsl`, `sim.ts` (a `SimFactory` that returns a handle with
   `dispose()`), and `controls.tsx` (default export; use `useGpuSim(factory)` and `SimLayout`).
2. Set `implemented: true` for it in `src/lib/subjects.ts`, with a `tagline` and `posterAlt`.
3. Register it in `src/simulations/loaders.tsx` and add its home-page preview to `src/simulations/previews.ts`.
4. Run `pnpm posters` to generate its plate poster (add its capture settings to `scripts/make-posters.mjs`).

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
