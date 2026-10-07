# N-Body Orbital Mechanics — research and implementation plan

> **Status: implemented** (orbit lab + galaxies, 4 challenges). Where the build differs from this plan:
> - Energies and momenta are computed on the CPU from a read-back instead of a GPU tree reduction: at ≤ 16 384
>   bodies the read-back is ≤ 768 KB a few times a second, simpler, and checked by the same twin functions.
> - Positions reach the vertex shader through vertex buffers (a GPU-to-GPU copy each frame), because
>   compatibility-mode adapters allow no storage buffers in vertex shaders — found by the GPU tests.
> - Trails are 1-px line strips with alpha blending (additive blending saturated orbits traced many times).
> - No first-run benchmark or close-encounter guard: a per-frame cap on pair interactions keeps big clouds
>   responsive, and softening plus a small fixed step handle close passes in the orbit lab.
> - Orbit-lab samples are taken after every ≤ 16 steps: sampling once per frame aliased Mercury's period on slow GPUs.

The fourth simulation (catalog id `n-body`, route `/physics/n-body`, already listed as "in preparation").
It must ship at the same bar as the other three: a TypeScript twin of every shader, GPU-vs-twin tests,
shareable settings, a measurement the tests prove correct, three guided challenges, e2e scenarios, a poster
and a live home-page preview.

## 1. What students should get out of it

Learning goals, taken from the well-tested PhET gravity simulations (*Gravity and Orbits*, *My Solar System*)
and the usual secondary/first-year curriculum:

1. Gravity controls orbital motion; the force depends on both masses and on distance (inverse square).
2. Predict the speed a planet needs for a **circular orbit** at a given distance; too slow → ellipse that
   dips inward, too fast → wider ellipse, fast enough → **escape** (total energy ≥ 0).
3. **Kepler's third law**: the period grows as distance^1.5 (T² ∝ a³) — measured, not just told.
4. **Conservation**: energy swaps between kinetic and potential, total stays constant; momentum and angular
   momentum are conserved.
5. Many bodies: three-body chaos, and (stretch) colliding galaxies — what the GPU makes possible.
6. Bonus for older students: *the numerical method matters* — a naive integrator (Euler) makes orbits spiral
   out and energy drift; a symplectic one (leapfrog) keeps the energy error bounded. That is a real lesson
   about how simulations work, and the energy graph makes it visible.

What PhET does not do and we can: thousands of bodies on the GPU, live measured graphs, and challenges whose
goals are checked against measured quantities.

## 2. Product shape: one simulation, two modes

| | **Orbit lab** (default) | **Star cluster / galaxies** |
|---|---|---|
| Bodies | 2–6, each with mass, position, velocity | 1 024 – 16 384 (32 768 on fast GPUs) |
| Interaction | drag a body to move it, drag its arrow to set velocity | pick a preset, pan/zoom, play |
| Visuals | bodies sized by mass, fading orbit trails, velocity arrows | additive-blended glowing points |
| Measurements | period, semi-major axis, eccentricity, energy graph, T²–a³ plot | total energy & angular momentum graph, kinetic vs potential |
| Presets | Sun–Earth, Sun–Earth–Moon, binary star, figure-eight 3-body, Sun–Jupiter–comet | Plummer cluster, rotating disk, two colliding disk galaxies |

Both modes share one physics engine (same shaders), which keeps the test surface small.

**Units** (orbit lab): astronomical units, years, solar masses, so G = 4π². Earth at 1 AU with 2π AU/yr has a
period of exactly 1 year — numbers students can check. Galaxy mode uses dimensionless units (G = 1).

## 3. Physics engine

- **Integrator:** leapfrog kick–drift–kick (velocity Verlet), symplectic: bounded energy error, exact momentum
  conservation, time-reversible. An optional **Euler** switch exists only for the "numerics matter" challenge.
- **Softening:** Plummer softening, `a = G m r / (|r|² + ε²)^{3/2}`, so close encounters don't blow up.
  ε is tiny in the orbit lab (exact Kepler behaviour) and ~0.01–0.05 of the system size in galaxy mode.
- **Time step:** fixed dt per substep, several substeps per frame (like the CA's steps-per-frame cap), so
  speed doesn't depend on frame rate. The orbit lab uses many small steps (cheap with ≤ 6 bodies); a
  **close-encounter guard** reduces dt when the closest pair gets too close (measured each step on the GPU).
- **Precision:** WGSL has no f64. f32 with centre-of-mass-relative coordinates and the units above keeps a
  1-year orbit accurate to < 0.1 % over hundreds of orbits (to be confirmed by the twin tests — see §6).
- **Collisions:** none in v1 (bodies pass through with softening); merging bodies is a possible later addition.

### GPU layout

- Bodies in storage buffers: `pos_mass: array<vec4f>` (xyz + mass) and `vel: array<vec4f>`. 3-D storage,
  2-D simulation (z = 0) for v1, so a 3-D camera can come later without changing the data.
- **Force kernel (`gravity.wgsl`)**: classic tiled O(N²) "all pairs" algorithm (GPU Gems 3, ch. 31): each
  workgroup of 64 loads a tile of 64 bodies into `var<workgroup>` memory, every invocation accumulates the
  force from the tile, then the next tile. Bounds checks for N not a multiple of 64. The same loop also
  accumulates each body's **potential energy** (free — the distances are already computed), used by the
  energy readout.
- **Kick/drift kernel(s)**: half-kick, drift, force, half-kick. Pure per-body updates.
- **Reduction kernel (`energy.wgsl`)**: kinetic + potential energy, momentum, angular momentum, centre of mass
  and closest-pair distance into a small ring buffer per step — exactly the pattern proven by the CA's
  `count.wgsl` (read back ≤ 10×/s, epoch-guarded, only when a host asks for stats). Floats can't use WGSL
  atomics, so this is a two-level tree reduction (workgroup → partial sums → one final workgroup).
- **Trails (orbit lab)**: a ring buffer of the last ~2 000 positions per body, written by the drift kernel,
  drawn as line strips with fading alpha.
- **Rendering**: vgpu's `draw()` with instancing (`instances: N`, additive blending), one quad per body read
  from the storage buffer — no CPU round trip. This is the first use of `draw()` in the codebase, so it is
  spiked first (§7, phase 0). Pan/zoom reuses the CA's view maths.

### Cost

16 384 bodies = 268 M pair interactions per step. A mid-range integrated GPU does ~200–500 G simple flops/s,
so ~20 flops per pair gives 1–3 steps per frame at 60 fps. Hence: default N chosen by a quick benchmark on
first run (1 024 / 4 096 / 16 384), a "bodies" selector, and the existing quality governor for resolution.
The orbit lab is trivially cheap.

## 4. Measurements (the core pedagogy)

| Readout | How | Proven by |
|---|---|---|
| Total, kinetic, potential energy; graph over time | GPU reduction | twin test: equals direct CPU sum |
| Energy drift % since start | from the graph | twin: leapfrog bounded, Euler grows |
| Angular momentum, centre-of-mass drift | GPU reduction | conserved to f32 precision |
| Orbit period, semi-major axis a, eccentricity e (per planet, orbit lab) | orbital elements from relative position/velocity (vis-viva) on the CPU from the read-back of ≤ 6 bodies, plus measured period from angle crossings | twin: Earth preset gives T = 1.000 yr, a = 1, e ≈ 0 |
| Kepler plot T² vs a³ | each completed orbit adds a point; dashed line = theory | twin: points fall on the line within 0.5 % |
| Bound / escaping | sign of the planet's orbital energy | twin: at v = √2·v_circ it escapes |

## 5. Shareable settings and challenges

**URL state** (`usePersistedParams`, as for the other sims): mode, preset, N, speed, integrator, softening,
trails, graph switches. In the orbit lab the bodies themselves (mass, x, y, vx, vy for ≤ 6 bodies) are
encoded compactly, so a shared link reproduces an exact initial setup — unlike the CA, every state here is
small enough to share.

**Challenges** (goals read the measured values, can't be shortcut, every claim tested on the twin):

1. **Circular orbit** — "The planet starts 2 AU from the Sun. Set its speed so the orbit is a circle."
   Prediction: faster or slower than Earth? (slower: v ∝ 1/√r). Goal: e < 0.05 over a full measured orbit.
2. **Kepler's third law** — "Earth takes 1 year. Predict the period of a planet 4× as far away." Options:
   4, 8, 16 years (8). Goal: measured period at a = 4 AU within 3 % of 8 years, with ≥ 1 full orbit measured.
3. **Escape** — "Find the slowest launch speed that never comes back." Goal: orbital energy ≥ 0 and speed
   within 5 % of √2·v_circ (so "just throw it very fast" doesn't count).
4. *(galaxy mode, optional)* **Numerics matter** — switch to Euler and predict what happens to the energy;
   goal: observe energy drift > 5 %, explanation of symplectic integration.

## 6. Testing (same standard as the existing sims)

- `nbody.ts` — CPU twin: identical leapfrog/softening maths, in f32 via `Math.fround`/`Float32Array`, plus
  orbital-element helpers. Unit tests: two-body circular orbit period = 2π√(a³/GM) to 0.1 %; energy error
  bounded over 1 000 orbits (leapfrog) vs growing (Euler); momentum exactly conserved; escape at √2·v_circ;
  figure-eight 3-body stays periodic for several periods; the presets are what their names say.
- GPU tests (`gpu-shaders.test.ts`): force kernel vs twin for N = 1, 2, 63, 64, 65, 1 000 (tile edges);
  K leapfrog steps vs twin within f32 tolerance; energy reduction vs direct CPU sum.
- `challenges.test.ts`: setup not already solved, intended answer solves it, shortcuts rejected.
- e2e: URL state + hostile links, presets render without GPU errors, readouts (Earth period ≈ 1 yr measured
  in the browser), drag-to-set-velocity, challenge flows, galaxy mode at small N (the software renderer is
  slow), phone layout, navigation churn, no-WebGPU message.
- Budgets: home JS unchanged (the sim is lazily loaded), frame pacing / quality governor still apply.

## 7. Phases and estimate (≈ 6–8 working days)

| Phase | Work | Done when |
|---|---|---|
| 0. Spike (½ day) | instanced `draw()` of points from a storage buffer with additive blending; tiled force kernel valid on a 128-invocation software adapter | both render/validate in the e2e browser and `check:wgsl` |
| 1. Engine (1½ days) | `nbody.ts` twin, `gravity.wgsl`, kick/drift, `energy.wgsl`, GPU tests | twin and GPU agree; conservation tests pass |
| 2. Orbit lab (2 days) | `sim.ts`, renderer + trails, drag body / velocity arrow, presets, readouts, energy graph, Kepler plot, URL state | Earth preset measures T = 1.00 yr in the browser |
| 3. Galaxy mode (1–1½ days) | Plummer / disk / colliding-galaxies initial conditions, N selector + first-run benchmark, glow rendering | 16 k bodies smooth on a laptop GPU, no errors on the software renderer at 1 k |
| 4. Challenges & polish (1–1½ days) | 3–4 challenges, explanations, e2e scenarios, catalog `implemented: true`, poster, home preview, docs | `pnpm verify` + full e2e green on prod, dev and static export |

## 8. Decisions needed

1. **Home page with four plates.** The home page was designed around three numbered plates. A fourth means a
   2 × 2 grid (or a row of four on wide screens); it has to keep the "about one screen" budget. Recommended:
   four plates in one row on desktop, horizontal scroll on phones (as now).
2. **Galaxy mode in v1, or orbit lab first?** Recommended: both, but if time is short, ship the orbit lab
   (the curriculum value) first and galaxy mode (the GPU showcase) as a follow-up PR.
3. **2-D only for v1** (recommended; data is already 3-D so a 3-D camera can follow).

## Sources

- PhET, *Gravity and Orbits* and *My Solar System* — learning goals and interaction model:
  <https://www.elmirador.edu.co/contenidos/phet/en/simulation/gravity-and-orbits.html>,
  <https://www.elmirador.edu.co/contenidos/phet/en/simulation/my-solar-system.html>
- Symplectic leapfrog (kick–drift–kick), bounded energy error: Dehnen & Read, *N-body simulations of
  gravitational dynamics* <https://arxiv.org/pdf/1105.1082>; Hernandez & Bertschinger, *Symplectic integration
  for the collisional gravitational N-body problem* <https://arxiv.org/pdf/1503.02728>
- Tiled all-pairs force kernel in workgroup memory: Nyland, Harris & Prins, *Fast N-Body Simulation with CUDA*,
  GPU Gems 3 ch. 31; WebGPU version: <https://webgpu.dudoxx.com/en/physics/nbody>
- vgpu `draw()` instancing and blend presets: `node_modules/vgpu/dist/draw.d.ts`
