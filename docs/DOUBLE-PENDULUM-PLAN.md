# Double Pendulum Chaos — research and implementation plan

> **Status: shipped** with all three views. Decisions taken: one scrolling row of plates on the home page (four at a
> time on desktop), 512² fractal by default with a 1024² option, all three views in v1. Two changes from the plan,
> both found while building it: (1) the Pendulum view and the measured Butterfly pair are integrated on the CPU in
> 64-bit, because a 32-bit float near 2 rad cannot store a 10⁻⁹ rad nudge at all (the GPU crowd starts one 32-bit step
> apart instead, and the page measures when it parts from the 64-bit pair); (2) the λ estimate is a least-squares slope
> of ln(gap) over the growing range instead of Benettin renormalisation, which keeps the plotted gap honest. The energy
> boundary on the map is drawn for any masses and lengths (a·cos θ₁ + b·cos θ₂ = c, `flipBoundary`).

The fifth simulation (catalog id `double-pendulum`, route `/physics/double-pendulum`, already listed as planned).
Same bar as the other four: a TypeScript twin of every shader, GPU-vs-twin tests, shareable settings, measurements a
test proves correct, guided challenges, e2e scenarios, a poster and a live home-page preview.

## 1. What students should get out of it

From the chaos-teaching literature (classroom double-pendulum sequences, Tracker-based teaching sequences, PhET-style
implicit scaffolding):

1. **Deterministic ≠ predictable.** The motion follows Newton's laws exactly, yet nearby starts end up completely
   different: *sensitive dependence on initial conditions* (the butterfly effect).
2. **Chaos depends on energy.** Small swings are regular (two clean "normal modes"); larger ones become chaotic.
   The classic classroom demo is two pendulums released "identically": at low angles they stay together, at high
   angles they part.
3. **How fast predictions fail** can be measured: the gap between two runs grows exponentially, e^{λt}; λ (the
   Lyapunov exponent) is the slope of log(gap) against time.
4. **Energy limits what is possible.** Some starts can never flip over the top, which is pure energy accounting. The
   rest form a fractal pattern of flip times, as intricate as the Mandelbrot set.
5. **Even computers are butterflies.** Rounding errors in the 7th digit grow just like a nudge: the GPU (32-bit) and
   the CPU (64-bit) agree for a while, then part. That is why no simulation can forecast chaos for long.

## 2. Product shape: one simulation, three views of the same physics

| View | What you see | What you do | What it measures |
|---|---|---|---|
| **Pendulum** (default) | One double pendulum with a glowing trail of the tip; a small θ₁–θ₂ phase plot | Drag either bob to set the angles; release; Play | Energy (and its drift), angles, a flip counter, normal-mode periods |
| **Butterfly** | 2 to 10 000 pendulums whose starts differ by 10⁻⁹ rad, drawn as faint overlapping arms | Pick the spread and how many; release | The spread between them on a log graph, and λ (the Lyapunov exponent) as its slope; the "time until visible" |
| **Fractal** | A map of starting angles (θ₁ across, θ₂ up), one pendulum per pixel, coloured by time to first flip, building up live | Zoom and pan into the map; click a pixel to open that start in the Pendulum view | Flip time under the cursor; the energy boundary where flips are impossible |

The linked views are the teaching device: the fractal shows *where* chaos lives, a click turns a pixel into a real
pendulum, and the Butterfly view shows *how fast* it destroys predictions.

## 3. Physics

- **Model:** two point masses on rigid, massless rods (masses m₁, m₂, lengths l₁, l₂, gravity g), angles from the
  vertical. The standard Lagrangian equations solved for θ̈₁ and θ̈₂ (explicit closed form, one 2×2 solve). Defaults
  m₁ = m₂ = 1, l₁ = l₂ = 1, g = 9.81, adjustable within sane ranges.
- **Integrator:** classical RK4 with a fixed step (default 1 ms, several sub-steps per frame). Leapfrog does not
  apply directly, because the kinetic energy depends on the angles, so the Hamiltonian isn't separable. RK4's energy error
  is tiny at this step and is *shown* (energy drift readout), not hidden. Optional "Euler" for the numerics lesson,
  as in N-body.
- **Flip:** an arm flips when its angle passes over the top (|θ| crosses π). The first-flip time of either arm
  colours the fractal.
- **Energy boundary (exact, testable):** for equal point masses and lengths released from rest, the energy is
  E = −mgl(2 cos θ₁ + cos θ₂). The cheapest flip is the lower arm passing over the top while the upper arm hangs
  straight down, at potential energy −mgl. So **neither arm can ever flip if 2 cos θ₁ + cos θ₂ > 1**. (The often-quoted
  3 cos θ₁ + cos θ₂ > 2 is for pendulums made of uniform rods.) This region is drawn as an outline on the fractal, and
  the tests prove that no start inside it ever flips.
- **Small-angle normal modes (exact, testable):** for equal masses and lengths, ω² = (g/l)(2 ∓ √2). In-phase mode
  θ₂ = √2 θ₁, anti-phase mode θ₂ = −√2 θ₁. A measured period must match 2π/ω.
- **Lyapunov estimate:** the separation d(t) between a reference and a nudged run in phase space, with the Benettin
  method of periodic renormalisation so it never saturates. λ is the slope of log d against time.

## 4. GPU design

Everything is the same kernel: **one invocation advances one pendulum** by k RK4 steps. Only the start conditions
and the bookkeeping differ.

- **State buffer:** `vec4f(θ₁, θ₂, ω₁, ω₂)` per pendulum, plus `flipTime` (f32, −1 = not yet) for the fractal.
- **`pendulum.wgsl` (compute):** RK4 sub-steps; records first-flip time; workgroup 64 (valid on every adapter).
  Parameters (masses, lengths, g, dt, steps) are uniforms.
- **Fractal:** grid W×H (default 512², up to 1024² = 1 M pendulums) initialised *on the GPU* from the view window
  (no upload). Each frame advances all of them; the map "develops" live as more pixels flip. Zooming re-seeds the
  grid for the new window. A fragment shader colours by log(flip time) with a perceptually uniform, colour-blind-safe
  palette (cividis or viridis). It reads the flip-time buffer, which works in fragment shaders everywhere: the
  vertex-stage storage limit we hit in N-body does not apply here.
- **Butterfly:** N pendulums (2 – 10 000) in one buffer; the arms are drawn as instanced thin quads from a vertex-buffer
  copy of the positions (the proven N-body renderer pattern); additive blending makes the spread look like smoke.
- **Pendulum view:** N = 1 on the same kernel, plus a ring-buffer trail of the tip (as N-body trails).
- **Read-backs:** small (≤ 10 000 × 16 B) for energies, spread and λ, at ≤ 10 Hz; the fractal only reads the pixel
  under the cursor.
- **Cost:** 1 M pendulums × 4 force evaluations × ~60 flops ≈ 0.25 GFLOP per step: trivial for any GPU, a few
  steps per frame even on integrated graphics. Tests use 128² on the software renderer.

## 5. Measurements and how each is proven

| Readout | Proven by |
|---|---|
| Total energy and drift | Twin: RK4 drift < 10⁻⁶ over 60 s at dt = 1 ms in a regular regime; the GPU's drift matches the twin's |
| Normal-mode periods | Twin and GPU: measured period within 0.5 % of 2π/ω for both modes at 2° amplitude |
| Flip times (fractal) | Twin vs GPU at regular (non-chaotic) pixels: same flip time ± one step. The energy boundary has no flips. Symmetry: the map is symmetric under (θ₁, θ₂) → (−θ₁, −θ₂) |
| Spread and λ | Twin: λ ≈ 0 (no growth) for small swings, λ > 0 and consistent between runs for large swings |
| "Computers are butterflies too" | Twin vs GPU: they agree for regular starts; for chaotic starts they agree at first and then part (the time is shown) |

## 6. Shareable settings and challenges

**URL state:** view, the two start angles (and velocities), masses, lengths, g, speed, integrator, butterfly count and
spread, and the fractal window and resolution. A link reproduces exactly what is on screen, and a fractal click is a link too.

**Challenges** (goals read the measured values, can't be shortcut, every claim tested on the twin):

1. **Swing in step.** Find the start where both arms swing together forever, the in-phase normal mode
   (θ₂ ≈ √2·θ₁ at small angles). Prediction: "the lower arm should start *more* / *less* / *as far* out" (more).
   Goal: measured motion stays periodic (period within 1 % of 2π/ω) for 20 s.
2. **The butterfly.** Two pendulums 10⁻⁹ rad apart. Predict when they visibly part: never / about 1 minute / within
   ~10 seconds (correct, for a high release). Goal: measure the time at which their gap exceeds 1 rad from a chaotic
   start, and see λ.
3. **Calm or chaos?** Lower the release angle until the butterfly spread stops growing (λ ≈ 0). Goal: a start
   whose spread stays below 10⁻⁶ rad for 30 s, plus the converse from a chaotic one.
4. **Can it flip?** Released from rest with the upper arm at 60° and the lower arm hanging straight down, can
   either arm ever flip over the top? Prediction: yes, eventually / only with luck / never (correct). Explain with
   energy: 2 cos 60° + cos 0° = 2 > 1, so the energy is too low, however long you wait. Goal: on the fractal, find a
   start on the boundary curve 2 cos θ₁ + cos θ₂ = 1 (for example θ₁ = 90°, θ₂ = 0°) and see that starts just inside it
   never flip, while some just outside do.

## 7. Testing

- `pendulum.ts` (twin): equations of motion, RK4, energy, flip detection, normal-mode frequencies, the energy
  criterion, the Lyapunov estimator, fractal seeding (pixel ↔ angles). Unit tests for every claim in §5 and every number
  in the challenge texts.
- GPU tests: the kernel vs the twin (short times / regular starts exact to f32 tolerance; grid edges and N not a
  multiple of 64; on-GPU fractal seeding = twin seeding); the flip-time map symmetric; instanced arm drawing.
- e2e: the three views render without GPU errors; dragging a bob sets the angles (and the link); the normal-mode
  period is measured in the browser; the butterfly spread graph grows for a chaotic start and not for a calm one;
  fractal zoom, a click opens the pendulum, and the boundary outline; URL state and hostile links; phone layout with
  pinch-zoom on the fractal; axe; reduced motion starts paused; challenge flows.

## 8. Phases and estimate (≈ 4–5 days)

| Phase | Work | Done when |
|---|---|---|
| 1. Physics (1 day) | `pendulum.ts` twin + tests; `pendulum.wgsl`; GPU-vs-twin tests | normal modes, energy, the boundary and twin = GPU proven |
| 2. Pendulum view (1 day) | renderer (arms, bobs, trail), drag to set angles, energy and phase plot, URL state | a dragged start reproduces from its link |
| 3. Butterfly view (½–1 day) | ensemble seeding, smoke rendering, spread graph and λ | spread grows only for chaotic starts, in the browser |
| 4. Fractal view (1 day) | GPU seeding, live map, palette, zoom and pan (and pinch), cursor readout, click to open, boundary outline | the map matches the twin at regular pixels; zoom reveals detail |
| 5. Challenges and polish (½–1 day) | 4 challenges, home plate, poster, preview, docs, e2e | `pnpm verify` and full e2e green on prod, dev and static |

## 9. Decisions needed

1. **Home page with five plates.** Four plates share one row today. Recommended: keep the row and let it scroll
   horizontally with snap on every screen size (as phones already do), showing 4 at a time on desktop.
   Alternative: 3 + 2 rows (pushes the page past its one-screen budget).
2. **Fractal default resolution.** Recommended 512² (fast everywhere), with a 1024² option ("high detail").
3. **Scope of v1.** Recommended: all three views. If time is short, Pendulum + Butterfly first (the core lesson),
   Fractal as a follow-up PR.

## Sources

- Fractal of flip times, its energy-forbidden centre and symmetry: J. F. Lindner et al. (Wooster),
  <https://physics.wooster.edu/for-teague/>, and Heyl, "The double pendulum fractal" (rod pendulums; the point-mass
  boundary above is derived the same way); a WebGL example with 122 500
  pendulums: <https://discourse.threejs.org/t/a-fractal-or-122500-double-pendulums/77584>
- Regular vs chaotic fraction by energy, Lyapunov exponents: "Regular and chaotic phase space fraction in the double
  pendulum", <https://arxiv.org/pdf/2312.13436>; "Chaos and Regularity in the Double Pendulum with Lagrangian
  Descriptors", <https://arxiv.org/pdf/2403.07000>
- Teaching chaos with the double pendulum (learning goals, side-by-side release demo): "Design and Development of a
  Teaching-Learning Sequence about Deterministic Chaos Using Tracker Software", Education Sciences 14 (2024) 842,
  <https://iris.unina.it/retrieve/5b966321-c9da-41d6-9032-da07a7618aa2/education-14-00842-v2.pdf>; Cornell CHESS
  double-pendulum lab manual, <https://www.chess.cornell.edu/sites/default/files/inline-files/ll-dp-manual.pdf>
