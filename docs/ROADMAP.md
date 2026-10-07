# EduSim improvement roadmap

Status: the foundation is stable (shared GPU runtime, three working simulations, registry routing,
unit + headless-GPU tests, shader validation, browser smoke test) and **all three simulations now meet the same
bar** — shareable settings, a measurement the tests prove correct, and three guided challenges (phases C, D and E
below). The fourth simulation, N-body orbital mechanics, shipped to the same bar. What remains is touch/phone
polish (phase B) and further simulations (phase F).
EduSim runs **locally** for now, so nothing below depends on hosting.

## Guiding principles

1. **Depth before breadth.** Three excellent simulations teach more than ten shallow ones. The catalog
   lists 21; finish the learning experience on the existing three before adding many more.
2. **Implicit scaffolding (PhET research).** Guide through what the sim affords — sensible defaults,
   measurement tools, linked representations, optional challenges — not walls of instructions.
   Students keep agency; multiple paths lead to the insight.
3. **Measure, don't just look.** Every sim should let students take numbers out of it (probes, plots,
   counters), so they can test predictions instead of just watching.
4. **Every claim is tested.** Keep the pattern established in the foundation: pure TypeScript "twin"
   of each shader's maths (like `hh.ts`), unit tests on the twin, headless-GPU test that the shader
   matches the twin.
5. **Local-first.** Works offline, state in the URL and `localStorage`, no accounts or servers.

WebGPU is now Baseline (Chrome/Edge, Firefox 141+ desktop, Safari 26 incl. iOS/iPadOS, Chrome on
Android 12+), so no WebGL fallback is planned; the existing "unsupported browser" message covers the rest.

---

## Phase A — Local workflow (½ day) · ✅ done

GitHub Actions is currently not starting jobs on this account, so verification must not depend on it.

- **Pre-push hook** running `pnpm verify` (lint, typecheck, tests, shader validation, build), via a
  small `scripts/install-hooks.mjs` + `prepare` script, so nothing broken reaches GitHub.
- **One-command smoke test:** `pnpm smoke:local` builds, starts the server, runs `scripts/smoke.mjs`,
  and stops the server.
- **Offline / classroom build:** every route is already static, so enable `output: "export"` behind
  `pnpm export` → an `out/` folder runnable with any static server (USB stick, school LAN, no internet).
- **Cloud-session setup hook** (Claude Code `SessionStart`): install deps and the vgpu software
  renderer so future sessions can run GPU tests immediately.

Done: `git push` runs `pnpm verify` (`.githooks/pre-push`, installed by `pnpm install`); `pnpm export` writes a
static site to `out/` that passes the browser smoke test from a plain file server; `pnpm smoke:local` is the
one-command browser check; `.claude/hooks/session-start.sh` prepares cloud sessions.

## Phase B — Interaction polish (1–2 days) · ✅ done

- ✅ **Cellular automata opens readable:** a library pattern is framed on load (at least 24 cells across, so a
  block opens close up and the Gosper gun fills the view); random soups and empty grids show the whole grid;
  "Fit pattern" (reads the grid back and frames the live cells) and "Fit grid" buttons.
- ✅ **Keyboard shortcuts** for every sim (in `SimLayout`): Space play/pause, `.` step, `R` reset, `?` help.
- ✅ **Touch:** pinch-zoom and two-finger pan for Cellular Automata and N-body (`lib/gestures.ts`; a touch only
  starts drawing once it moves, so a second finger can still turn it into a pinch); the sidebar is a bottom drawer
  on phones (peek / half / full, tap or drag the handle, Play always visible); compact two-row header; larger
  header targets on touch screens.
- ✅ **Adaptive quality + frame pacing:** bounded frame queue (Pause stays responsive on slow GPUs) and a
  governor that lowers render resolution (down to 35%) when GPU frames cost too much, restoring it only when the
  predicted cost fits; badge + `localStorage edusim:quality=full` to pin full resolution.
- ✅ **Accessibility basics:** a global `:focus-visible` outline (and a keyboard-tab test on every sim); a
  one-sentence summary of each canvas (e.g. "The neuron fires steadily at 68 spikes per second", "Generation 120:
  412 live cells. Still life.") in a polite live region, announced at most every 5 s;
  Okabe–Ito colours for the Hodgkin–Huxley traces with the voltage drawn thicker (the CA themes are single-hue on
  black, so they never depended on hue); simulations start paused under `prefers-reduced-motion`; every
  simulation page passes axe on desktop and phone; finger-sized slider thumbs on touch screens.

## Phase C — Shareable & persistent state (1 day) · ✅ done for all three simulations

- **State in the URL** for every sim (`?mode=double-slit&separation=0.8…`), parsed and clamped by a
  small typed schema per sim, so a configuration is a link — also the mechanism challenges use to set
  up an experiment.
- **Copy link** and **Reset to defaults** in `SimLayout`.
- **Remember last settings** per sim in `localStorage` (URL wins when present).
- **Save image** (PNG of the current canvas) for worksheets and reports.

Cellular automata links reproduce the rules, grid size, speed, colours and the **starting pattern** (a library
pattern, a fresh random soup or an empty grid) — not cells drawn by hand, which would not fit in a link.

## Phase D — Measurement tools & linked representations (3–5 days) · the core pedagogy · ✅ done for all three simulations

Shared building blocks: a small `MiniChart` component (SVG, theme-aware) and a "probe" pattern —
CPU twin for analytic sims, async GPU read-back for state-based ones.

**Wave interference**
- Movable **detector screen** on the right with a live **intensity-vs-position graph** — the classic
  double-slit plot.
- **Ruler** to measure fringe spacing; show λ, d, L; compare measured spacing with λL/d.
- Click-to-place **probe** reading amplitude/intensity at a point.
- Requires a TS twin of `phasor()` (`wave.ts`) + a GPU-vs-CPU test, as for HH.

**Hodgkin–Huxley**
- Hover **cursor readout** (time, V, m, h, n) and a dashed threshold line.
- **Firing-rate meter** and **spike counter**.
- **f–I curve builder:** sweep injected current on the CPU twin (`hh.ts`, fast) and plot firing rate
  vs current — shows threshold and rheobase.
- Optional **phase-plane view** (V vs n) as a second representation.

**Cellular automata** · ✅
- **Population graph** over the last 240 generations and a live-cell count. A reduction shader (`count.wgsl`)
  counts the live cells and computes an order-independent fingerprint of the whole grid for every generation,
  into a ring buffer; the page reads it back at most ten times a second, so the cost does not grow with speed.
- **Pattern detection** (`tracker.ts`): *extinct*, *still life*, *oscillator with its period*, *moving pattern*
  (a glider or spaceship: the population repeats but the fingerprint never does) or *evolving*. The shader is
  tested against a CPU twin (`life.ts`) cell for cell, and the classifier against block, blinker, toad,
  pulsar, glider and the Gosper gun.

Done when: each sim has at least one quantitative readout that a test asserts is correct.

## Phase E — Guided experiments (3–4 days) · ✅ framework + 3 challenges for each simulation

A data-driven **Challenges** tab in the sidebar (Explore | Challenges), optional by design.

- A challenge = `{ id, prompt, setup (URL-state params), prediction choices?, goal check on sim
  readouts?, hint, explanation }`, stored as TypeScript data next to each sim.
- **Predict → Observe → Explain** flow: commit to a prediction, run it, then see the explanation.
- Goal checks use the Phase D readouts (e.g. "make the fringes twice as far apart", "find the smallest
  current that makes the neuron fire repeatedly", "build a pattern that survives 100 generations").
- Progress saved locally; 3–5 challenges per sim to start.

Cellular automata: *build something that never changes* (a still life), *make it blink* (an oscillator) and
*grow without limit* (the Gosper glider gun's population passes 100). Each goal reads the GPU-measured status, so
it cannot be met by a shortcut (changing the rules, or a dense random soup), and every claim in the explanations is
asserted against the CPU twin in `challenges.test.ts`.

## Phase F — New simulations (1–3 days each)

Recommended order, each chosen to reuse what exists and to showcase the GPU:

1. **Double pendulum chaos** (easy): one pendulum plus a GPU ensemble of thousands of slightly
   different starts, showing divergence; a "flip-time fractal" map. Next up — plan in
   [`DOUBLE-PENDULUM-PLAN.md`](DOUBLE-PENDULUM-PLAN.md).
2. **Resting membrane potential** (easy): reuses the biology UI and HH concepts (Nernst, Goldman).
3. ✅ **N-body orbital mechanics** (medium): orbit lab + colliding galaxies, measured Kepler's laws, 4 challenges — see [`NBODY-PLAN.md`](NBODY-PLAN.md).
4. **Axon propagation** (medium): many HH compartments in parallel — the HH shader generalised from
   1 thread to N, which is where the GPU truly pays off for biology.

Each one ships with the full foundation checklist: shader validation, TS twin + tests, smoke test,
explanation text, at least one measurement tool and a few challenges.

## Phase G — Performance & scale (as needed)

- **Bit-packed CA** (32 cells per `u32`): 32× less memory, 8192² grids.
- GPU **timing overlay** in development (`timer(gpu)` with `timestamp-query`).
- Lower-resolution compute for the wave intensity view at high DPR.

## Phase H — Content & reach (later)

- Teacher notes per sim (learning goals, misconceptions, suggested activities).
- Internationalisation of UI and explanations.
- Screen-reader-friendly descriptions and full keyboard control of canvas tools.

---

## Recommended sequence

1. **Phase A** (small, protects everything after it).
2. **Vertical slice on Wave Interference:** Phase B polish + Phase C URL state + Phase D detector
   screen/ruler + Phase E with 3 challenges. This proves the whole learning loop on one sim.
3. ✅ Roll the same pattern out to Hodgkin–Huxley and Cellular Automata.
4. Then phone/touch and accessibility polish (the rest of Phase B), then new simulations (Phase F), starting
   with the double pendulum.

## Risks

| Risk | Mitigation |
|---|---|
| GitHub Actions blocked on the account | Local pre-push `pnpm verify` (Phase A) |
| `vgpu` is 0.x and may change its API | Version pinned in the lockfile; headless-GPU tests catch breakage on upgrade |
| Low-end phones struggle with per-pixel shaders | Adaptive DPR (Phase B) |
| Challenges feel like homework | Optional tab, short, predict-observe-explain rather than quizzes |
