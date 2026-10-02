# EduSim improvement roadmap

Status: the foundation is stable (shared GPU runtime, three working simulations, registry routing,
unit + headless-GPU tests, shader validation, browser smoke test). This plan covers what comes next.
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

## Phase B — Interaction polish (1–2 days)

- **Cellular automata opens readable:** "fit to content" zoom on load/pattern change instead of a
  tiny pattern in a 256² grid; "Fit grid" and "Fit pattern" buttons.
- **Keyboard shortcuts** for every sim: Space play/pause, `.` step, `R` reset, `?` shows shortcuts.
- **Touch:** pinch-zoom and two-finger pan for CA; larger hit targets; sidebar becomes a bottom
  drawer on phones so the canvas keeps most of the screen.
- **Adaptive quality:** if frame time stays high, lower the render DPR (wave and HH renderers are
  per-pixel; CA at 2048² is the heaviest), restore when it recovers.
- **Accessibility basics:** visible focus everywhere, an `aria-live` text summary of what the canvas
  shows (e.g. "Firing at 68 Hz", "Generation 120, 412 live cells"), colour-blind-safe CA themes and
  HH trace colours, respect `prefers-reduced-motion` (start paused).

## Phase C — Shareable & persistent state (1 day)

- **State in the URL** for every sim (`?mode=double-slit&separation=0.8…`), parsed and clamped by a
  small typed schema per sim, so a configuration is a link — also the mechanism challenges use to set
  up an experiment.
- **Copy link** and **Reset to defaults** in `SimLayout`.
- **Remember last settings** per sim in `localStorage` (URL wins when present).
- **Save image** (PNG of the current canvas) for worksheets and reports.

## Phase D — Measurement tools & linked representations (3–5 days) · the core pedagogy

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

**Cellular automata**
- **Population graph** over generations (GPU reduction or periodic read-back every N generations).
- Live-cell count and simple **period detection** (still life / oscillator / chaotic).

Done when: each sim has at least one quantitative readout that a test asserts is correct.

## Phase E — Guided experiments (3–4 days)

A data-driven **Challenges** tab in the sidebar (Explore | Challenges), optional by design.

- A challenge = `{ id, prompt, setup (URL-state params), prediction choices?, goal check on sim
  readouts?, hint, explanation }`, stored as TypeScript data next to each sim.
- **Predict → Observe → Explain** flow: commit to a prediction, run it, then see the explanation.
- Goal checks use the Phase D readouts (e.g. "make the fringes twice as far apart", "find the smallest
  current that makes the neuron fire repeatedly", "build a pattern that survives 100 generations").
- Progress saved locally; 3–5 challenges per sim to start.

## Phase F — New simulations (1–3 days each)

Recommended order, each chosen to reuse what exists and to showcase the GPU:

1. **Double pendulum chaos** (easy): one pendulum plus a GPU ensemble of thousands of slightly
   different starts, showing divergence; a "flip-time fractal" map as a stretch goal.
2. **Resting membrane potential** (easy): reuses the biology UI and HH concepts (Nernst, Goldman).
3. **N-body orbital mechanics** (medium): compute shader, a natural next GPU showcase.
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
3. Roll the same pattern out to Hodgkin–Huxley and Cellular Automata.
4. Then new simulations (Phase F), starting with the double pendulum.

## Risks

| Risk | Mitigation |
|---|---|
| GitHub Actions blocked on the account | Local pre-push `pnpm verify` (Phase A) |
| `vgpu` is 0.x and may change its API | Version pinned in the lockfile; headless-GPU tests catch breakage on upgrade |
| Low-end phones struggle with per-pixel shaders | Adaptive DPR (Phase B) |
| Challenges feel like homework | Optional tab, short, predict-observe-explain rather than quizzes |
