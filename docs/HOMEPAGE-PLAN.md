# Main page redesign — plan

Goal: a main page that is **compact, well engineered and elegant** — a small piece of design in its own
right — and that actually shows what EduSim is: live, GPU-computed science.

Concept mock-ups (static, not shipped): [`design/home-concept-desktop.png`](design/home-concept-desktop.png),
[`design/home-concept-phone.png`](design/home-concept-phone.png). Today's page for comparison:
[`design/home-current-desktop.png`](design/home-current-desktop.png).

## 1. What is wrong today (measured)

| | Today | Why it matters |
|---|---|---|
| Page height, 1440×900 | **3,270 px = 3.6 screens** | A catalogue of 3 working things shouldn't need scrolling |
| Page height, phone 390×844 | **7,412 px = 8.8 screens** | Students are on phones and Chromebooks |
| Space above the first simulation | **650 px** — nothing real fits above the fold | The product is hidden behind a banner |
| Cards that are real | **3 of 22** | 19 greyed "coming soon" cards take ~86 % of the space at equal visual weight |
| Same fact shown | subject appears 3× (summary cards, section heading, card footer); two badges per card | Noise, not information |
| Emoji as icons | inconsistent per OS; four duplicates (🌊 ⚡ 🧠 🔬) | Looks like a template |
| Live content on a GPU product's home page | none | The page doesn't demonstrate the thing |

## 2. Concept: "plates and an index"

An editorial layout, like a book of figures with an index at the back:

- **Three plates** — one per working simulation — are the main object, each showing *the real output* of that
  simulation (wave fringes, voltage trace, Game of Life) in a tall poster frame, with a mono caption
  (`PLATE I · PHYSICS`), a title, one sentence and a difficulty meter.
- **A typographic index** — hairline-ruled lists, one column per subject — replaces 19 greyed cards with
  ~30 px rows. Planned work stays visible (it shows direction) without competing with what exists.
- **Colour comes from the art.** The interface is near-black ink, warm white text and one accent (the teal
  "live" dot). Each plate's accent matches its subject. No gradients on text, no card shadows, no emoji.
- **Strict structure.** 12-column grid, 8 px baseline, 1 px hairlines instead of boxes, left-aligned type.

### Layout (desktop 1440×900, ~1 screen total)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ◎ EduSim                                         3 live · 19 in preparation  │ 56
├──────────────────────────────────────────────────────────────────────────────┤
│ INTERACTIVE · GPU-COMPUTED     ┌─ PLATE I ─┐ ┌─ PLATE II ┐ ┌ PLATE III ┐     │
│ Science you can                │ ● live    │ │ ● live    │ │ ● live    │     │
│ reach into.                    │  (art)    │ │  (art)    │ │  (art)    │     │ ~440
│ Live models of waves, …        │───────────│ │───────────│ │───────────│     │
│ ─ Engine        WebGPU         │ Title     │ │ Title     │ │ Title     │     │
│ ─ Install       none           │ one line  │ │ one line  │ │ one line  │     │
│ ─ Offline       yes            │ ●○○  Open→│ │ ●●○  Open→│ │ ●○○  Open→│     │
├──────────────────────────────────────────────────────────────────────────────┤
│ IN PREPARATION                                          ●○○ easy ●●○ ●●●     │
│ PHYSICS 4        CHEMISTRY 5       COMPUTER SCIENCE 5     BIOLOGY 5          │ ~230
│ N-Body …  ●●○    Molecular …       Sorting …              Axon …             │
│ …                …                 …                      …                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ Runs on WebGPU in current Chrome, Edge, Firefox and Safari.   open source    │ 40
└──────────────────────────────────────────────────────────────────────────────┘
```

**Phone:** headline and one sentence, then the plates as a horizontal scroll-snap row (one and a bit visible,
which invites swiping), then the index as four collapsed accordions. Target ≈ 1.2 screens (today: 8.8).

## 3. Visual system

- **Type:** Geist (UI) and Geist Mono (labels, numerals) — already loaded, so no new font cost. Display
  headline 34–46 px, weight 500, tracking −0.035 em, `text-wrap: balance`. Mono 10–11 px, tracked, uppercase
  for metadata. A modular scale rather than ad-hoc sizes.
- **Colour tokens** (CSS variables, defined once in `globals.css`): `--bg #0a0c10`, `--ink #e9ebef`,
  `--mut #939bab`, `--dim #7d8596`, `--hair rgba(255,255,255,.09)`, `--live #5eead4`; subject accents physics
  `#6ea8fe`, biology `#4ade80`, computer science `#fbbf24`, chemistry `#c084fc`. All text ≥ 4.5 : 1.
- **Mark:** two interleaved sets of concentric rings (an interference pattern) as an inline SVG — also used
  for the favicon, replacing the default Next.js one.
- **Motion:** restrained and CSS-only by default — a slow pulse on the "live" dot, a 0.8 s zoom and border-colour
  change on plate hover, a short staggered fade-up on load. All disabled under `prefers-reduced-motion`.

## 4. The art — real output, not decoration

Two layers, so the page is fast everywhere and alive where it counts:

1. **Posters (default).** Captured from the real simulations by a script
   (`scripts/make-posters.mjs`) using the headless-browser setup we already have, from fixed shareable-link
   settings (so they are reproducible), saved as optimised WebP in `public/plates/`. Zero GPU, instant paint,
   works without WebGPU and in the offline export. Regenerate with one command when a simulation changes.
2. **Live on intent.** When a plate is hovered or focused (or tapped on touch), its poster is replaced by the
   real simulation running small in the same frame. Rules: **one live plate at a time**; paused when scrolled
   out of view or the tab is hidden; never on `prefers-reduced-motion` or without WebGPU (poster stays); the
   adaptive-quality governor applies; the home page does **not** touch the GPU until the visitor shows intent.

Alternative for the posters: hand-built SVG art (what the mock-up uses) — resolution independent and a few
KB, but it can drift from what the simulations really look like. Recommended: captured posters.

## 5. Engineering

- **Stays a static, server-rendered page.** Components: `Masthead`, `Hero`, `Plate`, `PlannedIndex`, `Footer`
  — all server components. Only the live-preview island is client code, loaded on intent via dynamic import.
- **Catalog remains the single source of truth** (`lib/subjects.ts`): add optional `tagline` and `poster`
  fields to implemented entries; the page derives everything (counts, columns, plates) from it. A unit test
  fails if an implemented simulation has no poster.
- **No JS for the responsive index:** desktop columns and mobile accordions are one `<ul>` styled twice with
  CSS (`details` always open on desktop via the grid layout, collapsed on small screens), so no duplicated DOM
  and no layout flash.
- **Semantics:** one `h1`, `h2` per plate, planned items are plain list items (not fake links, no
  "not-allowed" cursor), plates are single links with a clear accessible name, visible focus rings.
- **Images:** posters have explicit dimensions (no layout shift), the first plate is `priority`.
  The offline static export uses pre-optimised files (no runtime image service).
- **Social preview:** an Open Graph image built from the three posters.

## 6. Budgets and tests (so it stays good)

New browser-suite scenarios, run on prod, dev and the static export:

- Page height ≤ **1.15 screens** at 1440×900 and ≤ **1.4 screens** on a 390×844 phone (index collapsed).
- **No GPU use on load:** count `navigator.gpu.requestAdapter` calls = 0 before any interaction.
- Hover a plate → a live canvas appears; moving to another plate stops the first (one live at a time);
  `prefers-reduced-motion` and no-WebGPU keep the poster; a GPU error falls back to the poster.
- Links: exactly the implemented simulations are links; every link has an accessible name; heading order valid.
- Automated accessibility audit (axe-core) with zero violations; contrast checked.
- Horizontal overflow = 0 at 320, 390, 768, 1024, 1440 px.
- First-load JavaScript for `/` measured at the start of Phase 1 and then locked as a budget (must not grow).

## 7. Phases (≈ 4 days)

| Phase | Work | Result |
|---|---|---|
| 1 · Foundation (½ d) | Design tokens, type scale and grid in `globals.css`; logo mark and favicon; remove emoji from the home page; baseline JS size measured | A coherent visual system |
| 2 · Static layout (1 d) | Masthead, hero, plates with placeholder art, planned index, footer; responsive (scroll-snap plates, accordions); a11y pass; height budget tests | Compact page, no art yet |
| 3 · Posters (1 d) | Poster script from the real simulations, WebP output, `next/image` sizing, OG image, catalog fields, staleness-proof tests | Real art on the plates |
| 4 · Live on intent (1 d) | Preview island (dynamic import), single-live rule, offscreen/hidden pause, reduced-motion and no-WebGPU fallbacks, quality governor | Plates come alive on hover |
| 5 · Polish (½ d) | Load-in motion, focus states, copy edit, axe audit, final browser matrix, README/screenshots | Ready to merge |

## 8. Decisions for you

1. **Posters:** captured from the real simulations (recommended) or hand-made SVG art?
2. **Live preview on hover:** include it (Phase 4, +1 day, the "wow") or ship posters only first?
3. **Tone of the copy:** "Science you can reach into." is a placeholder. Keep, or a plainer "Interactive science
   simulations in your browser"?
4. **Planned simulations:** keep the index visible (recommended — shows direction), or hide it behind a single
   "Coming next" link?
5. **Light theme:** not planned (the simulations are all dark). Say if you want it.

## 9. Deliberately not doing

A full-screen animated hero (heavy on weak devices, competes with the content), parallax, carousels on
desktop, a new font, glassmorphism, shadows, or any new dependency beyond a dev-only accessibility checker.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Posters go stale when a simulation changes | One-command regeneration; documented in the README and in the "adding a simulation" steps |
| Live preview costs battery on phones | Only on explicit intent, one at a time, paused off-screen, adaptive quality, off for reduced motion |
| "Art piece" tempts over-decoration | The system is deliberately small: two typefaces, one accent, hairlines; the art is the content |
