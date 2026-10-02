import type { Challenge } from "@/lib/challenges";
import type { DetectorReadouts, WaveParams } from "./wave";

/** Guided experiments for the wave simulation. Goals are checked live against params and readouts. */
export const WAVE_CHALLENGES: Challenge<WaveParams, DetectorReadouts>[] = [
  {
    id: "separation",
    title: "Squeeze the fringes",
    prompt: "Two slits make a pattern of bright fringes on the detector. What controls how far apart they are?",
    setup: { mode: "double-slit", view: "intensity", frequency: 6, separation: 0.5, slitWidth: 0.1, detectorX: 1.5, detector: true },
    prediction: {
      question: "If you double the slit separation, the fringes will be…",
      options: [{ label: "further apart" }, { label: "closer together (about half the spacing)", correct: true }, { label: "unchanged" }],
    },
    goal: {
      description: "Double the slit separation (0.5 → 1.0) and watch the measured fringe spacing.",
      check: ({ params }) => params.mode === "double-slit" && params.separation >= 0.9,
    },
    hint: "Drag the Slit Separation slider to the right. Compare the “measured” fringe spacing in the corner before and after.",
    explanation:
      "Fringe spacing is Δy = λL / d. The paths from the two slits differ by d·sinθ, so a larger separation d reaches a whole-wavelength path difference at a smaller angle — the fringes crowd together. Doubling d halves the spacing, and the measured value tracks the formula.",
  },
  {
    id: "wavelength",
    title: "Stretch the wavelength",
    prompt: "Keep the slits fixed. Can you make the fringes at least twice as far apart using only the wave itself?",
    setup: { mode: "double-slit", view: "intensity", frequency: 6, separation: 0.8, slitWidth: 0.1, detectorX: 1.5, detector: true },
    prediction: {
      question: "A longer wavelength makes the fringes…",
      options: [{ label: "wider apart", correct: true }, { label: "narrower" }, { label: "no different" }],
    },
    goal: {
      description: "Reach a fringe spacing of at least 1.2 units while keeping the slit separation at 0.8.",
      check: ({ params, readouts }) => params.mode === "double-slit" && params.separation === 0.8 && (readouts.theorySpacing ?? 0) >= 1.2,
    },
    hint: "Wavelength λ = wave speed ÷ frequency. Lower the frequency (or raise the wave speed).",
    explanation:
      "In Δy = λL / d the spacing grows in proportion to the wavelength. Waves with longer wavelength bend more strongly around the slits, so their bright fringes land further from the centre. Moving the screen further away (larger L) helps too, but only the wavelength gets you all the way here.",
  },
  {
    id: "diffraction",
    title: "Squeeze a wave through a gap",
    prompt: "A single slit doesn't just let a beam through — watch what happens to it as the gap shrinks.",
    setup: { mode: "single-slit", view: "intensity", frequency: 6, slitWidth: 0.5, detector: false },
    prediction: {
      question: "When the slit gets narrower, the wave on the far side will…",
      options: [{ label: "spread out more (diffraction)", correct: true }, { label: "form a narrower beam" }, { label: "stay the same" }],
    },
    goal: {
      description: "Narrow the slit to 0.1 or less and look at how the wave fans out.",
      check: ({ params }) => params.mode === "single-slit" && params.slitWidth <= 0.1,
    },
    hint: "Use the Slit Width slider. A slit about as narrow as the wavelength (λ ≈ 0.17 here) fans the wave out in every direction.",
    explanation:
      "Diffraction: the angular spread of a wave through a slit of width a is about λ / a. A wide slit passes a fairly straight beam; as a approaches λ the wave spreads out in a half-circle — the same Huygens–Fresnel picture that explains the double-slit fringes.",
  },
];
