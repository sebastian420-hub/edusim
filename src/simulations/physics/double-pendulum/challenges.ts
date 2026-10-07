import type { Challenge } from "@/lib/challenges";
import { normalModes, rad } from "./pendulum";
import type { PendulumSettings } from "./settings";
import { isSymmetric } from "./settings";
import type { PendulumStats } from "./sim";

/** What challenge goals can observe. */
export interface PendulumReadouts {
  view: PendulumSettings["view"] | null;
  /** Simulated seconds since the (re)start. */
  time: number;
  /** Last measured periods of the upper arm (s). */
  periods: number[];
  flips: number;
  firstFlip?: number;
  /** Butterfly: largest gap so far, when it passed 1 rad, and its growth rate. */
  maxSpread?: number;
  visibleAt?: number;
  lambda?: number;
}

export function readoutsFor(stats: PendulumStats | null): PendulumReadouts {
  if (!stats) return { view: null, time: 0, periods: [], flips: 0 };
  if (stats.view === "fractal") return { view: "fractal", time: stats.fractal.time, periods: [], flips: 0 };
  const { swing } = stats;
  const base: PendulumReadouts = { view: stats.view, time: swing.time, periods: swing.periods, flips: swing.flips, firstFlip: swing.firstFlip };
  if (stats.view === "pendulum") return base;
  return { ...base, maxSpread: stats.crowd.maxSpread, visibleAt: stats.crowd.visibleAt, lambda: stats.crowd.lambda };
}

/** 2 cos θ₁ + cos θ₂ for a release in degrees: above 1, neither arm can ever flip (equal masses and lengths). */
export const boundaryValue = (a1: number, a2: number) => 2 * Math.cos(rad(a1)) + Math.cos(rad(a2));

const plain = { m2: 1, l2: 1, g: 9.81, integrator: "rk4", speed: 1, trail: true } as const;

/** Guided experiments. Every number in the texts is checked against the CPU twin in challenges.test.ts. */
export const PENDULUM_CHALLENGES: Challenge<PendulumSettings, PendulumReadouts>[] = [
  {
    id: "normal-mode",
    title: "Swing in step",
    prompt:
      "Released from a small angle, the double pendulum usually wobbles irregularly. But there is a special start where both arms swing together, back and forth for ever, like a single pendulum. Find it with the upper arm at 10°.",
    setup: { ...plain, view: "pendulum", a1: 10, a2: 0 },
    prediction: {
      question: "To swing in step with the upper arm, the lower arm should start…",
      options: [{ label: "less far out than the upper arm" }, { label: "exactly as far out as the upper arm" }, { label: "further out than the upper arm", correct: true }],
    },
    goal: {
      description: "Release with the upper arm at 20° or less so that every measured period over 20 s is within 1 % of the in-phase mode's 2π/ω.",
      check: ({ params, readouts: r }) => {
        if (params.view !== "pendulum" || r.view !== "pendulum" || !isSymmetric(params) || Math.abs(params.a1) > 20 || r.time < 20 || r.periods.length < 6) return false;
        const T = normalModes({ g: params.g, l1: 1 }).slow.period;
        return r.periods.every((p) => Math.abs(p / T - 1) < 0.01);
      },
    },
    hint: "Keep the upper arm at 10° and try the lower arm at 12°, 14°, 16°… (sliders or drag the lower bob). The periods show under the canvas; they must all be the same.",
    explanation:
      "Small swings of a double pendulum are a mix of two ‘normal modes’. In the slow one both arms swing together with the lower arm √2 ≈ 1.41 times further out (10° and 14°), at ω² = (g/l)(2 − √2): a period of 2.62 s. In the fast one they swing in opposite directions, θ₂ = −√2·θ₁, at (2 + √2): 1.09 s. Any other start mixes the two, which is why the motion looks irregular even when it is not chaotic.",
  },
  {
    id: "butterfly",
    title: "The butterfly effect",
    prompt:
      "A hundred pendulums are released from a high start, each one a billionth of a radian (10⁻⁹) further out than the last: far less than the width of an atom at the bob. The white pair is computed in 64-bit precision; its gap is plotted on a log scale.",
    setup: { ...plain, view: "butterfly", a1: 120, a2: 120, count: 100, nudge: -9 },
    prediction: {
      question: "The two white pendulums start 10⁻⁹ rad apart. They will look completely different…",
      options: [{ label: "never: the difference is far too small to matter" }, { label: "after a few minutes" }, { label: "within about 20 seconds", correct: true }],
    },
    goal: {
      description: "Watch the gap between the white pair grow past 1 rad, starting 10⁻⁹ rad apart (or less).",
      check: ({ params, readouts: r }) => params.view === "butterfly" && r.view === "butterfly" && params.nudge <= -9 && r.visibleAt !== undefined,
    },
    hint: "Press Play and watch the spread graph: it climbs in a straight line on the log scale, then levels off at ‘completely different’.",
    explanation:
      "From 120°, the gap grows exponentially, by a factor e about every 0.7 s (the measured growth rate λ, the Lyapunov exponent, is about 1.5 per second). A billionth needs only ln(10⁹) ≈ 21 e-foldings to become 1 rad: about 15 seconds. That is chaos: fully deterministic, yet any uncertainty in the start, however tiny, soon swamps the prediction. Weather forecasts fail after about two weeks for the same reason. The colourful crowd is computed on the GPU in 32-bit numbers, which cannot hold a difference of 10⁻⁹ near 2 rad: its pendulums start one 32-bit step (about 2·10⁻⁷ rad) apart, so it fans out a few seconds sooner. Even computers are butterflies.",
  },
  {
    id: "calm",
    title: "Calm or chaos?",
    prompt: "Chaos needs energy. Lower the release angles until the two white pendulums, 10⁻⁹ rad apart, stay together.",
    setup: { ...plain, view: "butterfly", a1: 120, a2: 120, count: 100, nudge: -9 },
    prediction: {
      question: "Released from 60° (both arms), the two pendulums will…",
      options: [{ label: "part within a minute, just more slowly" }, { label: "stay together: their gap hardly grows", correct: true }],
    },
    goal: {
      description: "Find a start with the upper arm at 40° or more where the gap stays below 10⁻⁶ rad for 30 s (nudge 10⁻⁹ or more).",
      check: ({ params, readouts: r }) =>
        params.view === "butterfly" && r.view === "butterfly" && params.nudge >= -9 && Math.abs(params.a1) >= 40 && r.time >= 30 && (r.maxSpread ?? Infinity) < 1e-6,
    },
    hint: "Try both arms at 60°. The spread graph should stay flat, and λ should not appear.",
    explanation:
      "At 60° the gap stays around 10⁻⁸ rad after 30 s instead of exploding: the motion is regular (quasi-periodic), and errors grow at most in proportion to time, not exponentially. Between the calm low swings and the wild high ones lies a mixed zone where the outcome depends on the exact start. Physicists map it with the fraction of starts that are chaotic: it rises from 0 to almost 1 as the energy approaches that of a flip.",
  },
  {
    id: "flip",
    title: "Can it flip?",
    prompt:
      "Released from rest with the upper arm at 60° and the lower arm hanging straight down, can either arm ever swing over the top? The Fractal view colours every start by how soon it flips; the white curve is where 2 cos θ₁ + cos θ₂ = 1.",
    setup: { ...plain, view: "fractal", a1: 60, a2: 0, boundary: true, fx: 0, fy: 0, fspan: 2 * Math.PI, res: 512 },
    prediction: {
      question: "Upper arm at 60°, lower arm straight down: if you wait long enough, an arm will flip…",
      options: [{ label: "yes, eventually: chaos tries everything" }, { label: "only with luck" }, { label: "never", correct: true }],
    },
    goal: {
      description: "Release a start just outside the curve (2 cos θ₁ + cos θ₂ between 0.5 and 1) that flips, after at least 1 s of swinging.",
      check: ({ params, readouts: r }) => {
        const f = boundaryValue(params.a1, params.a2);
        return params.view === "pendulum" && r.view === "pendulum" && isSymmetric(params) && f > 0.5 && f < 1 && r.firstFlip !== undefined && r.firstFlip >= 1;
      },
    },
    hint: "Hover the map close to the white curve to read flip times, then click a coloured pixel just outside it: it opens in the Pendulum view. Try θ₁ = 30°, θ₂ = 140°.",
    explanation:
      "Energy decides. Released from rest, the energy is −mgl(2 cos θ₁ + cos θ₂); the cheapest way to flip (the lower arm over the top while the upper arm hangs down) needs −mgl. At 60° and 0°, 2 cos 60° + cos 0° = 2 > 1: the energy is too low, however long you wait. Inside the curve the map stays dark for ever; just outside it, flips are allowed but not guaranteed, and the flip times form a fractal.",
  },
];
