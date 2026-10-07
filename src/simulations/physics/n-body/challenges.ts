import type { Challenge } from "@/lib/challenges";
import { circularSpeed, escapeSpeed, primaryIndex } from "./nbody";
import type { Body } from "./nbody";
import { encodeBodies, setupBodies } from "./settings";
import type { NBodySettings } from "./settings";
import type { NBodyStats, OrbitReadout } from "./sim";

/** The planet every challenge is about: body 1 of its setup. */
export const PLANET = 1;

/** What challenge goals can observe. */
export interface NBodyReadouts {
  /** Simulated years since the setup was (re)started. */
  time: number;
  /** Relative change of the total energy since the start. */
  drift: number;
  /** The planet's measured orbit (null before the first measurement). */
  planet: OrbitReadout | null;
  /** Where the planet starts: distance from the star (AU) and speed (AU/yr), from the setup itself. */
  startDistance: number;
  startSpeed: number;
}

export function readoutsFor(settings: NBodySettings, stats: NBodyStats | null): NBodyReadouts {
  const bodies = setupBodies(settings);
  const p = bodies[primaryIndex(bodies)];
  const b = bodies[PLANET] ?? p;
  return {
    time: stats?.mode === "orbit" ? stats.time : 0,
    drift: stats?.drift ?? 0,
    planet: stats?.mode === "orbit" ? (stats.orbits.find((o) => o.index === PLANET) ?? null) : null,
    startDistance: Math.hypot(b.x - p.x, b.y - p.y),
    startSpeed: Math.hypot(b.vx - p.vx, b.vy - p.vy),
  };
}

const SUN: Body = { m: 1, x: 0, y: 0, vx: 0, vy: 0 };
const EARTH_MASS = 3.003e-6;
const planetAt = (r: number, v: number): Body => ({ m: EARTH_MASS, x: r, y: 0, vx: 0, vy: v });
const orbitSetup = (bodies: Body[]): Partial<NBodySettings> => ({ mode: "orbit", preset: "sun-earth", bodies: encodeBodies(bodies), integrator: "leapfrog", speed: 1, trails: true });

const isOrbitLab = (s: NBodySettings) => s.mode === "orbit";

/**
 * Guided experiments. Every number in the texts is checked against the CPU twin in challenges.test.ts.
 */
export const NBODY_CHALLENGES: Challenge<NBodySettings, NBodyReadouts>[] = [
  {
    id: "circular",
    title: "Make the orbit a circle",
    prompt: "A planet starts 2 AU from the Sun but too slowly, so its orbit is a stretched ellipse. Change only its speed until the orbit is a circle.",
    setup: orbitSetup([SUN, planetAt(2, 3.2)]),
    prediction: {
      question: "Earth needs 6.28 AU/yr for its circular orbit at 1 AU. A circular orbit at 2 AU needs…",
      options: [{ label: "more speed than Earth" }, { label: "the same speed as Earth" }, { label: "less speed than Earth", correct: true }],
    },
    goal: {
      description: "Complete a full orbit with eccentricity below 0.05, starting 2 AU from the Sun.",
      check: ({ params, readouts: r }) =>
        isOrbitLab(params) && params.integrator === "leapfrog" && Math.abs(r.startDistance - 2) < 0.05 && !!r.planet && r.planet.orbits >= 1 && r.planet.e < 0.05,
    },
    hint: "Select the planet and use the Orbital speed slider. Too slow and it dives in towards the Sun; too fast and it swings out.",
    explanation:
      "For a circle, gravity must supply exactly the force needed to keep turning: v²/r = G·M/r², so v = √(G·M/r). Twice as far away, gravity is four times weaker, and the circular speed drops by √2: from 6.28 to about 4.44 AU/yr. Outer planets really do move more slowly — Neptune crawls along at about 5 km/s, Mercury races at 48 km/s.",
  },
  {
    id: "kepler",
    title: "Kepler’s third law",
    prompt: "Earth takes 1 year to go round at 1 AU. Move the planet out to 4 AU, give it a roughly circular orbit, and measure its period.",
    setup: orbitSetup([SUN, planetAt(1, circularSpeed(1, 1, EARTH_MASS))]),
    prediction: {
      question: "Four times as far from the Sun, one orbit takes…",
      options: [{ label: "4 years" }, { label: "8 years", correct: true }, { label: "16 years" }],
    },
    goal: {
      description: "Measure a full orbit at 4 AU (semi-major axis 3.8–4.2 AU, eccentricity below 0.1).",
      check: ({ params, readouts: r }) =>
        isOrbitLab(params) && params.integrator === "leapfrog" && !!r.planet && r.planet.period !== undefined && r.planet.a > 3.8 && r.planet.a < 4.2 && r.planet.e < 0.1,
    },
    hint: "Set Distance from star to 4 AU, then the Orbital speed to about 3.1 AU/yr (half of Earth’s). Raise the simulation speed — it takes a while.",
    explanation:
      "Kepler found that the square of the period grows like the cube of the orbit’s size: T² ∝ a³. At 4 AU, a³ = 64, so T² = 64 and T = 8 years. The planet has farther to go (4× the distance) and moves more slowly (half the speed), so it takes 4 × 2 = 8 times as long. The Kepler plot shows every measured orbit on the line T² = a³.",
  },
  {
    id: "escape",
    title: "Escape the Sun",
    prompt: "Starting from Earth’s orbit at 1 AU, find the slowest launch speed at which the planet never comes back.",
    setup: orbitSetup([SUN, planetAt(1, circularSpeed(1, 1, EARTH_MASS))]),
    prediction: {
      question: "Earth orbits at 6.28 AU/yr. To escape from 1 AU it needs at least…",
      options: [{ label: "1.1 × that: about 6.9 AU/yr" }, { label: "√2 × that: about 8.9 AU/yr", correct: true }, { label: "2 × that: about 12.6 AU/yr" }],
    },
    goal: {
      description: "Launch the planet from 1 AU just fast enough to escape (at most 5 % above the escape speed) and watch it leave.",
      check: ({ params, readouts: r }) => {
        if (!isOrbitLab(params) || params.integrator !== "leapfrog" || !r.planet || r.time <= 0) return false;
        const vEsc = escapeSpeed(r.startDistance, 1, EARTH_MASS);
        return Math.abs(r.startDistance - 1) < 0.05 && r.startSpeed >= vEsc && r.startSpeed <= 1.05 * vEsc && r.planet.energy >= 0;
      },
    },
    hint: "Raise the Orbital speed step by step. The status shows ‘escaping’ once the planet’s energy is no longer negative.",
    explanation:
      "A planet escapes when its kinetic energy, ½v², is enough to pay for climbing out of the Sun’s gravity well, G·M/r. That gives v = √(2·G·M/r) — exactly √2 times the circular speed, about 8.89 AU/yr (42 km/s) at Earth’s distance. Just above it, the orbit is a parabola that never closes; just below, it is a very long ellipse that eventually returns.",
  },
  {
    id: "euler",
    title: "Numerics matter",
    prompt: "Simulations replace smooth motion with small steps. Switch the integrator from Leapfrog to Euler and watch Earth’s orbit and the energy graph.",
    setup: orbitSetup([SUN, planetAt(1, circularSpeed(1, 1, EARTH_MASS))]),
    prediction: {
      question: "With the simple Euler method, Earth’s orbit will…",
      options: [{ label: "stay exactly the same" }, { label: "slowly spiral inwards" }, { label: "slowly spiral outwards", correct: true }],
    },
    goal: {
      description: "Run with the Euler integrator until the total energy has drifted by more than 3 %.",
      check: ({ params, readouts: r }) => isOrbitLab(params) && params.integrator === "euler" && r.drift > 0.03,
    },
    hint: "Choose Euler under Integrator, press Play, and watch the energy graph and the trail.",
    explanation:
      "Euler moves each body along its current velocity for the whole step, so on a curved orbit it always overshoots outwards a little, and every step adds a little energy: the orbit spirals out, though nothing in the physics says it should. Leapfrog splits each step into half-kicks around a drift; it is symplectic, which keeps the energy error bounded for ever — that is why real astronomy codes use it.",
  },
];
