import { circularSpeed } from "./nbody";
import type { Body } from "./nbody";

/** Orbit-lab systems, in AU, years and solar masses (G = 4π²). Masses of planets are their real ones. */
export interface OrbitPreset {
  id: string;
  label: string;
  bodies: Body[];
  /** Names shown next to the bodies (same order). */
  names: string[];
  /** Half the width of the initial view, in AU. */
  extent: number;
}

const M_EARTH = 3.003e-6;
const planet = (m: number, a: number, M = 1): Body => ({ m, x: a, y: 0, vx: 0, vy: circularSpeed(a, M, m) });

/** Chenciner & Montgomery's figure-eight choreography (G = 1 units), rescaled to G = 4π²: velocities ×2π. */
const EIGHT_X = 0.97000436;
const EIGHT_Y = -0.24308753;
const EIGHT_VX = -0.93240737 * 2 * Math.PI;
const EIGHT_VY = -0.86473146 * 2 * Math.PI;

export const ORBIT_PRESETS: OrbitPreset[] = [
  {
    id: "sun-earth",
    label: "Sun and Earth",
    bodies: [{ m: 1, x: 0, y: 0, vx: 0, vy: 0 }, planet(M_EARTH, 1)],
    names: ["Sun", "Earth"],
    extent: 1.6,
  },
  {
    id: "inner-planets",
    label: "Inner planets",
    bodies: [{ m: 1, x: 0, y: 0, vx: 0, vy: 0 }, planet(1.66e-7, 0.387), planet(2.448e-6, 0.723), planet(M_EARTH, 1), planet(3.227e-7, 1.524)],
    names: ["Sun", "Mercury", "Venus", "Earth", "Mars"],
    extent: 1.9,
  },
  {
    id: "binary-star",
    label: "Binary star",
    // Two half-solar-mass stars 1 AU apart, each circling the centre of mass at 0.5 AU: v = √(G·m·r)/d = π.
    bodies: [
      { m: 0.5, x: -0.5, y: 0, vx: 0, vy: -Math.PI },
      { m: 0.5, x: 0.5, y: 0, vx: 0, vy: Math.PI },
      planet(M_EARTH, 3),
    ],
    names: ["Star A", "Star B", "Planet"],
    extent: 3.8,
  },
  {
    id: "figure-eight",
    label: "Figure-eight (3 stars)",
    bodies: [
      { m: 1, x: EIGHT_X, y: EIGHT_Y, vx: -EIGHT_VX / 2, vy: -EIGHT_VY / 2 },
      { m: 1, x: -EIGHT_X, y: -EIGHT_Y, vx: -EIGHT_VX / 2, vy: -EIGHT_VY / 2 },
      { m: 1, x: 0, y: 0, vx: EIGHT_VX, vy: EIGHT_VY },
    ],
    names: ["Star A", "Star B", "Star C"],
    extent: 1.5,
  },
  {
    id: "comet",
    label: "Sun, Jupiter and a comet",
    bodies: [
      { m: 1, x: 0, y: 0, vx: 0, vy: 0 },
      planet(9.546e-4, 5.2),
      // Aphelion 8 AU, perihelion 0.6 AU: v = √(μ(2/r − 1/a)) with a = 4.3 AU.
      { m: 1e-12, x: -8, y: 0, vx: 0, vy: -2 * Math.PI * Math.sqrt(2 / 8 - 1 / 4.3) },
    ],
    names: ["Sun", "Jupiter", "Comet"],
    extent: 9,
  },
];

export const DEFAULT_ORBIT_PRESET = "sun-earth";

export const orbitPreset = (id: string) => ORBIT_PRESETS.find((p) => p.id === id) ?? ORBIT_PRESETS[0];
