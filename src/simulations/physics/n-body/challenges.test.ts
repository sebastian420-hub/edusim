import { describe, expect, it } from "vitest";
import { NBODY_CHALLENGES, PLANET, readoutsFor } from "./challenges";
import { bodiesFromState, circularSpeed, computeAccelerations, conserved, escapeSpeed, G_ORBIT, orbitalElements, OrbitTracker, primaryIndex, stateFromBodies, step, toCentreOfMassFrame } from "./nbody";
import type { Body } from "./nbody";
import { encodeBodies, NBODY_DEFAULTS, setupBodies } from "./settings";
import type { NBodySettings } from "./settings";
import type { NBodyStats } from "./sim";

const DT = 1e-3;
const EPS2 = 1e-6;

/** What the simulation would report after `years` of the given settings — measured exactly like sim.ts does. */
function statsAfter(settings: NBodySettings, years: number): NBodyStats {
  const s = stateFromBodies(toCentreOfMassFrame(setupBodies(settings)));
  computeAccelerations(s, G_ORBIT, EPS2);
  const c0 = conserved(s);
  const tracker = new OrbitTracker();
  const angle = () => {
    const b = bodiesFromState(s);
    const p = b[primaryIndex(b)];
    return Math.atan2(b[PLANET].y - p.y, b[PLANET].x - p.x);
  };
  tracker.sample(0, angle());
  const steps = Math.round(years / DT);
  for (let k = 1; k <= steps; k++) {
    step(s, DT, G_ORBIT, EPS2, settings.integrator);
    if (k % 10 === 0) tracker.sample(k * DT, angle()); // the browser samples about every frame
  }
  const bodies = bodiesFromState(s);
  const primary = primaryIndex(bodies);
  const el = orbitalElements(bodies[PLANET], bodies[primary]);
  const c = conserved(s);
  return {
    mode: "orbit",
    time: steps * DT,
    n: s.n,
    kinetic: c.kinetic,
    potential: c.potential,
    total: c.total,
    drift: (c.total - c0.total) / Math.abs(c0.total),
    angularDrift: 0,
    history: { t: [], drift: [] },
    primary,
    orbits: [{ index: PLANET, r: el.r, v: el.v, a: el.a, e: el.e, energy: el.energy, period: tracker.period, orbits: tracker.orbits }],
    kepler: [],
  };
}

const byId = (id: string) => NBODY_CHALLENGES.find((c) => c.id === id)!;
const SUN: Body = { m: 1, x: 0, y: 0, vx: 0, vy: 0 };
const EARTH_MASS = 3.003e-6;
const settingsWith = (id: string, patch: Partial<NBodySettings> = {}): NBodySettings => ({ ...NBODY_DEFAULTS, ...byId(id).setup, ...patch });
const planet = (r: number, v: number): Partial<NBodySettings> => ({ bodies: encodeBodies([SUN, { m: EARTH_MASS, x: r, y: 0, vx: 0, vy: v }]) });
const solved = (id: string, settings: NBodySettings, years: number) => byId(id).goal.check({ params: settings, readouts: readoutsFor(settings, statsAfter(settings, years)) });

describe("n-body challenges", () => {
  it("have unique ids and exactly one correct prediction each", () => {
    expect(new Set(NBODY_CHALLENGES.map((c) => c.id)).size).toBe(NBODY_CHALLENGES.length);
    for (const c of NBODY_CHALLENGES) expect(c.prediction?.options.filter((o) => o.correct), c.id).toHaveLength(1);
  });

  it("are not already solved by their own setup, even after running for a while", () => {
    for (const c of NBODY_CHALLENGES) {
      const s = settingsWith(c.id);
      expect(c.goal.check({ params: s, readouts: readoutsFor(s, null) }), `${c.id} at the start`).toBe(false);
      expect(solved(c.id, s, 10), `${c.id} after 10 years`).toBe(false);
    }
  });

  it("circular orbit: solved at the circular speed √(GM/r) ≈ 4.44 AU/yr, not at the setup's 3.2", () => {
    const v = circularSpeed(2, 1, EARTH_MASS);
    expect(v).toBeCloseTo(4.443, 3);
    expect(solved("circular", settingsWith("circular", planet(2, v)), 3)).toBe(true);
    expect(solved("circular", settingsWith("circular", planet(2, 1.02 * v)), 3.5)).toBe(true); // 2 % off: e ≈ 0.04, still "a circle"
    expect(solved("circular", settingsWith("circular", planet(2, 1.08 * v)), 4)).toBe(false); // e ≈ 0.17
    // Needs a complete measured orbit, not just a good start.
    expect(solved("circular", settingsWith("circular", planet(2, v)), 2)).toBe(false);
    // Moving the planet instead of setting its speed is not the exercise.
    expect(solved("circular", settingsWith("circular", planet(1, circularSpeed(1, 1, EARTH_MASS))), 3)).toBe(false);
  });

  it("Kepler: the measured period at 4 AU is 8 years, and the goal needs it", () => {
    const s = settingsWith("kepler", planet(4, circularSpeed(4, 1, EARTH_MASS)));
    const stats = statsAfter(s, 8.4);
    expect(stats.orbits[0].period!).toBeCloseTo(8, 1);
    expect(byId("kepler").goal.check({ params: s, readouts: readoutsFor(s, stats) })).toBe(true);
    expect(solved("kepler", s, 7)).toBe(false); // not a full orbit yet
    expect(solved("kepler", settingsWith("kepler", planet(2, circularSpeed(2, 1, EARTH_MASS))), 3.2)).toBe(false); // wrong distance
  });

  it("escape: √2 × circular ≈ 8.89 AU/yr at 1 AU; well above it, or below it, does not count", () => {
    const vEsc = escapeSpeed(1, 1, EARTH_MASS);
    expect(vEsc).toBeCloseTo(8.886, 2);
    expect(solved("escape", settingsWith("escape", planet(1, 1.01 * vEsc)), 1)).toBe(true);
    expect(solved("escape", settingsWith("escape", planet(1, 0.98 * vEsc)), 1)).toBe(false); // bound: comes back
    expect(solved("escape", settingsWith("escape", planet(1, 1.5 * vEsc)), 1)).toBe(false); // "just throw it hard"
  });

  it("numerics: Euler drifts past 3 % within a few years; leapfrog never does", () => {
    expect(solved("euler", settingsWith("euler", { integrator: "euler" }), 4)).toBe(true);
    expect(solved("euler", settingsWith("euler", { integrator: "leapfrog" }), 30)).toBe(false);
    const euler = statsAfter(settingsWith("euler", { integrator: "euler" }), 4);
    expect(euler.orbits[0].r).toBeGreaterThan(1.03); // the orbit spirals outwards, as the correct prediction says
  });

  it("goal texts quote numbers the physics actually gives", () => {
    expect(2 * Math.PI).toBeCloseTo(6.28, 2); // Earth's circular speed in AU/yr
    expect(circularSpeed(4, 1) / circularSpeed(1, 1)).toBeCloseTo(0.5, 10); // "half of Earth's"
    expect(escapeSpeed(1, 1) / (2 * Math.PI)).toBeCloseTo(Math.SQRT2, 10);
  });
});
