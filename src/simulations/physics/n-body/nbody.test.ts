import { describe, expect, it } from "vitest";
import {
  bodiesFromState,
  circularSpeed,
  collidingGalaxies,
  computeAccelerations,
  conserved,
  diskGalaxy,
  escapeSpeed,
  G_ORBIT,
  orbitalElements,
  OrbitTracker,
  plummerCluster,
  stateFromBodies,
  step,
} from "./nbody";
import type { Body, Integrator, NBodyState } from "./nbody";
import { ORBIT_PRESETS, orbitPreset } from "./presets";

const EPS2 = 1e-8;
const SUN: Body = { m: 1, x: 0, y: 0, vx: 0, vy: 0 };

function run(bodies: Body[], years: number, dt = 1e-3, integrator: Integrator = "leapfrog", onStep?: (s: NBodyState, t: number) => void) {
  const s = stateFromBodies(bodies);
  computeAccelerations(s, G_ORBIT, EPS2);
  const steps = Math.round(years / dt);
  for (let k = 1; k <= steps; k++) {
    step(s, dt, G_ORBIT, EPS2, integrator);
    onStep?.(s, k * dt);
  }
  return s;
}

const angleOf = (s: NBodyState, i: number, primary = 0) => Math.atan2(s.pos[i * 4 + 1] - s.pos[primary * 4 + 1], s.pos[i * 4] - s.pos[primary * 4]);

describe("gravity", () => {
  it("pulls with G·M/r² towards the other body, and the potential is −G·M/r", () => {
    const s = stateFromBodies([SUN, { m: 0, x: 2, y: 0, vx: 0, vy: 0 }]);
    computeAccelerations(s, G_ORBIT, 0);
    expect(s.acc[4]).toBeCloseTo(-G_ORBIT / 4, 10);
    expect(s.acc[5]).toBeCloseTo(0, 12);
    expect(s.acc[7]).toBeCloseTo(-G_ORBIT / 2, 10);
  });

  it("is softened: two bodies at the same place feel no force and a finite potential", () => {
    const s = stateFromBodies([SUN, { ...SUN }]);
    computeAccelerations(s, 1, 0.01);
    expect([...s.acc.slice(0, 3)]).toEqual([0, 0, 0]);
    expect(s.acc[3]).toBeCloseTo(-10, 10);
  });
});

describe("Kepler's laws on the twin", () => {
  it("Earth on a circular orbit at 1 AU takes 1 year and stays at 1 AU", () => {
    const earth: Body = { m: 3e-6, x: 1, y: 0, vx: 0, vy: circularSpeed(1, 1, 3e-6) };
    const tracker = new OrbitTracker();
    let minR = Infinity;
    let maxR = 0;
    run([SUN, earth], 3.2, 1e-3, "leapfrog", (s, t) => {
      tracker.sample(t, angleOf(s, 1));
      const r = Math.hypot(s.pos[4] - s.pos[0], s.pos[5] - s.pos[1]);
      minR = Math.min(minR, r);
      maxR = Math.max(maxR, r);
    });
    expect(tracker.orbits).toBe(3);
    expect(tracker.period!).toBeCloseTo(1, 3);
    expect(minR).toBeGreaterThan(0.9999);
    expect(maxR).toBeLessThan(1.0001);
  });

  it("third law: the period grows as a^1.5 (4 AU → 8 years), measured from the motion", () => {
    for (const a of [0.5, 2, 4]) {
      const tracker = new OrbitTracker();
      const planet: Body = { m: 1e-9, x: a, y: 0, vx: 0, vy: circularSpeed(a, 1) };
      run([SUN, planet], a ** 1.5 * 1.1, 2e-3, "leapfrog", (s, t) => tracker.sample(t, angleOf(s, 1)));
      expect(tracker.orbits, `a = ${a}`).toBe(1);
      expect(tracker.period! / a ** 1.5, `a = ${a}`).toBeCloseTo(1, 3);
    }
  });

  it("an ellipse has the period of its semi-major axis, whatever its shape", () => {
    // Launched at 1 AU at 1.2× circular speed: an ellipse with a = 1/(2 − 1.44) AU.
    const planet: Body = { m: 1e-9, x: 1, y: 0, vx: 0, vy: 1.2 * circularSpeed(1, 1) };
    const el = orbitalElements(planet, SUN);
    expect(el.a).toBeCloseTo(1 / (2 - 1.44), 7);
    expect(el.e).toBeCloseTo(0.44, 7);
    const tracker = new OrbitTracker();
    run([SUN, planet], el.period * 1.05, 5e-4, "leapfrog", (s, t) => tracker.sample(t, angleOf(s, 1)));
    expect(tracker.period! / el.period).toBeCloseTo(1, 3);
  });

  it("escape speed is √2 × circular speed: just below comes back, just above never does", () => {
    expect(escapeSpeed(1, 1) / circularSpeed(1, 1)).toBeCloseTo(Math.SQRT2, 12);
    const at = (factor: number): Body => ({ m: 1e-9, x: 1, y: 0, vx: 0, vy: factor * escapeSpeed(1, 1) });
    expect(orbitalElements(at(0.99), SUN).energy).toBeLessThan(0);
    expect(orbitalElements(at(1.01), SUN).energy).toBeGreaterThan(0);
    expect(orbitalElements(at(1.01), SUN).e).toBeGreaterThan(1);
    const slow = run([SUN, at(0.97)], 40, 2e-3);
    const fast = run([SUN, at(1.03)], 40, 2e-3);
    expect(Math.hypot(slow.pos[4], slow.pos[5])).toBeLessThan(80); // bound: max distance 1/(2/1 − 0.97²·2) ≈ 16.9 AU
    expect(Math.hypot(fast.pos[4], fast.pos[5])).toBeGreaterThan(80);
  });
});

describe("conservation and the integrator", () => {
  const earth: Body = { m: 3e-6, x: 1, y: 0, vx: 0, vy: circularSpeed(1, 1) };

  it("leapfrog keeps the energy error bounded over 200 orbits", () => {
    const s0 = stateFromBodies([SUN, earth]);
    computeAccelerations(s0, G_ORBIT, EPS2);
    const e0 = conserved(s0).total;
    const s = run([SUN, earth], 200, 1e-2);
    expect(Math.abs(conserved(s).total / e0 - 1)).toBeLessThan(1e-3);
  });

  it("explicit Euler pumps energy in: the orbit spirals outwards", () => {
    const s0 = stateFromBodies([SUN, earth]);
    computeAccelerations(s0, G_ORBIT, EPS2);
    const e0 = conserved(s0).total;
    const s = run([SUN, earth], 5, 1e-3, "euler");
    // Energy is negative; drifting towards zero means the planet is less and less bound.
    expect(conserved(s).total / e0).toBeLessThan(0.95);
    expect(Math.hypot(s.pos[4] - s.pos[0], s.pos[5] - s.pos[1])).toBeGreaterThan(1.05);
  });

  it("momentum and angular momentum are conserved for a messy three-body system", () => {
    const bodies: Body[] = [
      { m: 1, x: 0, y: 0, vx: 0.3, vy: 0 },
      { m: 0.5, x: 1, y: 0.2, vx: 0, vy: 4 },
      { m: 0.2, x: -1.5, y: 0.7, vx: 1, vy: -3 },
    ];
    const s0 = stateFromBodies(bodies);
    computeAccelerations(s0, G_ORBIT, 1e-4);
    const c0 = conserved(s0);
    const s = run(bodies, 2, 1e-4);
    const c = conserved(s);
    for (let k = 0; k < 2; k++) expect(c.momentum[k]).toBeCloseTo(c0.momentum[k], 9);
    expect(c.angularMomentum).toBeCloseTo(c0.angularMomentum, 8);
  });
});

describe("orbit-lab presets", () => {
  it("have one name per body and unique ids", () => {
    expect(new Set(ORBIT_PRESETS.map((p) => p.id)).size).toBe(ORBIT_PRESETS.length);
    for (const p of ORBIT_PRESETS) expect(p.names, p.id).toHaveLength(p.bodies.length);
  });

  it("inner planets: measured periods follow Kepler's third law (Mercury 88 days … Mars 687 days)", () => {
    const p = orbitPreset("inner-planets");
    const trackers = p.bodies.map(() => new OrbitTracker());
    run(p.bodies, 2, 5e-4, "leapfrog", (s, t) => trackers.forEach((tr, i) => i > 0 && tr.sample(t, angleOf(s, i))));
    const days = trackers.slice(1).map((t) => t.period! * 365.25);
    expect(days[0]).toBeCloseTo(88, -0.5); // within ±1.5 days
    expect(days[1]).toBeCloseTo(224.7, -0.5);
    expect(days[2]).toBeCloseTo(365.25, -0.5);
    expect(days[3]).toBeCloseTo(687, -0.5);
  });

  it("figure-eight: the three stars swap places and return after about 1.007 years", () => {
    const p = orbitPreset("figure-eight");
    const s = run(p.bodies, 1.00679, 1e-4);
    const back = bodiesFromState(s);
    p.bodies.forEach((b, i) => {
      expect(back[i].x, `star ${i} x`).toBeCloseTo(b.x, 2);
      expect(back[i].y, `star ${i} y`).toBeCloseTo(b.y, 2);
    });
  });

  it("binary star: both stars stay 1 AU apart", () => {
    const p = orbitPreset("binary-star");
    let d = 0;
    run(p.bodies.slice(0, 2), 1, 1e-3, "leapfrog", (s) => (d = Math.max(d, Math.abs(Math.hypot(s.pos[0] - s.pos[4], s.pos[1] - s.pos[5]) - 1))));
    expect(d).toBeLessThan(1e-4);
  });
});

describe("OrbitTracker", () => {
  it("needs a full turn before reporting a period, and interpolates between samples", () => {
    const t = new OrbitTracker();
    // Uniform circular motion with period 2, sampled coarsely and off-grid.
    for (let time = 0.013; time < 2.5; time += 0.07) t.sample(time, (Math.PI * time) % (2 * Math.PI));
    expect(t.orbits).toBe(1);
    expect(t.period).toBeCloseTo(2, 6);
  });

  it("works for clockwise orbits and resets", () => {
    const t = new OrbitTracker();
    for (let time = 0; time < 3.1; time += 0.05) t.sample(time, -Math.PI * time);
    expect(t.orbits).toBe(1);
    expect(t.period).toBeCloseTo(2, 6);
    t.reset();
    expect(t.orbits).toBe(0);
    expect(t.period).toBeUndefined();
  });
});

describe("star clusters and galaxies (G = 1)", () => {
  it("Plummer cluster is close to virial equilibrium (2K ≈ −W) and has the requested size", () => {
    const c = plummerCluster(2000, 7);
    const s: NBodyState = { n: c.n, pos: c.pos, vel: c.vel, acc: new Float64Array(c.n * 4) };
    computeAccelerations(s, 1, 0);
    const { kinetic, potential, mass } = conserved(s);
    expect(mass).toBeCloseTo(1, 10);
    expect((2 * kinetic) / -potential).toBeGreaterThan(0.85);
    expect((2 * kinetic) / -potential).toBeLessThan(1.15);
  });

  it("is deterministic for a seed", () => {
    expect(plummerCluster(50, 3).pos).toEqual(plummerCluster(50, 3).pos);
    expect(plummerCluster(50, 3).pos).not.toEqual(plummerCluster(50, 4).pos);
  });

  it("disk galaxy: stars start on circular orbits around the core (they stay put for a while)", () => {
    const c = diskGalaxy(400, 2);
    const s: NBodyState = { n: c.n, pos: c.pos, vel: c.vel, acc: new Float64Array(c.n * 4) };
    const r0 = Array.from({ length: c.n }, (_, i) => Math.hypot(c.pos[i * 4], c.pos[i * 4 + 1]));
    computeAccelerations(s, 1, 0.05 * 0.05);
    for (let k = 0; k < 200; k++) step(s, 0.005, 1, 0.05 * 0.05);
    const moved = r0.slice(1).map((r, i) => Math.abs(Math.hypot(s.pos[(i + 1) * 4], s.pos[(i + 1) * 4 + 1]) - r) / r);
    moved.sort((a, b) => a - b);
    expect(moved[Math.floor(moved.length / 2)]).toBeLessThan(0.05); // median star within 5 % of its radius
    expect(c.group[0]).toBe(2);
  });

  it("colliding galaxies: two cores approaching each other, no net momentum", () => {
    const c = collidingGalaxies(600, 1);
    const cores = [...c.group].flatMap((g, i) => (g === 2 ? [i] : []));
    expect(cores).toHaveLength(2);
    const s: NBodyState = { n: c.n, pos: c.pos, vel: c.vel, acc: new Float64Array(c.n * 4) };
    computeAccelerations(s, 1, 0.01);
    const { momentum } = conserved(s);
    expect(Math.hypot(...momentum)).toBeLessThan(1e-9);
    const [a, b] = cores;
    const dx = c.pos[b * 4] - c.pos[a * 4];
    const dvx = c.vel[b * 4] - c.vel[a * 4];
    expect(dx * dvx).toBeLessThan(0); // closing in
  });
});
