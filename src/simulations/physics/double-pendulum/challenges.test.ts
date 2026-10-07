import { describe, expect, it } from "vitest";
import { boundaryValue, PENDULUM_CHALLENGES, readoutsFor } from "./challenges";
import type { PendulumReadouts } from "./challenges";
import { SpreadObserver, SwingObserver } from "./measure";
import { canNeverFlip, normalModes, step } from "./pendulum";
import type { State } from "./pendulum";
import { physicsOf, PENDULUM_DEFAULTS, startOf } from "./settings";
import type { PendulumSettings } from "./settings";
import { DT } from "./sim";

const challenge = (id: string) => PENDULUM_CHALLENGES.find((c) => c.id === id)!;
const settingsFor = (id: string, patch: Partial<PendulumSettings> = {}): PendulumSettings => ({ ...PENDULUM_DEFAULTS, ...challenge(id).setup, ...patch });

/** What the simulation would report after `seconds` of these settings, measured exactly as sim.ts does (64-bit CPU). */
function readoutsAfter(s: PendulumSettings, seconds: number): PendulumReadouts {
  const p = physicsOf(s);
  const [t1, t2] = startOf(s);
  let a: State = [t1, t2, 0, 0];
  let b: State = [t1 + 10 ** s.nudge, t2, 0, 0];
  const swing = new SwingObserver(a, p);
  const spread = new SpreadObserver();
  const steps = Math.round(seconds / DT);
  for (let k = 1; k <= steps; k++) {
    a = step(a, DT, p, s.integrator);
    if (s.view === "butterfly") b = step(b, DT, p, s.integrator);
    swing.sample(k * DT, a);
    if (s.view === "butterfly" && k % 8 === 0) spread.sample(k * DT, a, b);
  }
  if (s.view === "pendulum") return readoutsFor({ view: "pendulum", swing: swing.stats() });
  return readoutsFor({ view: "butterfly", swing: swing.stats(), crowd: { ...spread.stats(), n: s.count } });
}

const passes = (id: string, s: PendulumSettings, r: PendulumReadouts) => challenge(id).goal.check({ params: s, readouts: r });

describe("double pendulum challenges", () => {
  it("each has exactly one correct prediction", () => {
    for (const c of PENDULUM_CHALLENGES) expect(c.prediction?.options.filter((o) => o.correct)).toHaveLength(1);
  });

  it("swing in step: the setup (10°, 0°) does not pass; 10°, 14° (≈ √2) does; the periods quoted are right", () => {
    const modes = normalModes(physicsOf(PENDULUM_DEFAULTS));
    expect(modes.slow.period).toBeCloseTo(2.62, 2);
    expect(modes.fast.period).toBeCloseTo(1.09, 2);
    const setup = settingsFor("normal-mode");
    expect(passes("normal-mode", setup, readoutsAfter(setup, 21))).toBe(false);
    const inStep = settingsFor("normal-mode", { a2: 14 });
    expect(passes("normal-mode", inStep, readoutsAfter(inStep, 21))).toBe(true);
    expect(passes("normal-mode", settingsFor("normal-mode", { a2: -14 }), readoutsAfter(settingsFor("normal-mode", { a2: -14 }), 21))).toBe(false);
  });

  it("the butterfly: from 120°, 10⁻⁹ rad becomes 1 rad in about 15 s, with λ ≈ 1.5 per second", () => {
    const s = settingsFor("butterfly");
    const r = readoutsAfter(s, 20);
    expect(r.visibleAt).toBeGreaterThan(12);
    expect(r.visibleAt).toBeLessThan(18);
    expect(r.lambda).toBeGreaterThan(1.2);
    expect(r.lambda).toBeLessThan(1.8);
    expect(Math.log(1e9)).toBeCloseTo(21, 0);
    expect(passes("butterfly", s, r)).toBe(true);
    expect(passes("butterfly", { ...s, nudge: -3 }, r)).toBe(false); // a big nudge is no test
  });

  it("calm or chaos: 120° keeps growing; both arms at 60° stay within 10⁻⁸ rad for 30 s", () => {
    const chaotic = settingsFor("calm");
    expect(passes("calm", chaotic, readoutsAfter(chaotic, 30))).toBe(false);
    const calm = settingsFor("calm", { a1: 60, a2: 60 });
    const r = readoutsAfter(calm, 30);
    expect(r.maxSpread).toBeLessThan(2e-8);
    expect(r.lambda).toBeUndefined();
    expect(passes("calm", calm, r)).toBe(true);
    expect(passes("calm", { ...calm, a1: 20, a2: 20 }, readoutsAfter({ ...calm, a1: 20, a2: 20 }, 30))).toBe(false); // too easy
  });

  it("can it flip: 60°/0° never can (energy); 30°/140° is just outside the curve and flips after about 4.6 s", () => {
    expect(boundaryValue(60, 0)).toBeCloseTo(2, 12);
    expect(canNeverFlip([...startOf({ a1: 60, a2: 0 }), 0, 0] as State, physicsOf(PENDULUM_DEFAULTS))).toBe(true);
    const s = settingsFor("flip", { view: "pendulum", a1: 30, a2: 140 });
    expect(boundaryValue(30, 140)).toBeGreaterThan(0.5);
    expect(boundaryValue(30, 140)).toBeLessThan(1);
    const r = readoutsAfter(s, 8);
    expect(r.firstFlip).toBeCloseTo(4.6, 0);
    expect(passes("flip", s, r)).toBe(true);
    const far = settingsFor("flip", { view: "pendulum", a1: 170, a2: 170 }); // flips, but nowhere near the curve
    expect(passes("flip", far, readoutsAfter(far, 8))).toBe(false);
  });
});
