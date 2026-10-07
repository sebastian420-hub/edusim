import { describe, expect, it } from "vitest";
import {
  canNeverFlip,
  crowdAngles,
  DEFAULT_PENDULUM,
  derivatives,
  energy,
  flipForbiddenEqual,
  flipBoundary,
  flipTime,
  FULL_WINDOW,
  growthRate,
  hasFlipped,
  mapAngles,
  normalModes,
  PeriodTracker,
  pixelAngles,
  positions,
  rad,
  separation,
  step,
} from "./pendulum";
import type { Integrator, State } from "./pendulum";

const P = DEFAULT_PENDULUM;

function run(start: State, seconds: number, dt = 1e-3, integrator: Integrator = "rk4", each?: (s: State, t: number) => void): State {
  let s = start;
  const n = Math.round(seconds / dt);
  for (let k = 1; k <= n; k++) {
    s = step(s, dt, P, integrator);
    each?.(s, k * dt);
  }
  return s;
}

describe("equations of motion", () => {
  it("hanging straight down is an equilibrium; tilted, gravity pulls it back", () => {
    expect(derivatives([0, 0, 0, 0], P).map((v) => v + 0)).toEqual([0, 0, 0, 0]); // + 0 folds −0 into 0
    const [, , a1] = derivatives([0.1, 0.1, 0, 0], P);
    expect(a1).toBeLessThan(0);
  });

  it("a single pendulum is recovered when the lower mass is tiny (θ̈₁ ≈ −g/l·sin θ₁)", () => {
    const [, , a1] = derivatives([0.3, 0.3, 0, 0], { ...P, m2: 1e-9 });
    expect(a1).toBeCloseTo(-P.g * Math.sin(0.3), 6);
  });

  it("bob positions: arms of length 1, y pointing up", () => {
    const { x1, y1, x2, y2 } = positions([Math.PI / 2, 0, 0, 0], P);
    expect([x1, y1, x2, y2].map((v) => Number(v.toFixed(12)) + 0)).toEqual([1, 0, 1, -1]);
  });
});

describe("energy", () => {
  it("RK4 conserves it: drift below 1e-6 over 60 s of regular motion at dt = 1 ms", () => {
    const start: State = [rad(30), rad(20), 0, 0];
    const e0 = energy(start, P).total;
    const end = run(start, 60);
    expect(Math.abs(energy(end, P).total / e0 - 1)).toBeLessThan(1e-6);
  });

  it("RK4 keeps it even through wild, chaotic motion (below 1e-5 over 30 s)", () => {
    const start: State = [rad(120), rad(150), 0, 0];
    const e0 = energy(start, P).total;
    const end = run(start, 30);
    expect(Math.abs((energy(end, P).total - e0) / e0)).toBeLessThan(1e-5);
  });

  it("explicit Euler pumps energy in, steadily (over 1 % in 20 s at dt = 1 ms, a million times RK4's error)", () => {
    const start: State = [rad(30), rad(20), 0, 0];
    const e0 = energy(start, P).total;
    let previous = e0;
    let rises = 0;
    run(start, 20, 1e-3, "euler", (s, t) => {
      if (Math.round(t * 1000) % 2000 !== 0) return;
      const e = energy(s, P).total;
      if (e > previous) rises++;
      previous = e;
    });
    expect(rises).toBe(10); // higher at every 2-second check
    expect(previous - e0).toBeGreaterThan(0.01 * Math.abs(e0));
  });
});

describe("small swings: two normal modes", () => {
  const modes = normalModes(P);

  it("ω² = (g/l)(2 ∓ √2)", () => {
    expect(modes.slow.omega ** 2).toBeCloseTo(P.g * (2 - Math.SQRT2), 10);
    expect(modes.fast.omega ** 2).toBeCloseTo(P.g * (2 + Math.SQRT2), 10);
    expect(modes.slow.period).toBeCloseTo(2.6206, 3);
  });

  for (const [name, ratio] of [
    ["in phase (θ₂ = √2·θ₁)", Math.SQRT2],
    ["anti-phase (θ₂ = −√2·θ₁)", -Math.SQRT2],
  ] as const) {
    it(`${name}: the measured period matches 2π/ω within 0.5 %, and the shape stays`, () => {
      const mode = ratio > 0 ? modes.slow : modes.fast;
      const tracker = new PeriodTracker();
      let worstShape = 0;
      run([rad(2), ratio * rad(2), 0, 0], 20, 1e-3, "rk4", (s, t) => {
        tracker.sample(t, s[0]);
        if (Math.abs(s[0]) > rad(1)) worstShape = Math.max(worstShape, Math.abs(s[1] / s[0] - ratio));
      });
      const periods = tracker.periods();
      expect(periods.length).toBeGreaterThan(4);
      for (const T of periods) expect(Math.abs(T / mode.period - 1)).toBeLessThan(0.005);
      expect(worstShape).toBeLessThan(0.02);
    });
  }

  it("any other small start mixes both modes: θ₁'s zero crossings are no longer evenly spaced", () => {
    const tracker = new PeriodTracker();
    run([rad(2), 0, 0, 0], 30, 1e-3, "rk4", (s, t) => tracker.sample(t, s[0]));
    const p = tracker.periods();
    expect(Math.max(...p) - Math.min(...p)).toBeGreaterThan(0.1);
  });
});

describe("flips and the energy boundary", () => {
  it("2 cos θ₁ + cos θ₂ > 1 forbids every flip (energy), for equal masses and lengths", () => {
    expect(flipForbiddenEqual(rad(60), 0)).toBe(true); // 2·½ + 1 = 2
    expect(flipForbiddenEqual(rad(90), rad(10))).toBe(false); // just outside: energy allows it
    for (const [a, b] of [[60, 0], [80, 40], [0, 170], [45, 90]]) {
      const s: State = [rad(a), rad(b), 0, 0];
      expect(flipForbiddenEqual(s[0], s[1]), `${a}°, ${b}°`).toBe(true);
      expect(canNeverFlip(s, P), `${a}°, ${b}°`).toBe(true);
      expect(flipTime(s, P, 2e-3, 30), `${a}°, ${b}°`).toBeUndefined();
    }
  });

  it("the general energy test agrees with the closed form for equal masses on a grid of starts", () => {
    for (let a = -180; a <= 180; a += 15) {
      for (let b = -180; b <= 180; b += 15) {
        expect(canNeverFlip([rad(a), rad(b), 0, 0], P), `${a}, ${b}`).toBe(flipForbiddenEqual(rad(a), rad(b)));
      }
    }
  });

  it("the boundary curve a·cos θ₁ + b·cos θ₂ > c is the same test, for any masses and lengths", () => {
    expect(flipBoundary(P)).toEqual({ a: 2, b: 1, c: 1 });
    for (const q of [P, { ...P, m2: 3, l2: 0.5 }, { ...P, m1: 0.4, l2: 1.8 }]) {
      const { a, b, c } = flipBoundary(q);
      for (let x = -180; x <= 180; x += 10) {
        for (let y = -180; y <= 180; y += 10) {
          const [t1, t2] = [rad(x), rad(y)];
          const f = a * Math.cos(t1) + b * Math.cos(t2) - c;
          if (Math.abs(f) > 1e-9) expect(canNeverFlip([t1, t2, 0, 0], q), `${x}, ${y}`).toBe(f > 0);
        }
      }
    }
  });

  it("a high release flips quickly; flips are detected by |θ| passing π", () => {
    const t = flipTime([rad(170), rad(170), 0, 0], P, 1e-3, 20);
    expect(t).toBeDefined();
    expect(t!).toBeLessThan(5);
    expect(hasFlipped([3.2, 0, 0, 0])).toBe(true);
    expect(hasFlipped([3.1, -3.1, 0, 0])).toBe(false);
  });

  it("flip times are symmetric under (θ₁, θ₂) → (−θ₁, −θ₂)", () => {
    for (const [a, b] of [[120, 30], [150, -100], [100, 170]]) {
      const plus = flipTime([rad(a), rad(b), 0, 0], P, 2e-3, 20);
      const minus = flipTime([-rad(a), -rad(b), 0, 0], P, 2e-3, 20);
      expect(minus, `${a}, ${b}`).toBe(plus);
    }
  });
});

describe("chaos: the gap between two nearly identical pendulums", () => {
  function gaps(start: State, nudge: number, seconds: number) {
    let a = start;
    let b: State = [start[0] + nudge, start[1], start[2], start[3]];
    const t: number[] = [];
    const d: number[] = [];
    for (let k = 1; k <= Math.round(seconds / 1e-3); k++) {
      a = step(a, 1e-3, P);
      b = step(b, 1e-3, P);
      if (k % 50 === 0) {
        t.push(k * 1e-3);
        d.push(separation(a, b));
      }
    }
    return { t, d };
  }

  it("a high release: a 1e-9 rad nudge grows to a completely different motion within 20 s", () => {
    const { t, d } = gaps([rad(120), rad(120), 0, 0], 1e-9, 20);
    const visible = t[d.findIndex((x) => x > 1)];
    expect(visible).toBeDefined();
    expect(visible).toBeLessThan(20);
    const lambda = growthRate(t, d)!;
    expect(lambda).toBeGreaterThan(0.5); // per second: errors grow e-fold in under 2 s
  });

  it("a small swing: the same nudge stays tiny (no exponential growth)", () => {
    const { t, d } = gaps([rad(10), rad(10), 0, 0], 1e-9, 30);
    expect(Math.max(...d)).toBeLessThan(1e-6);
    expect(growthRate(t, d)).toBeUndefined(); // never enters the growing range
  });

  it("growthRate recovers a known exponent", () => {
    const t = Array.from({ length: 50 }, (_, i) => i * 0.2);
    const d = t.map((x) => 1e-6 * Math.exp(1.5 * x));
    expect(growthRate(t, d)).toBeCloseTo(1.5, 6);
  });
});

describe("the GPU crowd's starting angles (32-bit)", () => {
  it("keep the nudge when 32-bit floats can hold it, and stay distinct (one float step apart) when they cannot", () => {
    const wide = crowdAngles(rad(120), 1e-3, 5);
    for (let i = 0; i < 5; i++) expect(wide[i]).toBeCloseTo(rad(120) + i * 1e-3, 6);
    const tiny = crowdAngles(rad(120), 1e-9, 100);
    expect(new Set(tiny).size).toBe(100);
    for (let i = 1; i < 100; i++) {
      expect(tiny[i]).toBeGreaterThan(tiny[i - 1]);
      expect(tiny[i] - tiny[i - 1]).toBeLessThan(3e-7); // one step: ~2.4e-7 near 2 rad
    }
    expect(Math.fround(rad(120) + 1e-9)).toBe(Math.fround(rad(120))); // why: the nudge alone would vanish
  });
});

describe("fractal map coordinates", () => {
  it("pixel centres cover the window, θ₂ increasing upwards", () => {
    const [a, b] = pixelAngles(0, 0, 4, FULL_WINDOW);
    expect(a).toBeCloseTo(-Math.PI + Math.PI / 4, 12);
    expect(b).toBeCloseTo(Math.PI - Math.PI / 4, 12);
    const [c, d] = pixelAngles(3, 3, 4, FULL_WINDOW);
    expect(c).toBeCloseTo(Math.PI - Math.PI / 4, 12);
    expect(d).toBeCloseTo(-Math.PI + Math.PI / 4, 12);
    expect(mapAngles(0.5, 0.5, { cx: 1, cy: 2, span: 3 })).toEqual([1, 2]);
  });
});
