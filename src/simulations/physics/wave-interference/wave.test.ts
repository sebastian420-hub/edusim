import { describe, expect, it } from "vitest";
import {
  DEFAULTS,
  detectorReadouts,
  findPeaks,
  intensity,
  measuredFringeSpacing,
  SOURCE_X,
  theoreticalFringeSpacing,
  viewToWorld,
  worldToView,
  wavelength,
} from "./wave";
import type { WaveParams } from "./wave";

const doubleSlit = (patch: Partial<WaveParams> = {}): WaveParams => ({
  ...DEFAULTS,
  mode: "double-slit",
  frequency: 6,
  separation: 0.5,
  slitWidth: 0.1,
  damping: 0,
  ...patch,
});

describe("wave twin: physics", () => {
  it("shows the incident plane wave left of the barrier", () => {
    expect(intensity(doubleSlit(), SOURCE_X - 1, 0.7)).toBeCloseTo(0.5, 10);
  });

  it("is mirror-symmetric about y = 0 for symmetric sources", () => {
    const p = doubleSlit();
    for (const y of [0.2, 0.9, 1.7]) {
      expect(intensity(p, 1.5, y)).toBeCloseTo(intensity(p, 1.5, -y), 10);
    }
  });

  it("measured fringe spacing matches the lambda L / d prediction", () => {
    for (const separation of [0.5, 0.8, 1.2]) {
      const r = detectorReadouts(doubleSlit({ separation }));
      expect(r.theorySpacing).toBeDefined();
      expect(r.measuredSpacing).toBeDefined();
      expect(Math.abs(r.measuredSpacing! / r.theorySpacing! - 1)).toBeLessThan(0.08);
    }
  });

  it("doubling the separation halves the fringe spacing", () => {
    const a = detectorReadouts(doubleSlit({ separation: 0.5 })).measuredSpacing!;
    const b = detectorReadouts(doubleSlit({ separation: 1.0 })).measuredSpacing!;
    expect(b / a).toBeCloseTo(0.5, 1);
  });

  it("a longer wavelength (lower frequency) widens the fringes", () => {
    const hi = detectorReadouts(doubleSlit({ frequency: 6 })).measuredSpacing!;
    const lo = detectorReadouts(doubleSlit({ frequency: 4 })).measuredSpacing!;
    expect(lo).toBeGreaterThan(hi * 1.3);
  });

  it("moving the screen further away widens the fringes", () => {
    const near = detectorReadouts(doubleSlit({ detectorX: 1.0 })).measuredSpacing!;
    const far = detectorReadouts(doubleSlit({ detectorX: 2.5 })).measuredSpacing!;
    expect(far).toBeGreaterThan(near);
  });

  it("a narrower single slit spreads the wave over a wider angle", () => {
    const spread = (slitWidth: number) => {
      const p = doubleSlit({ mode: "single-slit", slitWidth });
      return intensity(p, 2.0, 1.5) / intensity(p, 2.0, 0);
    };
    expect(spread(0.1)).toBeGreaterThan(spread(0.5));
  });

  it("only defines a theoretical spacing for two-source modes", () => {
    expect(theoreticalFringeSpacing(doubleSlit({ mode: "single-slit" }))).toBeUndefined();
    expect(theoreticalFringeSpacing(doubleSlit({ mode: "point" }))).toBeUndefined();
    expect(theoreticalFringeSpacing(doubleSlit())).toBeCloseTo((wavelength({ waveSpeed: 1, frequency: 6 }) * 3) / 0.5, 10);
  });
});

describe("wave twin: peak finding and geometry", () => {
  it("finds evenly spaced peaks and ignores weak ones", () => {
    const values = Array.from({ length: 401 }, (_, i) => 0.5 + 0.5 * Math.cos((i / 400) * 8 * Math.PI)); // 4 periods
    expect(findPeaks(values)).toHaveLength(3); // the two edge maxima are not interior peaks
    expect(measuredFringeSpacing(values)).toBeCloseTo(1, 1); // 4 units / 4 periods
    expect(measuredFringeSpacing([0, 1, 0])).toBeUndefined();
  });

  it("round-trips view and world coordinates", () => {
    const [x, y] = viewToWorld(300, 120, 800, 500);
    const [px, py] = worldToView(x, y, 800, 500);
    expect(px).toBeCloseTo(300, 8);
    expect(py).toBeCloseTo(120, 8);
    expect(viewToWorld(400, 250, 800, 500)).toEqual([0, 0]);
    expect(viewToWorld(0, 0, 800, 500)[1]).toBe(2); // top edge is y = +2
  });
});
