import { describe, expect, it } from "vitest";
import { WAVE_CHALLENGES } from "./challenges";
import { DEFAULTS, detectorReadouts } from "./wave";
import type { WaveParams } from "./wave";

const state = (params: WaveParams) => ({ params, readouts: detectorReadouts(params) });

// A known solution for each challenge: the change a student is expected to discover.
const SOLUTIONS: Record<string, Partial<WaveParams>> = {
  separation: { separation: 1.0 },
  wavelength: { frequency: 3 },
  diffraction: { slitWidth: 0.1 },
};

describe("wave challenges", () => {
  it("have unique ids and exactly one correct prediction each", () => {
    expect(new Set(WAVE_CHALLENGES.map((c) => c.id)).size).toBe(WAVE_CHALLENGES.length);
    for (const c of WAVE_CHALLENGES) {
      expect(c.prediction?.options.filter((o) => o.correct)).toHaveLength(1);
    }
  });

  it("are not already solved by their own setup", () => {
    for (const c of WAVE_CHALLENGES) {
      expect(c.goal.check(state({ ...DEFAULTS, ...c.setup })), c.id).toBe(false);
    }
  });

  it("are solved by the intended change", () => {
    for (const c of WAVE_CHALLENGES) {
      const solution = SOLUTIONS[c.id];
      expect(solution, `a solution is defined for ${c.id}`).toBeDefined();
      expect(c.goal.check(state({ ...DEFAULTS, ...c.setup, ...solution })), c.id).toBe(true);
    }
  });

  it("start from a setup in which the measurement they rely on is available", () => {
    const sep = WAVE_CHALLENGES.find((c) => c.id === "separation")!;
    const start = detectorReadouts({ ...DEFAULTS, ...sep.setup });
    const doubled = detectorReadouts({ ...DEFAULTS, ...sep.setup, separation: 1.0 });
    expect(start.measuredSpacing).toBeDefined();
    expect(doubled.measuredSpacing! / start.measuredSpacing!).toBeCloseTo(0.5, 1);
  });
});
