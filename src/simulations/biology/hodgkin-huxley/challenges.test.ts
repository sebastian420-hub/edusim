import { describe, expect, it } from "vitest";
import { HH_CHALLENGES } from "./challenges";
import { fiAnalysis } from "./hh";
import { computeReadouts, HH_DEFAULTS, toParams } from "./settings";
import type { HHSettings } from "./settings";

const state = (settings: HHSettings) => ({
  params: settings,
  readouts: computeReadouts(settings, fiAnalysis(toParams({ ...HH_DEFAULTS, ...settings, pulse_mode: 0 }))),
});

// The change a student is expected to discover for each challenge.
const SOLUTIONS: Record<string, Partial<HHSettings>> = {
  threshold: { I_inj: 6.5 },
  ttx: { g_Na: 30 },
  temperature: { temperature: 16.3 },
};

describe("hodgkin-huxley challenges", () => {
  it("have unique ids and exactly one correct prediction each", () => {
    expect(new Set(HH_CHALLENGES.map((c) => c.id)).size).toBe(HH_CHALLENGES.length);
    for (const c of HH_CHALLENGES) expect(c.prediction?.options.filter((o) => o.correct)).toHaveLength(1);
  });

  it("are not already solved by their own setup", () => {
    for (const c of HH_CHALLENGES) expect(c.goal.check(state({ ...HH_DEFAULTS, ...c.setup })), c.id).toBe(false);
  });

  it("are solved by the intended change", () => {
    for (const c of HH_CHALLENGES) {
      const solution = SOLUTIONS[c.id];
      expect(solution, `a solution is defined for ${c.id}`).toBeDefined();
      expect(c.goal.check(state({ ...HH_DEFAULTS, ...c.setup, ...solution })), c.id).toBe(true);
    }
  });

  it("cannot be solved by a shortcut that dodges the lesson", () => {
    const byId = (id: string) => HH_CHALLENGES.find((c) => c.id === id)!;
    // Silencing the neuron by removing the current doesn't count as the TTX lesson.
    const ttx = byId("ttx");
    expect(ttx.goal.check(state({ ...HH_DEFAULTS, ...ttx.setup, I_inj: 0 }))).toBe(false);
    // Overshooting the threshold isn't "as little current as you can".
    const th = byId("threshold");
    expect(th.goal.check(state({ ...HH_DEFAULTS, ...th.setup, I_inj: 20 }))).toBe(false);
    // Changing the channels doesn't count as "only the temperature".
    const temp = byId("temperature");
    expect(temp.goal.check(state({ ...HH_DEFAULTS, ...temp.setup, I_inj: 25 }))).toBe(false);
  });
});
