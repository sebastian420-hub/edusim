import { describe, expect, it } from "vitest";
import { SIM_LOADERS } from "@/simulations/loaders";
import { findSimulation, implementedSimulations, SIMULATIONS, SUBJECTS } from "./subjects";

describe("simulation catalog", () => {
  it("has unique ids per subject and valid subjects", () => {
    const keys = SIMULATIONS.map((s) => `${s.subject}/${s.id}`);
    expect(new Set(keys).size).toBe(keys.length);
    for (const s of SIMULATIONS) expect(SUBJECTS).toHaveProperty(s.subject);
  });

  it("derives every path from subject and id", () => {
    for (const s of SIMULATIONS) expect(s.path).toBe(`/${s.subject}/${s.id}`);
  });

  it("registers a loader for exactly the implemented simulations", () => {
    const implemented = implementedSimulations().map((s) => s.id).sort();
    expect(Object.keys(SIM_LOADERS).sort()).toEqual(implemented);
  });

  it("finds simulations by subject and id", () => {
    expect(findSimulation("physics", "wave-interference")?.title).toMatch(/Wave/);
    expect(findSimulation("physics", "hodgkin-huxley")).toBeUndefined();
  });
});
