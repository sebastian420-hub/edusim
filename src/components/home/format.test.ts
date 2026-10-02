import { describe, expect, it } from "vitest";
import { plannedBySubject, implementedSimulations, SIMULATIONS } from "@/lib/subjects";
import { roman } from "./format";

describe("roman numerals for plates", () => {
  it("formats small numbers", () => {
    expect([1, 2, 3, 4, 5, 6, 9, 10, 12].map(roman)).toEqual(["I", "II", "III", "IV", "V", "VI", "IX", "X", "XII"]);
  });
});

describe("home page data", () => {
  it("every implemented simulation has a one-line tagline for its plate", () => {
    for (const s of implementedSimulations()) expect(s.tagline, s.id).toBeTruthy();
  });

  it("the planned index lists exactly the simulations that are not implemented, grouped by subject", () => {
    const groups = plannedBySubject();
    const listed = groups.flatMap((g) => g.sims.map((s) => s.id));
    const expected = SIMULATIONS.filter((s) => !s.implemented).map((s) => s.id);
    expect(listed.sort()).toEqual(expected.sort());
    for (const g of groups) for (const s of g.sims) expect(s.subject).toBe(g.subject);
  });
});
