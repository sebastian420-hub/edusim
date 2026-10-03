import { describe, expect, it } from "vitest";
import { niceMax } from "./graph";

describe("niceMax", () => {
  it("rounds up to a short round number", () => {
    expect(niceMax(3)).toBe(4);
    expect(niceMax(36)).toBe(40);
    expect(niceMax(130)).toBe(150);
    expect(niceMax(213)).toBe(250);
    expect(niceMax(13_218)).toBe(15_000);
  });

  it("always leaves headroom above the data and never collapses the axis", () => {
    for (const value of [0, 1, 4, 5, 9, 10, 99, 100, 101, 999, 1000, 65_536, 4_194_304]) {
      const max = niceMax(value);
      expect(max, `${value}`).toBeGreaterThanOrEqual(value);
      expect(max, `${value}`).toBeGreaterThanOrEqual(4);
      expect(max, `${value}`).toBeLessThan(Math.max(4, value) * 1.7);
    }
  });

  it("is stable for small changes in the data (the scale does not jitter)", () => {
    expect(niceMax(120)).toBe(niceMax(125));
    expect(niceMax(120)).toBe(niceMax(130));
  });
});
