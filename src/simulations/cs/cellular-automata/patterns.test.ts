import { describe, expect, it } from "vitest";
import { DEFAULT_PATTERN, getPatternBounds, patterns, rasterizePattern } from "./patterns";

describe("patterns", () => {
  it("has unique names and includes the default pattern", () => {
    const names = patterns.map((p) => p.name);
    expect(new Set(names).size).toBe(names.length);
    expect(names).toContain(DEFAULT_PATTERN);
  });

  it("computes bounds", () => {
    const gun = patterns.find((p) => p.name === DEFAULT_PATTERN)!;
    expect(getPatternBounds(gun)).toEqual({ width: 36, height: 9 });
  });

  it("rasterises every pattern fully inside a 256 grid", () => {
    for (const p of patterns) {
      const cells = rasterizePattern(p, 256, 256);
      expect(cells.reduce((a, b) => a + b, 0)).toBe(p.points.length);
    }
  });

  it("centres the pattern and drops cells that fall outside a tiny grid", () => {
    const glider = patterns.find((p) => p.name === "Glider")!;
    const cells = rasterizePattern(glider, 8, 8);
    // 3x3 glider centred on an 8x8 grid starts at (2, 2).
    const alive = [...cells.keys()].filter((i) => cells[i]).map((i) => [i % 8, Math.floor(i / 8)]);
    expect(alive).toEqual([[3, 2], [4, 3], [2, 4], [3, 4], [4, 4]]);
    // The 36-wide gun doesn't fit an 8x8 grid: out-of-range cells are dropped, not wrapped.
    const gun = rasterizePattern(patterns.find((p) => p.name === DEFAULT_PATTERN)!, 8, 8);
    expect(gun.length).toBe(64);
    expect(gun.reduce((a, b) => a + b, 0)).toBeLessThan(36);
  });
});
