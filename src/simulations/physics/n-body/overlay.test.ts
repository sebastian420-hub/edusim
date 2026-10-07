import { describe, expect, it } from "vitest";
import { circularSpeed } from "./nbody";
import { arrowYears, hitTest, scaleBarLength, toScreen } from "./overlay";
import type { OverlayState } from "./overlay";
import { orbitPreset } from "./presets";

describe("orbit-lab overlay", () => {
  it("arrow length: a quarter of a typical planet's r/v (Earth: 0.25/2π yr), ignoring the star", () => {
    expect(arrowYears(orbitPreset("sun-earth").bodies)).toBeCloseTo(0.25 / circularSpeed(1, 1), 3);
    const inner = arrowYears(orbitPreset("inner-planets").bodies);
    expect(inner).toBeGreaterThan(0.01);
    expect(inner).toBeLessThan(0.04);
  });

  it("maps world to screen with y up, and hit-tests arrow tips before bodies", () => {
    const s: OverlayState = { bodies: orbitPreset("sun-earth").bodies, names: ["Sun", "Earth"], colors: [], view: { center: [0, 0], pxPerUnit: 200 }, pixelRatio: 2, width: 400, height: 300, selected: null, arrows: true, arrowTime: 0.05 };
    expect(toScreen(s, 1, 0)).toEqual([300, 150]);
    expect(toScreen(s, 0, 1)).toEqual([200, 50]);
    expect(hitTest(s, 302, 151)).toEqual({ kind: "body", index: 1 });
    const [tx, ty] = toScreen(s, 1, 2 * Math.PI * 0.05);
    expect(hitTest(s, tx, ty)).toEqual({ kind: "arrow", index: 1 });
    expect(hitTest(s, 10, 10)).toBeNull();
  });

  it("scale bar picks 1, 2 or 5 × 10ⁿ", () => {
    expect(scaleBarLength(3.2)).toBe(0.5);
    expect(scaleBarLength(12)).toBe(2);
    expect(scaleBarLength(30)).toBe(5);
  });
});
