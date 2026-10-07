import { describe, expect, it } from "vitest";
import { createPinchTracker } from "./gestures";
import type { PinchStep } from "./gestures";

/** Accumulates the steps the way a view would: pans add up, zoom factors multiply. */
function total(steps: (PinchStep | null)[]) {
  const t = { dx: 0, dy: 0, factor: 1, at: [0, 0] };
  for (const s of steps) {
    if (!s) continue;
    t.dx += s.dx;
    t.dy += s.dy;
    t.factor *= s.factor;
    t.at = [s.x, s.y];
  }
  return t;
}

describe("pinch tracker", () => {
  it("ignores a single finger (that is the caller's tool)", () => {
    const t = createPinchTracker();
    t.down(1, 10, 10);
    expect(t.move(1, 50, 50)).toBeNull();
    expect(t.active).toBe(false);
  });

  it("spreading two fingers zooms in about their midpoint", () => {
    const t = createPinchTracker();
    t.down(1, 100, 100);
    t.down(2, 200, 100);
    expect(t.active).toBe(true);
    const g = total([t.move(1, 50, 100), t.move(2, 250, 100)]);
    expect(g.factor).toBeCloseTo(2, 10);
    expect(g.at).toEqual([150, 100]);
    expect(g.dx).toBeCloseTo(0, 10);
  });

  it("moving both fingers together pans without zooming", () => {
    const t = createPinchTracker();
    t.down(1, 0, 0);
    t.down(2, 100, 0);
    const g = total([t.move(1, 30, 40), t.move(2, 130, 40)]);
    expect(g.dx).toBeCloseTo(30, 10);
    expect(g.dy).toBeCloseTo(40, 10);
    expect(g.factor).toBeCloseTo(1, 10);
  });

  it("ends when a finger lifts, and unknown pointers are ignored", () => {
    const t = createPinchTracker();
    t.down(1, 0, 0);
    t.down(2, 10, 0);
    t.up(2);
    expect(t.active).toBe(false);
    expect(t.move(9, 5, 5)).toBeNull();
    t.reset();
    expect(t.count).toBe(0);
  });
});
