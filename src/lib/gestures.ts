/**
 * Two-finger gestures for canvases: pinch to zoom about the fingers' midpoint, and drag with two fingers to pan.
 * Feed it every pointer event of the canvas. While two or more pointers are down, `move` returns the gesture
 * step to apply and the caller should ignore its single-pointer behaviour (drawing, dragging bodies…).
 * Pure logic, no DOM.
 */
export interface PinchStep {
  /** Move the view by this many CSS pixels. */
  dx: number;
  dy: number;
  /** Scale the view by `factor` about the point (x, y) in CSS pixels. */
  factor: number;
  x: number;
  y: number;
}

export function createPinchTracker() {
  const points = new Map<number, { x: number; y: number }>();

  const geometry = () => {
    const [a, b] = [...points.values()];
    return { cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) };
  };

  return {
    /** Pointers currently down. */
    get count() {
      return points.size;
    },
    /** True while a two-finger gesture is in progress. */
    get active() {
      return points.size >= 2;
    },
    down(id: number, x: number, y: number) {
      points.set(id, { x, y });
    },
    /** The gesture step for this move, or null when it is not part of a two-finger gesture. */
    move(id: number, x: number, y: number): PinchStep | null {
      if (!points.has(id)) return null;
      if (points.size < 2) {
        points.set(id, { x, y });
        return null;
      }
      const before = geometry();
      points.set(id, { x, y });
      const after = geometry();
      return { dx: after.cx - before.cx, dy: after.cy - before.cy, factor: before.d > 0 && after.d > 0 ? after.d / before.d : 1, x: after.cx, y: after.cy };
    },
    up(id: number) {
      points.delete(id);
    },
    reset() {
      points.clear();
    },
  };
}

export type PinchTracker = ReturnType<typeof createPinchTracker>;
