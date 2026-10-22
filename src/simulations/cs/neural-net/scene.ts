/**
 * Where things go on the Playground canvas (device pixels, y down): the square plot of the plane and the network
 * diagram, one column per layer. The shaders compute node positions with the same formula (nodeCenter), so the
 * page can hit-test and label nodes without reading anything back.
 */
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface SceneLayout {
  plot: Rect;
  diagram: Rect;
  /** Columns: the inputs, each hidden layer, the output. */
  columns: number[];
  /** Vertical distance between neighbouring nodes, and a node's side. */
  spacing: number;
  node: number;
}

export function sceneLayout(width: number, height: number, columns: number[], dpr = 1): SceneLayout {
  const m = Math.round(16 * dpr);
  const wide = width >= height * 1.15;
  let plot: Rect;
  let diagram: Rect;
  if (wide) {
    const side = Math.max(1, Math.min(height - 2 * m, (width - 3 * m) * 0.55));
    plot = { x: m, y: (height - side) / 2, w: side, h: side };
    diagram = { x: plot.x + side + m, y: m + 18 * dpr, w: Math.max(1, width - side - 3 * m), h: Math.max(1, height - 2 * m - 18 * dpr) };
  } else {
    const side = Math.max(1, Math.min(width - 2 * m, (height - 3 * m) * 0.58));
    plot = { x: (width - side) / 2, y: m, w: side, h: side };
    diagram = { x: m, y: plot.y + side + m + 14 * dpr, w: Math.max(1, width - 2 * m), h: Math.max(1, height - side - 3 * m - 14 * dpr) };
  }
  const tallest = Math.max(...columns);
  const colW = diagram.w / columns.length;
  const spacing = Math.min(diagram.h / tallest, 64 * dpr);
  const node = Math.max(3, Math.min(spacing * 0.82, colW * 0.42, 44 * dpr));
  return { plot, diagram, columns, spacing, node };
}

export function nodeCenter(s: SceneLayout, column: number, k: number): [number, number] {
  const n = s.columns[column];
  return [s.diagram.x + ((column + 0.5) * s.diagram.w) / s.columns.length, s.diagram.y + s.diagram.h / 2 + (k - (n - 1) / 2) * s.spacing];
}

/** The plane point (x, y ∈ [−1, 1], y up) under a device pixel of the plot, or null outside it. */
export function planeAt(s: SceneLayout, px: number, py: number): [number, number] | null {
  const u = (px - s.plot.x) / s.plot.w;
  const v = (py - s.plot.y) / s.plot.h;
  if (u < 0 || v < 0 || u > 1 || v > 1) return null;
  return [u * 2 - 1, 1 - v * 2];
}
