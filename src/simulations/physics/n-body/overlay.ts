import { primaryIndex } from "./nbody";
import type { Body } from "./nbody";
import type { View } from "./renderer";

/**
 * A velocity arrow points to where the body would be `arrowYears` later at constant velocity. The time is chosen
 * per system (a quarter of the median r/v of the planets around the primary), so arrows stay readable for
 * Mercury and Jupiter alike.
 */
export function arrowYears(bodies: readonly Body[]): number {
  if (bodies.length < 2) return 0.05;
  const p = bodies[primaryIndex(bodies)];
  const ratios = bodies
    .filter((b) => b !== p)
    .map((b) => Math.hypot(b.x - p.x, b.y - p.y) / Math.max(1e-6, Math.hypot(b.vx - p.vx, b.vy - p.vy)))
    .sort((a, b) => a - b);
  return Math.min(0.5, Math.max(0.005, 0.25 * ratios[Math.floor((ratios.length - 1) / 2)]));
}
const HIT_BODY_PX = 16;
const HIT_ARROW_PX = 12;

export interface OverlayState {
  bodies: Body[];
  names: string[];
  colors: string[];
  view: View;
  /** Device pixels per CSS pixel. */
  pixelRatio: number;
  /** CSS size of the canvas. */
  width: number;
  height: number;
  selected: number | null;
  arrows: boolean;
  /** Arrow length in years of motion (see arrowYears). */
  arrowTime: number;
}

/** World → canvas CSS pixels (world y points up). */
export function toScreen(s: Pick<OverlayState, "view" | "pixelRatio" | "width" | "height">, x: number, y: number): [number, number] {
  const k = s.view.pxPerUnit / s.pixelRatio;
  return [s.width / 2 + (x - s.view.center[0]) * k, s.height / 2 - (y - s.view.center[1]) * k];
}

export type Hit = { kind: "body" | "arrow"; index: number } | null;

/** What is under the pointer: an arrow tip wins over a body (tips are small targets). */
export function hitTest(s: OverlayState, xCss: number, yCss: number): Hit {
  let best: Hit = null;
  let bestDistance = Infinity;
  s.bodies.forEach((b, i) => {
    if (s.arrows) {
      const [tx, ty] = toScreen(s, b.x + b.vx * s.arrowTime, b.y + b.vy * s.arrowTime);
      const d = Math.hypot(tx - xCss, ty - yCss);
      if (d < HIT_ARROW_PX && d - 4 < bestDistance) {
        best = { kind: "arrow", index: i };
        bestDistance = d - 4;
      }
    }
    const [bx, by] = toScreen(s, b.x, b.y);
    const d = Math.hypot(bx - xCss, by - yCss);
    if (d < HIT_BODY_PX && d < bestDistance) {
      best = { kind: "body", index: i };
      bestDistance = d;
    }
  });
  return best;
}

/** A "nice" length (1, 2 or 5 × 10ⁿ) about a fifth of the canvas wide, for the scale bar. */
export function scaleBarLength(unitsAcross: number): number {
  const target = unitsAcross / 5;
  const base = 10 ** Math.floor(Math.log10(target));
  return [5, 2, 1].map((f) => f * base).find((v) => v <= target) ?? base;
}

const formatLength = (v: number) => (v >= 1 ? `${v} AU` : `${Number(v.toPrecision(2))} AU`);

/** Draws labels, velocity arrows, the selection ring and a scale bar on a 2D canvas over the WebGPU one. */
export function drawOverlay(ctx: CanvasRenderingContext2D, s: OverlayState): void {
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, s.width, s.height);
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.lineCap = "round";

  s.bodies.forEach((b, i) => {
    const [x, y] = toScreen(s, b.x, b.y);
    const color = s.colors[i] ?? "#cbd5e1";
    if (s.arrows) {
      const [tx, ty] = toScreen(s, b.x + b.vx * s.arrowTime, b.y + b.vy * s.arrowTime);
      const len = Math.hypot(tx - x, ty - y);
      if (len > 4) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(tx, ty);
        const ang = Math.atan2(ty - y, tx - x);
        ctx.moveTo(tx, ty);
        ctx.lineTo(tx - 7 * Math.cos(ang - 0.45), ty - 7 * Math.sin(ang - 0.45));
        ctx.moveTo(tx, ty);
        ctx.lineTo(tx - 7 * Math.cos(ang + 0.45), ty - 7 * Math.sin(ang + 0.45));
        ctx.stroke();
        // Drag handle at the tip.
        ctx.globalAlpha = 0.35;
        ctx.beginPath();
        ctx.arc(tx, ty, 5, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    if (s.selected === i) {
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "rgba(226, 232, 240, 0.85)";
    ctx.fillText(s.names[i] ?? "", x + 10, y - 9);
  });

  // Scale bar, bottom right.
  const k = s.view.pxPerUnit / s.pixelRatio;
  const length = scaleBarLength(s.width / k);
  const px = length * k;
  const x1 = s.width - 16;
  const y1 = s.height - 16;
  ctx.strokeStyle = "rgba(226, 232, 240, 0.7)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x1 - px, y1 - 4);
  ctx.lineTo(x1 - px, y1);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x1, y1 - 4);
  ctx.stroke();
  ctx.fillStyle = "rgba(226, 232, 240, 0.8)";
  ctx.textAlign = "right";
  ctx.fillText(formatLength(length), x1, y1 - 7);
  ctx.textAlign = "left";
}
