import { describe, expect, it } from "vitest";
import { decodeParams, encodeParams } from "@/lib/urlState";
import { FULL_WINDOW, mapAngles } from "./pendulum";
import { MIN_SPAN, panWindow, PENDULUM_DEFAULTS, PENDULUM_SCHEMA, zoomWindow } from "./settings";
import type { PendulumSettings } from "./settings";

const decode = (search: string) => decodeParams(PENDULUM_SCHEMA, PENDULUM_DEFAULTS, search);
const encode = (s: PendulumSettings) => encodeParams(PENDULUM_SCHEMA, PENDULUM_DEFAULTS, s);

describe("double pendulum settings", () => {
  it("round-trip through a link, and the defaults make an empty link", () => {
    expect(encode(PENDULUM_DEFAULTS)).toBe("");
    const changed: PendulumSettings = { ...PENDULUM_DEFAULTS, view: "fractal", a1: 33.25, a2: -140, m2: 2, l2: 0.5, g: 3.7, count: 10000, nudge: -6, res: 1024, boundary: false, integrator: "euler" };
    expect(decode(encode(changed))).toEqual(changed);
  });

  it("a deep fractal zoom survives its link to 7 significant digits", () => {
    const deep = { ...PENDULUM_DEFAULTS, view: "fractal" as const, fx: 2.0123456, fy: -1.2345678, fspan: 0.0012345 };
    const back = decode(encode(deep));
    expect(back.fx).toBeCloseTo(deep.fx, 6);
    expect(back.fy).toBeCloseTo(deep.fy, 6);
    expect(back.fspan / deep.fspan).toBeCloseTo(1, 5);
  });

  it("hostile links are ignored or clamped, never crash", () => {
    const s = decode("?view=wormhole&a1=1e9&a2=abc&m2=-1&count=7&res=4096&nudge=-99&fspan=0&integrator=rk45&g=");
    expect(s).toEqual({ ...PENDULUM_DEFAULTS, a1: 180, m2: 0.1, nudge: -12, fspan: MIN_SPAN });
  });

  it("zooming keeps the point under the cursor fixed; panning moves by a fraction of the window", () => {
    const zoomed = zoomWindow(FULL_WINDOW, 4, 0.8, 0.3);
    expect(zoomed.span).toBeCloseTo(FULL_WINDOW.span / 4, 12);
    const before = mapAngles(0.8, 0.3, FULL_WINDOW);
    const after = mapAngles(0.8, 0.3, zoomed);
    expect(after[0]).toBeCloseTo(before[0], 12);
    expect(after[1]).toBeCloseTo(before[1], 12);
    expect(zoomWindow(FULL_WINDOW, 1e-3, 0.5, 0.5).span).toBe(2 * Math.PI); // never wider than all starts
    expect(zoomWindow({ cx: 0, cy: 0, span: 0.002 }, 100, 0.5, 0.5).span).toBe(MIN_SPAN);
    const panned = panWindow(FULL_WINDOW, 0.25, 0.1); // dragged right and down: the view moves left and up
    expect(panned.cx).toBeCloseTo(-Math.PI / 2, 12);
    expect(panned.cy).toBeCloseTo(0.2 * Math.PI, 12);
  });
});
