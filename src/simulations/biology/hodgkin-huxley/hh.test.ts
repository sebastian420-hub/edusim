import { describe, expect, it } from "vitest";
import {
  appliedCurrent,
  countSpikes,
  DEFAULT_PARAMS,
  rates,
  REST_STATE,
  simulate,
  temperatureFactor,
} from "./hh";
import type { HHParams } from "./hh";

const params = (patch: Partial<HHParams>): HHParams => ({ ...DEFAULT_PARAMS, ...patch });

describe("Hodgkin-Huxley reference model", () => {
  it("rates are finite at the removable singularities", () => {
    for (const V of [-40, -55, -40.0004, -54.9996]) {
      for (const v of Object.values(rates(V))) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("rests near -65 mV with no stimulus", () => {
    const trace = simulate(params({ I_inj: 0 }), 50);
    expect(trace.V.at(-1)).toBeGreaterThan(-67);
    expect(trace.V.at(-1)).toBeLessThan(-63);
    expect(countSpikes(trace.V)).toBe(0);
  });

  it("fires a full action potential for a suprathreshold pulse", () => {
    const trace = simulate(params({ pulse_mode: 1, I_inj: 10 }), 20);
    expect(Math.max(...trace.V)).toBeGreaterThan(30);
    expect(Math.min(...trace.V)).toBeLessThan(-65); // after-hyperpolarisation
    expect(countSpikes(trace.V)).toBe(1);
  });

  it("does not fire for a subthreshold pulse", () => {
    const trace = simulate(params({ pulse_mode: 1, I_inj: 2 }), 20);
    expect(Math.max(...trace.V)).toBeLessThan(-40);
  });

  it("fires repetitively for sustained DC current", () => {
    expect(countSpikes(simulate(params({ I_inj: 10 }), 100).V)).toBeGreaterThanOrEqual(5);
  });

  it("TTX (no sodium conductance) abolishes the spike", () => {
    const trace = simulate(params({ pulse_mode: 1, g_Na: 0, I_inj: 20 }), 20);
    expect(Math.max(...trace.V)).toBeLessThan(0);
  });

  it("anode break: a hyperpolarising pulse can trigger a rebound spike", () => {
    const trace = simulate(params({ pulse_mode: 1, I_inj: -20 }), 25);
    expect(Math.min(...trace.V)).toBeLessThan(-70);
  });

  it("warmer temperature raises the DC firing rate", () => {
    const cold = countSpikes(simulate(params({ I_inj: 10, temperature: 6.3 }), 100).V);
    const warm = countSpikes(simulate(params({ I_inj: 10, temperature: 16.3 }), 100).V);
    expect(warm).toBeGreaterThan(cold);
    expect(temperatureFactor(16.3)).toBeCloseTo(3, 10);
  });

  it("twin-pulse mode applies current in two windows per period", () => {
    const p = { I_inj: 10, pulse_mode: 2 as const };
    expect(appliedCurrent(5.5, p)).toBe(10);
    expect(appliedCurrent(9, p)).toBe(0);
    expect(appliedCurrent(13.5 + 25, p)).toBe(10);
  });

  it("starts from the documented resting state", () => {
    expect(REST_STATE.V).toBe(-65);
  });
});
