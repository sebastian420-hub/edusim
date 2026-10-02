import { describe, expect, it } from "vitest";
import {
  appliedCurrent,
  countSpikes,
  DEFAULT_PARAMS,
  fiAnalysis,
  HISTORY_SAMPLES,
  rates,
  REST_STATE,
  sampleAt,
  SAMPLE_EVERY,
  DT_MS,
  simulate,
  spikesPerCycle,
  spikeTimes,
  steadyFiringRate,
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

describe("measurements", () => {
  it("interpolates spike times between samples", () => {
    const trace = { t: [0, 1, 2, 3], V: [-10, -2, 6, -20] };
    expect(spikeTimes(trace)).toEqual([1 + 2 / 8]); // crosses 0 mV a quarter of the way from t=1 to t=2
  });

  it("steady rate is 0 below threshold and ~68 Hz at 10 uA/cm2 (classic HH, 6.3 degC)", () => {
    expect(steadyFiringRate(params({ I_inj: 3 }))).toBe(0);
    const rate = steadyFiringRate(params({ I_inj: 10 }));
    expect(rate).toBeGreaterThan(60);
    expect(rate).toBeLessThan(76);
  });

  it("rate rises with current and with temperature (Q10 = 3)", () => {
    const at = (I_inj: number, temperature = 6.3) => steadyFiringRate(params({ I_inj, temperature }));
    expect(at(20)).toBeGreaterThan(at(10));
    expect(at(10, 16.3) / at(10)).toBeGreaterThan(2);
    expect(at(10, 16.3) / at(10)).toBeLessThan(3);
  });

  it("finds the onset current (type II: abrupt jump from silence to ~50 Hz)", () => {
    const fi = fiAnalysis(DEFAULT_PARAMS);
    expect(fi.rheobase).toBeGreaterThan(5);
    expect(fi.rheobase).toBeLessThan(6.5);
    const firstRate = fi.rates.find((r) => r > 0)!;
    expect(firstRate).toBeGreaterThan(35); // no slow firing: it starts fast
    for (let i = 1; i < fi.rates.length; i++) expect(fi.rates[i]).toBeGreaterThanOrEqual(fi.rates[i - 1] - 1); // monotonic
  });

  it("onset moves with the channels: no sodium means no firing at any current", () => {
    expect(fiAnalysis(params({ g_Na: 0 })).rheobase).toBeUndefined();
  });

  it("counts spikes per stimulus cycle", () => {
    expect(spikesPerCycle(params({ pulse_mode: 1, I_inj: 10 }))).toBe(1);
    expect(spikesPerCycle(params({ pulse_mode: 1, I_inj: 2 }))).toBe(0);
  });
});

describe("sampleAt (GPU ring buffer)", () => {
  // Fill the ring with the sample index in V so we can see which slot is returned.
  const ring = new Float32Array(HISTORY_SAMPLES * 4);
  for (let i = 0; i < HISTORY_SAMPLES; i++) ring[i * 4] = i;

  it("the oldest sample is at the head, the newest just before it", () => {
    const head = 100;
    expect(sampleAt(ring, head, 0).V).toBe(100);
    expect(sampleAt(ring, head, 1).V).toBe(99);
  });

  it("wraps around the end of the buffer", () => {
    const head = HISTORY_SAMPLES - 10;
    expect(sampleAt(ring, head, 0).V).toBe(HISTORY_SAMPLES - 10);
    expect(sampleAt(ring, head, 10 / (HISTORY_SAMPLES - 1)).V).toBe(0);
  });

  it("reports how long ago the sample was recorded", () => {
    expect(sampleAt(ring, 0, 1).msAgo).toBe(0);
    expect(sampleAt(ring, 0, 0).msAgo).toBeCloseTo((HISTORY_SAMPLES - 1) * SAMPLE_EVERY * DT_MS, 6);
  });

  it("clamps the cursor position", () => {
    expect(sampleAt(ring, 0, -3).V).toBe(0);
    expect(sampleAt(ring, 0, 9).V).toBe(HISTORY_SAMPLES - 1);
  });
});
