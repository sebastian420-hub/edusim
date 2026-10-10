import { describe, expect, it } from "vitest";
import { DT, extracellular, runFibre, restState, scratchFor, stepCable, stimulusAt } from "./cable";
import { AXON_CHALLENGES } from "./challenges";
import type { AxonReadouts } from "./challenges";
import { axonReadout } from "./measure";
import { axonFibre, compartmentAt, NERVE_HEIGHT, NERVE_SWEEP_MS, nerveComposition, nerveFibre, nerveStimulus, sweepFor } from "./model";
import { AXON_DEFAULTS } from "./settings";
import type { AxonSettings } from "./settings";
import { capLatencies } from "./sim";

const challenge = (id: string) => AXON_CHALLENGES.find((c) => c.id === id)!;
const settingsFor = (id: string, patch: Partial<AxonSettings> = {}): AxonSettings => ({ ...AXON_DEFAULTS, ...challenge(id).setup, ...patch });

/** A finished sweep of the Axon view, measured as the simulation measures it (64-bit twin). */
function sweep(s: AxonSettings): AxonReadouts {
  const f = axonFibre(s);
  const sw = sweepFor(s, f);
  const r = runFibre(f, s.temp, sw.b ? [sw.a, sw.b] : [sw.a], sw.ms);
  const peaks: [number, number] = [r.peak[compartmentAt(f, Math.min(s.e1, s.e2))], r.peak[compartmentAt(f, Math.max(s.e1, s.e2))]];
  return { view: "axon", axon: axonReadout(f, r.arrival, r.crossings, s.e1, s.e2, peaks), nerve: null };
}

const passes = (id: string, s: AxonSettings, r: AxonReadouts) => challenge(id).goal.check({ params: s, readouts: r });

describe("axon challenges", () => {
  it("each has exactly one correct prediction", () => {
    for (const c of AXON_CHALLENGES) expect(c.prediction?.options.filter((o) => o.correct)).toHaveLength(1);
  });

  it("all or none: 1.2× and 2.4× threshold make spikes of the same height (about +25 mV)", () => {
    const weak = sweep(settingsFor("all-or-none"));
    const strong = sweep(settingsFor("all-or-none", { stim: 2.4 }));
    expect(Math.abs(weak.axon!.peak2! - strong.axon!.peak2!)).toBeLessThan(0.5);
    expect(strong.axon!.peak2!).toBeCloseTo(25, -1);
    expect(passes("all-or-none", settingsFor("all-or-none"), weak)).toBe(false);
    expect(passes("all-or-none", settingsFor("all-or-none", { stim: 2.4 }), strong)).toBe(true);
  });

  it("thick or thin: 476 µm → 18.5 m/s, 952 µm → 26 m/s (√2), measured between the electrodes", () => {
    const thin = sweep(settingsFor("diameter"));
    expect(thin.axon!.velocity!).toBeCloseTo(18.5, 0);
    const thick = settingsFor("diameter", { diameter: 952 });
    const r = sweep(thick);
    expect(r.axon!.velocity! / thin.axon!.velocity!).toBeCloseTo(Math.SQRT2, 2);
    expect(r.axon!.velocity!).toBeCloseTo(26, 0);
    expect(passes("diameter", settingsFor("diameter"), thin)).toBe(false);
    expect(passes("diameter", thick, r)).toBe(true);
  });

  it("myelin: a 10 µm fibre goes from about 2.7 to 25 m/s", () => {
    const bare = sweep(settingsFor("myelin"));
    expect(bare.axon!.velocity!).toBeCloseTo(2.7, 0);
    const wrapped = settingsFor("myelin", { myelin: true });
    const r = sweep(wrapped);
    expect(r.axon!.velocity!).toBeGreaterThan(23);
    expect(r.axon!.velocity!).toBeLessThan(28);
    expect(passes("myelin", settingsFor("myelin"), bare)).toBe(false);
    expect(passes("myelin", wrapped, r)).toBe(true);
  });

  it("multiple sclerosis: half the myelin gone still conducts (later); stripped bare, it blocks", () => {
    const healthy = sweep(settingsFor("multiple-sclerosis"));
    const half = sweep(settingsFor("multiple-sclerosis", { myelinLeft: 0.5 }));
    expect(half.axon!.outcome).toBe("conducted");
    expect(half.axon!.t2!).toBeGreaterThan(healthy.axon!.t2!);
    const bare = settingsFor("multiple-sclerosis", { myelinLeft: 0 });
    const r = sweep(bare);
    expect(r.axon!.outcome).toBe("blocked");
    expect(passes("multiple-sclerosis", settingsFor("multiple-sclerosis"), healthy)).toBe(false);
    expect(passes("multiple-sclerosis", bare, r)).toBe(true);
  });

  it("head-on: with both ends stimulated every electrode sees exactly one spike", () => {
    const s = settingsFor("collision");
    const r = sweep(s);
    expect([r.axon!.spikes1, r.axon!.spikes2]).toEqual([1, 1]);
    expect(passes("collision", s, r)).toBe(true);
    expect(passes("collision", settingsFor("collision", { pulses: "single" }), sweep(settingsFor("collision", { pulses: "single" })))).toBe(false);
  });

  it("lidocaine: the setup's short stretch is bridged; one length constant (an eighth of the axon) blocks", () => {
    const setup = settingsFor("lidocaine");
    expect(sweep(setup).axon!.outcome).toBe("conducted");
    const long = settingsFor("lidocaine", { to: 0.45 + 1 / 8 });
    const r = sweep(long);
    expect(r.axon!.outcome).toBe("blocked");
    expect(passes("lidocaine", long, r)).toBe(true);
    expect(passes("lidocaine", settingsFor("lidocaine", { from: 0, to: 1 }), sweep(settingsFor("lidocaine", { from: 0, to: 1 })))).toBe(false); // the whole axon is no answer
  });

  it("the compound action potential: A and C waves both recorded, further apart at 1.5 cm than at 0.5 cm", () => {
    const latencies = (distance: number) => {
      const fibres = nerveComposition(100).map(nerveFibre);
      const states = fibres.map(restState);
      const scratches = fibres.map((f) => scratchFor(f.n));
      const stims = fibres.map((f) => new Float64Array(f.n));
      const pulses = fibres.map((f) => nerveStimulus(f, 6));
      const t: number[] = [];
      const v: number[] = [];
      for (let k = 0; k < Math.round(NERVE_SWEEP_MS / DT); k++) {
        let sum = 0;
        fibres.forEach((f, j) => {
          const on = stimulusAt([pulses[j]], k * DT, stims[j]);
          stepCable(f, states[j], 18.3, on ? stims[j] : null, scratches[j]);
          sum += extracellular(f, states[j].V, on ? stims[j] : null, distance, NERVE_HEIGHT);
        });
        if (k % 10 === 0) {
          t.push((k + 1) * DT);
          v.push(sum);
        }
      }
      return capLatencies(t, v);
    };
    const near = latencies(0.5);
    const far = latencies(1.5);
    for (const l of [near, far]) {
      expect(l.aLatency).toBeDefined();
      expect(l.cLatency).toBeDefined();
    }
    expect(far.cLatency! - far.aLatency!).toBeGreaterThan(2 * (near.cLatency! - near.aLatency!));
    expect(1.5 / far.cLatency! * 10).toBeLessThan(1.2); // C fibres: under ~1 m/s
    const s = settingsFor("compound", { distance: 1.5 });
    expect(passes("compound", s, { view: "nerve", axon: null, nerve: { t: [], v: [], ...far } })).toBe(true);
    expect(passes("compound", settingsFor("compound"), { view: "nerve", axon: null, nerve: { t: [], v: [], ...near } })).toBe(false);
  }, 120_000);
});
