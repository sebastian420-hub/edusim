import { describe, expect, it } from "vitest";
import {
  BARE,
  bareFibre,
  blockSodium,
  conductionVelocity,
  demyelinate,
  DT,
  E_L,
  HH_TEMPERATURE,
  HH_VELOCITY,
  INTERNODE_PER_DIAMETER,
  kick,
  lengthConstant,
  myelinatedFibre,
  NODE_LENGTH,
  restState,
  runFibre,
  scratchFor,
  SQUID_DIAMETER,
  stepCable,
} from "./cable";
import type { Fibre } from "./cable";

const LAMBDA = lengthConstant(SQUID_DIAMETER);
/** The squid axon of Hodgkin & Huxley, 8 length constants long (8.5 cm), 0.2 mm compartments. */
const squid = (n = 400) => bareFibre(SQUID_DIAMETER, 8 * LAMBDA, n);
const myelinated = (d: number, internodes = 30) => myelinatedFibre(d, internodes * INTERNODE_PER_DIAMETER * d * 1e-4);
const fired = (r: { crossings: Uint16Array }, i: number) => r.crossings[i] > 0;

describe("cable numerics", () => {
  it("the resting axon stays at rest", () => {
    const f = squid(100);
    const r = runFibre(f, HH_TEMPERATURE, [], 10);
    for (const v of r.state.V) expect(Math.abs(v + 65)).toBeLessThan(0.05);
  });

  it("a passive cable matches the analytic cosh profile, V − E ∝ cosh((L − x)/λ)", () => {
    // Channels off: only leak. Steady current into the first compartment of a sealed cable 2λ long.
    const f = bareFibre(SQUID_DIAMETER, 2 * LAMBDA, 400);
    f.gNa.fill(0);
    f.gK.fill(0);
    const s = restState(f);
    s.V.fill(E_L);
    const stim = new Float64Array(f.n);
    stim[0] = 50;
    const scratch = scratchFor(f.n);
    for (let k = 0; k < 4000; k++) stepCable(f, s, HH_TEMPERATURE, stim, scratch, 0.05);
    const L = 2 * LAMBDA;
    const at = (x: number) => Math.cosh((L - x) / LAMBDA);
    const ref = (s.V[200] - E_L) / at(f.x[200]);
    for (const i of [20, 100, 300, 399]) expect((s.V[i] - E_L) / at(f.x[i]) / ref).toBeCloseTo(1, 2);
  });

  it("the implicit solve stays stable far beyond the explicit limit (Δt = 0.05 ms vs 0.004 ms)", () => {
    const f = squid();
    const D = (1000 * (SQUID_DIAMETER * 1e-4)) / 2 / (2 * 35.4) / BARE.cm; // cm²/ms
    const dx = f.length[0];
    expect((dx * dx) / (2 * D)).toBeLessThan(0.005); // what an explicit step would need
    const r = runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2)], 8, 0.05);
    expect(r.state.V.every(Number.isFinite)).toBe(true);
    expect(fired(r, f.n - 10)).toBe(true);
  });
});

describe("the squid giant axon (Hodgkin & Huxley 1952)", () => {
  it("conducts at their computed 18.8 m/s: within 2 % once the grid is fine, and the default grid within 1 % of that", () => {
    const fine = squid(1600);
    const vFine = conductionVelocity(fine, runFibre(fine, HH_TEMPERATURE, [kick(fine)], 5, 0.0025).arrival)!;
    expect(Math.abs(vFine / HH_VELOCITY - 1)).toBeLessThan(0.02);
    const f = squid();
    const v = conductionVelocity(f, runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2)], 5, DT).arrival)!;
    expect(Math.abs(v / vFine - 1)).toBeLessThan(0.01);
  });

  it("all-or-none: a stimulus twice as strong makes the same spike", () => {
    const peaks = [1.5, 4].map((k) => {
      const f = squid();
      return runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, k)], 6).peak[300];
    });
    expect(Math.abs(peaks[0] - peaks[1])).toBeLessThan(0.5);
    const f = squid();
    expect(fired(runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 0.8)], 6), 300)).toBe(false); // below threshold: nothing travels
  });

  it("two spikes that meet head-on annihilate instead of passing through", () => {
    const f = squid();
    const r = runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2), kick(f, 0.5, 2, f.n - 30)], 10);
    expect(Math.max(...r.crossings)).toBe(1); // every point fires exactly once
    expect(r.crossings[200]).toBe(1);
  });

  it("refractory period: a second stimulus 2 ms after the first does not travel; 6 ms after, it does", () => {
    for (const [gap, travels] of [[2, false], [6, true]] as const) {
      const f = squid();
      const r = runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2), kick(f, 0.5 + gap, 2)], 16);
      expect(r.crossings[f.n - 20], `gap ${gap} ms`).toBe(travels ? 2 : 1);
    }
  });

  it("warmer is faster, until heat block (around 30–35 °C for squid channels)", () => {
    const v = (T: number) => {
      const f = squid();
      return conductionVelocity(f, runFibre(f, T, [kick(f, 0.5, 2)], 25).arrival);
    };
    const cold = v(6.3)!;
    const warm = v(HH_TEMPERATURE)!;
    expect(warm).toBeGreaterThan(1.3 * cold);
    expect(v(35)).toBeUndefined();
  });
});

describe("geometry and myelin", () => {
  it("bare axons: speed grows as √diameter (doubling the diameter: ×1.41, not ×2)", () => {
    const v = (d: number) => {
      const f = bareFibre(d, 8 * lengthConstant(d), 400);
      return conductionVelocity(f, runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2)], 1 + (8 * lengthConstant(d)) / (0.0188 * Math.sqrt(d / 476)) * 1.3).arrival)!;
    };
    expect(v(2) / v(1)).toBeCloseTo(Math.SQRT2, 2);
    expect(v(952) / v(476)).toBeCloseTo(Math.SQRT2, 2);
    expect(v(1)).toBeCloseTo(0.85, 1); // a 1 µm bare fibre (C fibre size) crawls at under 1 m/s
  });

  it("myelinated fibres: speed proportional to diameter (Rushton), about 2.5 m/s per µm at 18 °C", () => {
    const v = (d: number) => {
      const f = myelinated(d);
      return conductionVelocity(f, runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2)], 10).arrival)!;
    };
    const v4 = v(4);
    expect(v(8) / v4).toBeCloseTo(2, 1);
    expect(v(16) / v4).toBeCloseTo(4, 1);
    expect(v4 / 4).toBeCloseTo(2.54, 1);
  });

  it("at 10 µm, myelin makes the fibre about 9× faster than a bare one of the same size", () => {
    const m = myelinated(10);
    const vm = conductionVelocity(m, runFibre(m, HH_TEMPERATURE, [kick(m, 0.5, 2)], 10).arrival)!;
    const b = bareFibre(10, 8 * lengthConstant(10), 400);
    const vb = conductionVelocity(b, runFibre(b, HH_TEMPERATURE, [kick(b, 0.5, 2)], 25).arrival)!;
    expect(vm / vb).toBeGreaterThan(8);
    expect(vm / vb).toBeLessThan(11);
  });

  it("saltatory conduction: the spike is regenerated node by node, at a steady time per internode", () => {
    const f = myelinated(10);
    const r = runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2)], 10);
    const nodes = [...f.active].flatMap((a, i) => (a ? [i] : [])).slice(8, 24);
    const hops = nodes.slice(1).map((n, k) => r.arrival[n] - r.arrival[nodes[k]]);
    const mean = hops.reduce((a, b) => a + b, 0) / hops.length;
    for (const h of hops) expect(Math.abs(h / mean - 1)).toBeLessThan(0.05);
    // Inside an internode, the voltage only rises passively: the peak sags between nodes.
    const mid = nodes[4] + 5;
    expect(r.peak[mid]).toBeLessThan(r.peak[nodes[4]]);
  });
});

describe("drugs and disease", () => {
  const blockAt = (len: number, fraction: number) => {
    const f = squid();
    const x0 = 3 * LAMBDA;
    blockSodium(f, x0, x0 + len * LAMBDA, fraction);
    return fired(runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2)], 12), f.n - 10);
  };

  it("TTX (all sodium channels blocked): a stretch a quarter of a length constant long is bridged; half of one blocks", () => {
    expect(blockAt(0.25, 1)).toBe(true);
    expect(blockAt(0.5, 1)).toBe(false);
  });

  it("lidocaine (90 % blocked): half a length constant is bridged; one length constant blocks", () => {
    expect(blockAt(0.5, 0.9)).toBe(true);
    expect(blockAt(1, 0.9)).toBe(false);
  });

  const demyelinated = (remaining: number, internodes: number) => {
    const f = myelinated(10);
    const span = INTERNODE_PER_DIAMETER * 10e-4 + NODE_LENGTH;
    demyelinate(f, 15 * span, (15 + internodes) * span - 1e-5, remaining);
    return { f, r: runFibre(f, HH_TEMPERATURE, [kick(f, 0.5, 2)], 10) };
  };

  it("demyelination slows conduction, then blocks it: one stripped internode is enough", () => {
    const healthy = demyelinated(1, 3);
    const thin = demyelinated(0.1, 3);
    const end = (x: { f: Fibre; r: { arrival: Float64Array } }) => x.r.arrival[x.f.n - 1];
    expect(end(thin)).toBeGreaterThan(end(healthy) + 0.3); // arrives later
    expect(fired(demyelinated(0, 1).r, healthy.f.n - 1)).toBe(false);
  });
});
