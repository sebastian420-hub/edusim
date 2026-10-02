import { describe, expect, it } from "vitest";
import { FramePacer, QualityGovernor } from "./pacing";

/** A promise we resolve by hand, to control when "GPU work" completes. */
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => (resolve = r));
  return { promise, resolve };
}

describe("FramePacer", () => {
  it("allows up to maxInFlight frames, then skips until one completes", async () => {
    const pacer = new FramePacer({ maxInFlight: 2 });
    const a = deferred();
    const b = deferred();
    expect(pacer.shouldRender()).toBe(true);
    pacer.track(a.promise, () => {});
    expect(pacer.shouldRender()).toBe(true);
    pacer.track(b.promise, () => {});
    expect(pacer.shouldRender()).toBe(false);
    expect(pacer.shouldRender()).toBe(false);
    expect(pacer.skipped).toBe(2);

    a.resolve();
    await a.promise;
    await Promise.resolve();
    expect(pacer.pending).toBe(1);
    expect(pacer.shouldRender()).toBe(true);
  });

  it("estimates per-frame GPU cost from completion time and queue depth", async () => {
    let t = 0;
    const pacer = new FramePacer({ now: () => t });
    const first = deferred();
    const second = deferred();
    const costs: number[] = [];

    pacer.track(first.promise, (c) => costs.push(c)); // submitted at t=0, alone in the queue
    t = 10;
    pacer.track(second.promise, (c) => costs.push(c)); // submitted at t=10 behind the first
    t = 40;
    first.resolve();
    await first.promise;
    await Promise.resolve();
    t = 70;
    second.resolve();
    await second.promise;
    await Promise.resolve();

    expect(costs[0]).toBeCloseTo(40, 5); // 40 ms wait / 1 frame ahead
    expect(costs[1]).toBeCloseTo((70 - 10) / 2, 5); // 60 ms wait shared with the frame before it
  });

  it("frees the slot even if the completion promise rejects", async () => {
    const pacer = new FramePacer({ maxInFlight: 1 });
    pacer.track(Promise.reject(new Error("device lost")), () => {});
    await new Promise((r) => setTimeout(r, 0));
    expect(pacer.shouldRender()).toBe(true);
  });
});

/** Drives the governor with a cost model: GPU time proportional to pixels (scale squared). */
function simulate(governor: QualityGovernor, fullResCostMs: number, frames: number, spike?: (frame: number) => number) {
  const history: number[] = [];
  for (let i = 0; i < frames; i++) {
    const cost = fullResCostMs * governor.scale ** 2 + (spike?.(i) ?? 0);
    governor.record(cost);
    history.push(governor.level);
  }
  return history;
}

describe("QualityGovernor", () => {
  it("stays at full quality when frames are cheap", () => {
    const g = new QualityGovernor();
    simulate(g, 6, 600);
    expect(g.level).toBe(0);
  });

  it("steps down when frames are persistently over budget, until they fit", () => {
    const g = new QualityGovernor();
    simulate(g, 80, 600); // 80 ms at full res; budget 25 ms -> needs scale <= ~0.55
    expect(g.scale).toBeLessThanOrEqual(0.5);
    expect(80 * g.scale ** 2).toBeLessThanOrEqual(25);
  });

  it("bottoms out at the lowest level instead of going further", () => {
    const g = new QualityGovernor();
    simulate(g, 5000, 2000);
    expect(g.level).toBe(3);
    expect(g.scale).toBe(0.35);
  });

  it("ignores warm-up frames and isolated spikes", () => {
    const g = new QualityGovernor();
    for (let i = 0; i < 20; i++) g.record(900); // shader compilation during warm-up
    simulate(g, 6, 300, (i) => (i % 50 === 0 ? 400 : 0)); // a one-frame hitch every 50 frames
    expect(g.level).toBe(0);
  });

  it("steps back up when the load disappears, one level at a time", () => {
    const g = new QualityGovernor();
    simulate(g, 80, 400);
    const degraded = g.level;
    expect(degraded).toBeGreaterThan(0);
    simulate(g, 3, 2000); // now trivially cheap at any resolution
    expect(g.level).toBe(0);
  });

  it("does not oscillate when full quality only just misses the budget", () => {
    const g = new QualityGovernor();
    const history = simulate(g, 32, 3000); // 32 ms at full res: over budget, fits at 0.75 (18 ms)
    const changes = history.filter((v, i) => i > 0 && v !== history[i - 1]).length;
    expect(g.level).toBe(1);
    expect(changes).toBeLessThanOrEqual(1); // settles and stays: predicted cost at full res stays over the headroom
  });
});
