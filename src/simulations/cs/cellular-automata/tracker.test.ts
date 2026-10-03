import { describe, expect, it } from "vitest";
import { countAlive, stateHash, stepLife } from "./life";
import { patterns, rasterizePattern } from "./patterns";
import { analyze, PopulationTracker } from "./tracker";
import { parseRule } from "./rules";

const B3 = parseRule("3");
const S23 = parseRule("23");

/** Runs the named pattern on a small torus and records per-generation stats, like the GPU pipeline does. */
function run(name: string | null, generations: number, size = 64) {
  const pattern = patterns.find((p) => p.name === name);
  let cells = pattern ? rasterizePattern(pattern, size, size) : new Uint32Array(size * size);
  const tracker = new PopulationTracker(2000);
  tracker.push(0, countAlive(cells), stateHash(cells));
  for (let g = 1; g <= generations; g++) {
    cells = stepLife(cells, size, size, B3, S23);
    tracker.push(g, countAlive(cells), stateHash(cells));
  }
  return tracker;
}

describe("pattern classification (CPU twin of the GPU pipeline)", () => {
  it("empty grid is extinct", () => {
    expect(run(null, 10).status()).toEqual({ kind: "empty" });
  });

  it("block is a still life", () => {
    expect(run("Block", 12).status()).toEqual({ kind: "still" });
  });

  it("blinker is an oscillator of period 2", () => {
    expect(run("Blinker", 12).status()).toEqual({ kind: "oscillator", period: 2 });
  });

  it("toad is an oscillator of period 2", () => {
    expect(run("Toad", 14).status()).toEqual({ kind: "oscillator", period: 2 });
  });

  it("pulsar is an oscillator of period 3", () => {
    expect(run("Pulsar", 20).status()).toEqual({ kind: "oscillator", period: 3 });
  });

  it("glider is a moving pattern (population constant, pattern travels)", () => {
    expect(run("Glider", 30).status().kind).toBe("moving");
  });

  it("the glider gun keeps evolving and its population grows without bound", () => {
    const gun = run("Glider Gun (Gosper)", 330, 128);
    expect(gun.status()).toEqual({ kind: "evolving" });
    const counts = gun.counts;
    expect(counts[counts.length - 1]).toBeGreaterThan(90);
    expect(counts[counts.length - 1]).toBeGreaterThan(counts[60]);
  });

  it("needs enough history before declaring a period", () => {
    expect(analyze([4, 4, 4], [1, 1, 1])).toEqual({ kind: "evolving" });
  });
});

describe("PopulationTracker", () => {
  it("keeps a rolling window", () => {
    const t = new PopulationTracker(5);
    for (let g = 0; g < 9; g++) t.push(g, g, g);
    expect(t.gens).toEqual([4, 5, 6, 7, 8]);
  });

  it("starts over when generations jump (a reset or an edit)", () => {
    const t = new PopulationTracker();
    t.push(0, 5, 1);
    t.push(1, 5, 1);
    t.push(7, 9, 2);
    expect(t.gens).toEqual([7]);
  });
});

describe("stateHash", () => {
  it("is equal for equal grids and different when a cell moves", () => {
    const a = new Uint32Array(16);
    a[3] = 1;
    const b = new Uint32Array(16);
    b[3] = 1;
    const c = new Uint32Array(16);
    c[4] = 1;
    expect(stateHash(a)).toBe(stateHash(b));
    expect(stateHash(a)).not.toBe(stateHash(c));
  });
});
