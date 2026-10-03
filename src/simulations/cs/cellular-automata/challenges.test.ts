import { describe, expect, it } from "vitest";
import { CA_CHALLENGES, GUN_PATTERN } from "./challenges";
import { countAlive, stateHash, stepLife } from "./life";
import { patterns, rasterizePattern } from "./patterns";
import type { Pattern } from "./patterns";
import { parseRule } from "./rules";
import { CA_DEFAULTS } from "./settings";
import type { CASettings } from "./settings";
import type { CellularAutomataStats } from "./sim";
import { PopulationTracker } from "./tracker";

const BLOCK = [[0, 0], [1, 0], [0, 1], [1, 1]] satisfies [number, number][];
const ROW_OF_FOUR = [[0, 0], [1, 0], [2, 0], [3, 0]] satisfies [number, number][];
const T_SHAPE = [[0, 0], [1, 0], [2, 0], [1, 1]] satisfies [number, number][];
const BLINKER = [[0, 0], [1, 0], [2, 0]] satisfies [number, number][];
const GLIDER = [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]] satisfies [number, number][];

const byId = (id: string) => CA_CHALLENGES.find((c) => c.id === id)!;
const drawn = (points: [number, number][], size: number) => rasterizePattern({ name: "drawn", points } satisfies Pattern, size, size);
const library = (name: string, size: number) => rasterizePattern(patterns.find((p) => p.name === name)!, size, size);
const empty = (size: number) => new Uint32Array(size * size);

/** What the controls show after each generation of `start`, measured like the GPU pipeline measures it. */
function* history(start: Uint32Array, size: number, generations: number, rules: Pick<CASettings, "birth" | "survive"> = CA_DEFAULTS) {
  const birth = parseRule(rules.birth);
  const survive = parseRule(rules.survive);
  const tracker = new PopulationTracker(600);
  let cells: Uint32Array = start;
  for (let g = 0; g <= generations; g++) {
    if (g > 0) cells = stepLife(cells, size, size, birth, survive);
    tracker.push(g, countAlive(cells), stateHash(cells));
    const from = Math.max(0, tracker.counts.length - 240);
    const stats: CellularAutomataStats = {
      generation: g,
      population: tracker.counts[tracker.counts.length - 1],
      cells: size * size,
      gens: tracker.gens.slice(from),
      counts: tracker.counts.slice(from),
      status: tracker.status(),
    };
    yield stats;
  }
}

const last = (stats: Iterable<CellularAutomataStats>) => [...stats].at(-1)!;
const state = (settings: CASettings, readouts: CellularAutomataStats) => ({ params: settings, readouts });
const solved = (id: string, settings: CASettings, readouts: CellularAutomataStats) => byId(id).goal.check(state(settings, readouts));

/** First generation at which the challenge's goal holds, or undefined. */
function firstSolved(id: string, settings: CASettings, start: Uint32Array, size: number, generations: number): number | undefined {
  for (const stats of history(start, size, generations, settings)) if (solved(id, settings, stats)) return stats.generation;
  return undefined;
}

const SMALL = 48; // plenty of room for the small patterns, and fast

describe("cellular-automata challenges", () => {
  it("have unique ids and exactly one correct prediction each", () => {
    expect(new Set(CA_CHALLENGES.map((c) => c.id)).size).toBe(CA_CHALLENGES.length);
    for (const c of CA_CHALLENGES) expect(c.prediction?.options.filter((o) => o.correct), c.id).toHaveLength(1);
  });

  it("only use patterns that exist, and set up Conway's rules", () => {
    expect(patterns.map((p) => p.name)).toContain(GUN_PATTERN);
    for (const c of CA_CHALLENGES) {
      expect(c.setup.birth, c.id).toBe("3");
      expect(c.setup.survive, c.id).toBe("23");
      const pattern = c.setup.pattern;
      expect(pattern === "clear" || patterns.some((p) => p.name === pattern), `${c.id}: ${pattern}`).toBe(true);
    }
  });

  it("are not already solved by their own setup", () => {
    for (const c of CA_CHALLENGES) {
      const settings: CASettings = { ...CA_DEFAULTS, ...c.setup };
      const start = settings.pattern === "clear" ? empty(SMALL) : library(settings.pattern, 256);
      const size = settings.pattern === "clear" ? SMALL : 256;
      expect(solved(c.id, settings, last(history(start, size, 0))), c.id).toBe(false);
    }
  });

  it("are solved by the intended pattern", () => {
    const settings = (id: string): CASettings => ({ ...CA_DEFAULTS, ...byId(id).setup });
    expect(firstSolved("still-life", settings("still-life"), drawn(BLOCK, SMALL), SMALL, 20)).toBeLessThanOrEqual(8);
    expect(firstSolved("oscillator", settings("oscillator"), drawn(BLINKER, SMALL), SMALL, 20)).toBeLessThanOrEqual(10);
    // The gun: a few hundred generations on the challenge's own 256x256 grid.
    const gun = firstSolved("gun", settings("gun"), library(GUN_PATTERN, 256), 256, 400);
    expect(gun, "gun never reached the goal").toBeDefined();
    expect(gun!).toBeGreaterThan(150); // takes a real run, not a glance
    expect(gun!).toBeLessThan(320); // ...but not so long that a student gives up (about 8 s at the challenge's speed)
  });

  it("also accept other correct answers", () => {
    const still = { ...CA_DEFAULTS, ...byId("still-life").setup };
    const osc = { ...CA_DEFAULTS, ...byId("oscillator").setup };
    // A row of four really does settle into a (six-cell) still life, as the explanation says...
    const row = last(history(drawn(ROW_OF_FOUR, SMALL), SMALL, 12));
    expect(row.status).toEqual({ kind: "still" });
    expect(row.population).toBe(6);
    expect(solved("still-life", still, row)).toBe(true);
    // ...and the T-shape ends as the twelve-cell oscillator the explanation calls the traffic light.
    const t = last(history(drawn(T_SHAPE, SMALL), SMALL, 24));
    expect(t.status).toEqual({ kind: "oscillator", period: 2 });
    expect(t.population).toBe(12);
    expect(solved("oscillator", osc, t)).toBe(true);
    expect(solved("still-life", still, t)).toBe(false);
    // A library oscillator is just as valid as a hand-drawn one.
    expect(solved("oscillator", osc, last(history(library("Pulsar", SMALL), SMALL, 30)))).toBe(true);
  });

  it("cannot be solved by a shortcut that dodges the lesson", () => {
    const still = { ...CA_DEFAULTS, ...byId("still-life").setup };
    const osc = { ...CA_DEFAULTS, ...byId("oscillator").setup };
    const gunSettings = { ...CA_DEFAULTS, ...byId("gun").setup };
    const blockStats = last(history(drawn(BLOCK, SMALL), SMALL, 20));
    const blinkerStats = last(history(drawn(BLINKER, SMALL), SMALL, 20));
    const gliderStats = last(history(drawn(GLIDER, SMALL), SMALL, 40));
    const emptyStats = last(history(empty(SMALL), SMALL, 20));

    // Each goal wants its own kind of pattern, nothing else.
    expect(solved("still-life", still, blinkerStats)).toBe(false);
    expect(solved("still-life", still, gliderStats)).toBe(false);
    expect(solved("still-life", still, emptyStats)).toBe(false);
    expect(solved("oscillator", osc, blockStats)).toBe(false);
    expect(solved("oscillator", osc, gliderStats)).toBe(false);
    expect(solved("oscillator", osc, emptyStats)).toBe(false);

    // Changing the rules doesn't count, even though a block and a blinker behave the same under HighLife.
    const highLife = { birth: "36", survive: "23" };
    expect(solved("still-life", { ...still, ...highLife }, blockStats)).toBe(false);
    expect(solved("oscillator", { ...osc, ...highLife }, blinkerStats)).toBe(false);
    expect(solved("still-life", { ...still, birth: "3", survive: "32" }, blockStats)).toBe(true); // typed backwards: still Conway

    // A random soup has thousands of cells from generation 0, so the gun goal must name the gun.
    const soup = { ...gunSettings, pattern: "random" };
    const dense: CellularAutomataStats = { ...blockStats, population: 13000 };
    expect(solved("gun", soup, dense)).toBe(false);
    expect(solved("gun", { ...gunSettings, ...highLife }, dense)).toBe(false);
    expect(solved("gun", gunSettings, dense)).toBe(true);
  });

  it("not too early: nothing is claimed before the history can show it", () => {
    const still = { ...CA_DEFAULTS, ...byId("still-life").setup };
    for (const stats of history(drawn(BLOCK, SMALL), SMALL, 4)) expect(solved("still-life", still, stats), `generation ${stats.generation}`).toBe(false);
  });
});

describe("what the challenge texts claim", () => {
  const phases = (generations: number) => [...history(library(GUN_PATTERN, 256), 256, generations)];

  it("a 2×2 square never changes; a row of four settles into a six-cell beehive within two generations", () => {
    const block = [...history(drawn(BLOCK, SMALL), SMALL, 10)];
    expect(new Set(block.map((s) => s.population))).toEqual(new Set([4]));
    const row = [...history(drawn(ROW_OF_FOUR, SMALL), SMALL, 10)].map((s) => s.population);
    expect(row.slice(0, 3)).toEqual([4, 6, 6]);
    expect(new Set(row.slice(2))).toEqual(new Set([6]));
  });

  it("three in a row flips between horizontal and vertical with period 2", () => {
    const grids: Uint32Array[] = [drawn(BLINKER, SMALL)];
    for (let g = 1; g <= 4; g++) grids.push(stepLife(grids[g - 1], SMALL, SMALL, parseRule("3"), parseRule("23")));
    expect(stateHash(grids[0])).not.toBe(stateHash(grids[1]));
    expect(stateHash(grids[0])).toBe(stateHash(grids[2]));
    expect(stateHash(grids[1])).toBe(stateHash(grids[3]));
    expect(new Set(grids.map((g) => countAlive(g)))).toEqual(new Set([3]));
  });

  it("the gun has 36 cells and every 30 generations its population is 5 higher (one more glider)", () => {
    expect(library(GUN_PATTERN, 256).reduce((a, b) => a + b, 0)).toBe(36);
    const pops = phases(330).map((s) => s.population);
    for (let g = 60; g <= 300; g += 7) expect(pops[g + 30] - pops[g], `generation ${g}`).toBe(5);
  });
});
