import type { Challenge } from "@/lib/challenges";
import { isConway } from "./settings";
import type { CASettings } from "./settings";
import type { CellularAutomataStats } from "./sim";

/** The pattern the third challenge is about (it must exist in the pattern library). */
export const GUN_PATTERN = "Glider Gun (Gosper)";

/**
 * Guided experiments for the Game of Life. Every claim in the texts below is checked against the CPU twin of the
 * shaders in challenges.test.ts / tracker.test.ts: a 2×2 square never changes, a row of four settles into a
 * six-cell beehive within two generations, a T-shape ends as the twelve-cell "traffic light", three in a row
 * is the blinker, and the Gosper gun's population keeps climbing (it first tops 100 cells around generation 250).
 */
export const CA_CHALLENGES: Challenge<CASettings, CellularAutomataStats>[] = [
  {
    id: "still-life",
    title: "Build something that never changes",
    prompt: "On an empty grid, draw a few live cells and press Play. Can you make a pattern that survives and then never changes?",
    setup: { birth: "3", survive: "23", pattern: "clear", gridSize: 256, speed: 20 },
    prediction: {
      question: "Which four-cell shape is already perfectly stable — it never changes at all?",
      options: [
        { label: "A straight line of four" },
        { label: "A 2×2 square", correct: true },
        { label: "A T-shape (three in a row with one below the middle)" },
      ],
    },
    goal: {
      description: "Press Play and end up with a still life: at least 4 live cells that stay exactly the same, generation after generation.",
      check: ({ params, readouts }) => isConway(params) && readouts.status.kind === "still" && readouts.population >= 4,
    },
    hint: "The Draw tool is ready: click four cells into a 2×2 square. Then try other shapes, like a row of four, and see what they turn into.",
    explanation:
      "In a 2×2 square every cell has exactly three live neighbours, so all four survive, and no empty cell has three live neighbours, so none are born: nothing ever changes. A row of four cells isn’t stable at first, but within two generations it settles into a six-cell shape called a beehive, which is also a still life. The T-shape never freezes: it keeps changing for a while and ends up as a twelve-cell oscillator called the traffic light.",
  },
  {
    id: "oscillator",
    title: "Make it blink",
    prompt: "Some patterns don’t stay still — they repeat. Draw three cells in a row, press Play, and watch the status line.",
    setup: { birth: "3", survive: "23", pattern: "clear", gridSize: 256, speed: 6 },
    prediction: {
      question: "Three live cells in a horizontal row, left alone under Conway’s rules, will…",
      options: [
        { label: "all die out after one step" },
        { label: "stay exactly as they are" },
        { label: "flip between a horizontal and a vertical row, forever", correct: true },
      ],
    },
    goal: {
      description: "Press Play and end up with an oscillator: a pattern of at least 3 cells that repeats itself, over and over.",
      check: ({ params, readouts }) => isConway(params) && readouts.status.kind === "oscillator" && readouts.population >= 3,
    },
    hint: "Draw three cells side by side. After a few generations the status line says ‘Oscillator, period 2’.",
    explanation:
      "The two end cells each have only one neighbour and die; the middle cell has two and survives; and the cells directly above and below the middle each have exactly three live neighbours, so they are born. The result is a vertical row of three — and the same rule turns it back into a horizontal one. This ‘blinker’ repeats every 2 generations, so its period is 2.",
  },
  {
    id: "gun",
    title: "Grow without limit",
    prompt: "A pattern of 36 cells is loaded. Press Play and watch the population graph.",
    setup: { birth: "3", survive: "23", pattern: GUN_PATTERN, gridSize: 256, speed: 40 },
    prediction: {
      question: "Starting from 36 cells, over the next few hundred generations the population of this pattern will…",
      options: [
        { label: "settle at a fixed number of cells" },
        { label: "die out" },
        { label: "keep growing", correct: true },
      ],
    },
    goal: {
      description: "Press Play and let the glider gun’s population climb past 100 live cells (with Conway’s rules).",
      // Tied to the gun: any dense random soup is far above 100 cells from the start, which would prove nothing.
      check: ({ params, readouts }) => isConway(params) && params.pattern === GUN_PATTERN && readouts.population > 100,
    },
    hint: "Raise the speed if it is slow. The gun fires a new glider every 30 generations, and every glider adds 5 cells.",
    explanation:
      "This is the Gosper glider gun. It is a finite pattern, yet it fires a new 5-cell glider every 30 generations, and each glider flies away for ever — so the population keeps growing. When Bill Gosper’s team found it in 1970 it settled a question Conway had put a prize on: can a finite pattern grow without limit? (This grid wraps around, so after about a thousand generations the gliders arrive back at the gun and the growth stops. On an endless grid it never would.)",
  },
];
