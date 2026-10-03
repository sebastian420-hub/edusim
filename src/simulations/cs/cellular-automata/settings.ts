import { boolField, enumField, numberField } from "@/lib/urlState";
import type { FieldCodec, Schema } from "@/lib/urlState";
import { DEFAULT_PATTERN, patterns } from "./patterns";
import { GRID_SIZES } from "./sim-constants";

/** Everything a shared link reproduces: the rules, grid, speed, look and the starting pattern. */
export interface CASettings {
  /** Birth rule digits: how many live neighbours bring a dead cell to life ("3"). */
  birth: string;
  /** Survival rule digits: how many live neighbours keep a live cell alive ("23"). */
  survive: string;
  /** Generations per second. */
  speed: number;
  /** 0 = green, 1 = neon, 2 = white, 3 = amber. */
  theme: 0 | 1 | 2 | 3;
  gridSize: (typeof GRID_SIZES)[number];
  /** The starting pattern: a library pattern, "random" (a fresh random soup) or "clear" (an empty grid). */
  pattern: string;
  /** Show the population graph. */
  showGraph: boolean;
}

export const THEMES = [0, 1, 2, 3] as const;
export const PATTERN_CHOICES = [...patterns.map((p) => p.name), "random", "clear"];

export const CA_DEFAULTS: CASettings = {
  birth: "3",
  survive: "23",
  speed: 10,
  theme: 0,
  gridSize: 256,
  pattern: DEFAULT_PATTERN,
  showGraph: true,
};

/** Keeps only the digits 0-8 (a cell has at most 8 neighbours), without repeats, in ascending order. */
export function sanitizeRuleDigits(raw: string): string {
  return [...new Set(raw.replace(/[^0-8]/g, ""))].sort().join("");
}

/** True for Conway's Life (B3/S23), however the digits were typed. */
export const isConway = (s: Pick<CASettings, "birth" | "survive">) =>
  sanitizeRuleDigits(s.birth) === "3" && sanitizeRuleDigits(s.survive) === "23";

/**
 * A rule is a set of neighbour counts. An empty value is a valid rule ("never born" / "never survives"), but a
 * non-empty value without a single usable digit is garbage and falls back to the default.
 */
const ruleField: FieldCodec<string> = {
  decode(raw) {
    const digits = sanitizeRuleDigits(raw);
    return digits === "" && raw !== "" ? undefined : digits;
  },
  encode: sanitizeRuleDigits,
};

/** A value that must be one of `values` (numbers travel through the URL as text). */
const oneOf = <T extends number>(values: readonly T[]): FieldCodec<T> => ({
  decode: (raw) => values.find((v) => String(v) === raw),
  encode: String,
});

export const CA_SCHEMA: Schema<CASettings> = {
  birth: ruleField,
  survive: ruleField,
  speed: numberField(1, 60),
  theme: oneOf(THEMES),
  gridSize: oneOf(GRID_SIZES),
  pattern: enumField(PATTERN_CHOICES),
  showGraph: boolField,
};
