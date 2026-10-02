/** Life-like cellular automaton rules, written as "B3/S23" style digit strings. */

/** Converts a digit string such as "23" into a bitmask where bit n means "n neighbours". */
export function parseRule(digits: string): number {
  let mask = 0;
  for (const ch of digits) {
    const d = ch.charCodeAt(0) - 48;
    if (d >= 0 && d <= 8) mask |= 1 << d;
  }
  return mask;
}

/** Inverse of `parseRule`: the neighbour counts set in `mask`, ascending. */
export function formatRule(mask: number): string {
  let out = "";
  for (let d = 0; d <= 8; d++) if (mask & (1 << d)) out += d;
  return out;
}

export interface RulePreset {
  name: string;
  birth: string;
  survive: string;
}

export const RULE_PRESETS: RulePreset[] = [
  { name: "Conway's Life", birth: "3", survive: "23" },
  { name: "HighLife", birth: "36", survive: "23" },
  { name: "Day & Night", birth: "3678", survive: "34678" },
  { name: "Seeds", birth: "2", survive: "" },
  { name: "Maze", birth: "3", survive: "12345" },
];
