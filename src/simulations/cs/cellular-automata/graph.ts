const AXIS_STEPS = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];

/**
 * A round maximum a little above `value` for the population graph's vertical axis (at least 4), so the scale
 * stays put instead of jittering with every update: 3 → 4, 130 → 150, 13 218 → 15 000.
 */
export function niceMax(value: number): number {
  const target = Math.max(4, value * 1.1);
  const base = 10 ** Math.floor(Math.log10(target));
  return (AXIS_STEPS.find((f) => f * base >= target) ?? 10) * base;
}
