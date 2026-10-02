import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const css = fs.readFileSync(path.join(process.cwd(), "src/app/globals.css"), "utf8");

function token(name: string): string {
  const match = css.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (!match) throw new Error(`token --${name} not found as a #rrggbb value in globals.css`);
  return match[1];
}

/** WCAG relative luminance and contrast ratio. */
function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe("design tokens", () => {
  const bg = token("bg");

  it("all text colours meet WCAG AA (4.5:1) on the page background", () => {
    for (const name of ["ink", "mut", "dim", "live", "acc-physics", "acc-chemistry", "acc-cs", "acc-biology"]) {
      expect(contrast(token(name), bg), `--${name}`).toBeGreaterThanOrEqual(4.5);
    }
  });

  it("has a visible hierarchy: ink > mut > dim", () => {
    expect(contrast(token("ink"), bg)).toBeGreaterThan(contrast(token("mut"), bg));
    expect(contrast(token("mut"), bg)).toBeGreaterThan(contrast(token("dim"), bg));
  });
});
