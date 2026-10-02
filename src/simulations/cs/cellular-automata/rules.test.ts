import { describe, expect, it } from "vitest";
import { formatRule, parseRule, RULE_PRESETS } from "./rules";

describe("parseRule", () => {
  it("sets one bit per neighbour count", () => {
    expect(parseRule("3")).toBe(0b000001000);
    expect(parseRule("23")).toBe(0b000001100);
    expect(parseRule("012345678")).toBe(0b111111111);
  });

  it("ignores non-digits, 9 and duplicates", () => {
    expect(parseRule("")).toBe(0);
    expect(parseRule("a9-3x3")).toBe(0b000001000);
  });

  it("round-trips through formatRule", () => {
    for (const p of RULE_PRESETS) {
      expect(formatRule(parseRule(p.birth))).toBe(p.birth);
      expect(formatRule(parseRule(p.survive))).toBe(p.survive);
    }
  });
});
