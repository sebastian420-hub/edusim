import { describe, expect, it } from "vitest";
import { decodeParams, encodeParams } from "@/lib/urlState";
import { DEFAULT_PATTERN, patterns } from "./patterns";
import { CA_DEFAULTS, CA_SCHEMA, isConway, PATTERN_CHOICES, sanitizeRuleDigits } from "./settings";
import type { CASettings } from "./settings";

const decode = (search: string) => decodeParams(CA_SCHEMA, CA_DEFAULTS, search);
const encode = (settings: CASettings) => encodeParams(CA_SCHEMA, CA_DEFAULTS, settings);

describe("cellular-automata settings", () => {
  it("round-trips through a shareable link and keeps links short", () => {
    expect(encode(CA_DEFAULTS)).toBe("");
    const changed: CASettings = { ...CA_DEFAULTS, birth: "36", speed: 30, theme: 3, gridSize: 512, pattern: "Spaceship (LWSS)", showGraph: false };
    const search = encode(changed);
    expect(search).toBe("?birth=36&speed=30&theme=3&gridSize=512&pattern=Spaceship+%28LWSS%29&showGraph=0");
    expect(decode(search)).toEqual(changed);
  });

  it("round-trips every library pattern, random and clear", () => {
    for (const pattern of PATTERN_CHOICES) {
      expect(decode(encode({ ...CA_DEFAULTS, pattern })).pattern, pattern).toBe(pattern);
    }
    expect(PATTERN_CHOICES).toEqual(expect.arrayContaining([...patterns.map((p) => p.name), "random", "clear"]));
    expect(PATTERN_CHOICES).toContain(DEFAULT_PATTERN);
  });

  it("sanitises hostile links", () => {
    const s = decode("?birth=abc&survive=9&speed=1e9&theme=9&gridSize=999&pattern=nope&showGraph=maybe");
    expect(s).toEqual({ ...CA_DEFAULTS, speed: 60 });
    expect(decode("?speed=-5&gridSize=1e3&theme=1.5")).toEqual({ ...CA_DEFAULTS, speed: 1 });
  });

  it("accepts only the grid sizes and themes that exist", () => {
    expect(decode("?gridSize=1024&theme=2")).toMatchObject({ gridSize: 1024, theme: 2 });
    expect(decode("?gridSize=300").gridSize).toBe(CA_DEFAULTS.gridSize);
  });

  it("writes rules in their canonical form, so equivalent rules are the same link", () => {
    expect(sanitizeRuleDigits("6 3-x3")).toBe("36");
    expect(sanitizeRuleDigits("9")).toBe("");
    expect(decode("?birth=63&survive=3322")).toMatchObject({ birth: "36", survive: "23" });
    expect(encode({ ...CA_DEFAULTS, survive: "32" })).toBe(""); // 32 is the default rule 23, typed backwards
    expect(encode({ ...CA_DEFAULTS, birth: "63" })).toBe("?birth=36");
  });

  it("allows an empty rule (nothing is ever born / nothing survives) without mistaking garbage for one", () => {
    const seeds = { ...CA_DEFAULTS, birth: "2", survive: "" };
    expect(encode(seeds)).toBe("?birth=2&survive=");
    expect(decode(encode(seeds))).toEqual(seeds);
    expect(decode("?survive=zzz").survive).toBe(CA_DEFAULTS.survive);
  });

  it("recognises Conway's rules however they were typed", () => {
    expect(isConway(CA_DEFAULTS)).toBe(true);
    expect(isConway({ birth: "3", survive: "32" })).toBe(true);
    expect(isConway({ birth: "33", survive: "2233" })).toBe(true);
    expect(isConway({ birth: "36", survive: "23" })).toBe(false);
    expect(isConway({ birth: "", survive: "" })).toBe(false);
  });
});
