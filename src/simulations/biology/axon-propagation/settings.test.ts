import { describe, expect, it } from "vitest";
import { decodeParams, encodeParams } from "@/lib/urlState";
import { AXON_DEFAULTS, AXON_SCHEMA, effectiveDiameter } from "./settings";
import type { AxonSettings } from "./settings";

const decode = (search: string) => decodeParams(AXON_SCHEMA, AXON_DEFAULTS, search);
const encode = (s: AxonSettings) => encodeParams(AXON_SCHEMA, AXON_DEFAULTS, s);

describe("axon settings", () => {
  it("round-trip through a link, and the defaults make an empty link", () => {
    expect(encode(AXON_DEFAULTS)).toBe("");
    const changed: AxonSettings = { ...AXON_DEFAULTS, view: "nerve", diameter: 10, myelin: true, temp: 25, stim: 3, pulses: "pair", gap: 3.5, drug: "ttx", from: 0.2, to: 0.3, myelinLeft: 0.25, fibres: 1000, distance: 1.2 };
    expect(decode(encode(changed))).toEqual(changed);
  });

  it("hostile links are ignored or clamped, never crash", () => {
    const s = decode("?view=brain&diameter=1e9&temp=-40&pulses=many&drug=coffee&fibres=7&myelinLeft=3&distance=abc");
    expect(s).toEqual({ ...AXON_DEFAULTS, diameter: 1000, temp: 0, myelinLeft: 1 });
  });

  it("myelinated fibres are capped at 20 µm", () => {
    expect(effectiveDiameter({ diameter: 476, myelin: true })).toBe(20);
    expect(effectiveDiameter({ diameter: 476, myelin: false })).toBe(476);
  });
});
