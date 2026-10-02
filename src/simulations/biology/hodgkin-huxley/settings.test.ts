import { describe, expect, it } from "vitest";
import { decodeParams, encodeParams } from "@/lib/urlState";
import { DEFAULT_PARAMS, fiAnalysis } from "./hh";
import { computeReadouts, HH_DEFAULTS, HH_SCHEMA, toParams } from "./settings";
import type { HHSettings } from "./settings";

describe("hodgkin-huxley settings", () => {
  it("round-trips through a shareable link and keeps links short", () => {
    expect(encodeParams(HH_SCHEMA, HH_DEFAULTS, HH_DEFAULTS)).toBe("");
    const changed: HHSettings = { ...HH_DEFAULTS, I_inj: 12.5, g_Na: 0, pulse_mode: 2, temperature: 18.2 };
    const search = encodeParams(HH_SCHEMA, HH_DEFAULTS, changed);
    expect(search).toBe("?I_inj=12.5&temperature=18.2&g_Na=0&pulse_mode=2");
    expect(decodeParams(HH_SCHEMA, HH_DEFAULTS, search)).toEqual(changed);
  });

  it("sanitises hostile links", () => {
    const s = decodeParams(HH_SCHEMA, HH_DEFAULTS, "?I_inj=1e9&pulse_mode=7&g_K=-5&temperature=NaN");
    expect(s.I_inj).toBe(50);
    expect(s.pulse_mode).toBe(0);
    expect(s.g_K).toBe(0);
    expect(s.temperature).toBe(HH_DEFAULTS.temperature);
  });

  it("maps settings onto the full model, keeping the fixed constants", () => {
    const p = toParams({ ...HH_DEFAULTS, g_Na: 50 });
    expect(p.g_Na).toBe(50);
    expect(p.E_Na).toBe(DEFAULT_PARAMS.E_Na);
    expect(p.C_m).toBe(DEFAULT_PARAMS.C_m);
  });

  it("readouts: DC reports a rate, pulse modes report spikes per cycle", () => {
    const fi = fiAnalysis(toParams(HH_DEFAULTS));
    const dc = computeReadouts({ ...HH_DEFAULTS, I_inj: 10 }, fi);
    expect(dc.rate).toBeGreaterThan(60);
    expect(dc.spikesPerCycle).toBe(0);
    const pulse = computeReadouts({ ...HH_DEFAULTS, I_inj: 10, pulse_mode: 1 }, fi);
    expect(pulse.rate).toBe(0);
    expect(pulse.spikesPerCycle).toBe(1);
  });
});
