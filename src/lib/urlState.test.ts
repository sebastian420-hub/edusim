import { describe, expect, it } from "vitest";
import { boolField, decodeParams, encodeParams, enumField, hasKnownParams, numberField } from "./urlState";
import type { Schema } from "./urlState";

interface P {
  freq: number;
  mode: "a" | "b";
  on: boolean;
}
const schema: Schema<P> = { freq: numberField(1, 10), mode: enumField(["a", "b"] as const), on: boolField };
const defaults: P = { freq: 3, mode: "a", on: true };

describe("urlState", () => {
  it("omits defaults and round-trips changes", () => {
    expect(encodeParams(schema, defaults, defaults)).toBe("");
    const changed: P = { freq: 5.5, mode: "b", on: false };
    const search = encodeParams(schema, defaults, changed);
    expect(search).toBe("?freq=5.5&mode=b&on=0");
    expect(decodeParams(schema, defaults, search)).toEqual(changed);
  });

  it("clamps numbers and ignores malformed or unknown values", () => {
    expect(decodeParams(schema, defaults, "?freq=999").freq).toBe(10);
    expect(decodeParams(schema, defaults, "?freq=-4").freq).toBe(1);
    expect(decodeParams(schema, defaults, "?freq=abc&mode=zzz&on=maybe&junk=1")).toEqual(defaults);
    expect(decodeParams(schema, defaults, "?freq=")).toEqual(defaults);
  });

  it("only changes what the URL specifies", () => {
    expect(decodeParams(schema, defaults, "?mode=b")).toEqual({ freq: 3, mode: "b", on: true });
  });

  it("detects known parameters", () => {
    expect(hasKnownParams(schema, "?mode=b")).toBe(true);
    expect(hasKnownParams(schema, "?utm=1")).toBe(false);
    expect(hasKnownParams(schema, "")).toBe(false);
  });

  it("trims float noise when encoding", () => {
    expect(encodeParams(schema, defaults, { ...defaults, freq: 0.1 + 0.2 + 3 })).toBe("?freq=3.3");
  });
});
