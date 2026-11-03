import { describe, expect, it } from "vitest";
import { decodeParams, encodeParams } from "@/lib/urlState";
import { parameterCount } from "./model";
import { workMegabytes } from "./plan";
import { fittedBatch, LM_DEFAULTS, LM_SCHEMA, lmConfigOf, MEMORY_LIMIT_MB, specOf } from "./settings";

describe("language-model settings", () => {
  it("round-trip through a link, writing only what differs from the defaults", () => {
    const s = { ...LM_DEFAULTS, view: "predict" as const, text: "names" as const, model: "mlp" as const, k: 5, lr: 0.0123, temp: 0.35, tk: 2 };
    const query = encodeParams(LM_SCHEMA, LM_DEFAULTS, s);
    expect(query).toBe("?view=predict&text=names&model=mlp&k=5&lr=0.0123&temp=0.35&tk=2");
    expect(decodeParams(LM_SCHEMA, LM_DEFAULTS, query)).toEqual(s);
    expect(encodeParams(LM_SCHEMA, LM_DEFAULTS, LM_DEFAULTS)).toBe("");
  });

  it("ignore hostile values: unknown names, sizes the kernels can't run, NaN", () => {
    const s = decodeParams(LM_SCHEMA, LM_DEFAULTS, "?view=brain&text=../etc&model=gpt5&layers=99&heads=3&width=7&ctx=1e9&lr=-1&temp=nan&tk=12&k=0&batch=1000");
    expect(s.view).toBe("train");
    expect(s.text).toBe("shakespeare");
    expect(s.model).toBe("gpt");
    expect(s.layers).toBe(4);
    expect(s.heads).toBe(LM_DEFAULTS.heads);
    expect(s.width).toBe(LM_DEFAULTS.width);
    expect(s.ctx).toBe(LM_DEFAULTS.ctx);
    expect(s.lr).toBe(1e-4);
    expect(s.temp).toBe(LM_DEFAULTS.temp);
    expect(s.tk).toBe(6);
    expect(s.k).toBe(1);
    expect(s.batch).toBe(LM_DEFAULTS.batch);
  });

  it("keep every head at most 64 wide and every model within GPU memory", () => {
    expect(specOf({ ...LM_DEFAULTS, width: 128, heads: 1 }).H).toBe(2);
    for (const width of [32, 64, 128]) {
      for (const layers of [1, 2, 3, 4]) {
        for (const ctx of [16, 32, 64, 128]) {
          for (const batch of [8, 16, 32]) {
            const s = { ...LM_DEFAULTS, width, layers, ctx, batch };
            const b = fittedBatch(s);
            expect(b).toBeGreaterThanOrEqual(1);
            expect(workMegabytes({ ...specOf(s), V: 96 }, b)).toBeLessThanOrEqual(MEMORY_LIMIT_MB);
          }
        }
      }
    }
    // The default and the "bigger" model keep their batch.
    expect(lmConfigOf(LM_DEFAULTS).batch).toBe(16);
    expect(fittedBatch({ ...LM_DEFAULTS, layers: 4, width: 128 })).toBe(16);
  });

  it("describe the plan's model sizes: about 0.1 M and 0.8 M parameters", () => {
    expect(parameterCount({ ...specOf(LM_DEFAULTS), V: 65 })).toBe(112_577);
    expect(parameterCount({ ...specOf({ ...LM_DEFAULTS, layers: 4, width: 128 }), V: 65 })).toBe(818_241);
  });
});
