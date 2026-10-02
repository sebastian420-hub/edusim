import { describe, expect, it } from "vitest";
import { implementedSimulations } from "@/lib/subjects";
import { PREVIEWS } from "./previews";

describe("home-page previews", () => {
  it("every implemented simulation has a preview, and nothing else does", () => {
    expect(Object.keys(PREVIEWS).sort()).toEqual(implementedSimulations().map((s) => s.id).sort());
  });

  it("each preview has a factory and a setup", () => {
    for (const [id, preview] of Object.entries(PREVIEWS)) {
      expect(typeof preview.factory, id).toBe("function");
      expect(typeof preview.setup, id).toBe("function");
    }
  });
});
