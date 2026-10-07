import { describe, expect, it } from "vitest";
import { decodeParams, encodeParams } from "@/lib/urlState";
import { orbitPreset } from "./presets";
import { bodyNames, decodeBodies, encodeBodies, NBODY_DEFAULTS, NBODY_SCHEMA, setupBodies } from "./settings";
import type { NBodySettings } from "./settings";

const decode = (search: string) => decodeParams(NBODY_SCHEMA, NBODY_DEFAULTS, search);
const encode = (s: NBodySettings) => encodeParams(NBODY_SCHEMA, NBODY_DEFAULTS, s);

describe("n-body settings", () => {
  it("round-trip through a link, and the defaults make an empty link", () => {
    expect(encode(NBODY_DEFAULTS)).toBe("");
    const changed: NBodySettings = { ...NBODY_DEFAULTS, mode: "galaxy", galaxy: "disk", count: 16384, galaxySpeed: 2, integrator: "euler" };
    expect(decode(encode(changed))).toEqual(changed);
  });

  it("a link carries an exact orbit-lab setup", () => {
    const bodies = [
      { m: 1, x: 0, y: 0, vx: 0, vy: 0 },
      { m: 3.003e-6, x: 2, y: 0, vx: 0, vy: 4.44288 },
    ];
    const s = { ...NBODY_DEFAULTS, bodies: encodeBodies(bodies) };
    const back = setupBodies(decode(encode(s)));
    expect(back).toEqual(bodies);
  });

  it("without an edited setup, the preset's bodies are used", () => {
    expect(setupBodies({ preset: "inner-planets", bodies: "" })).toEqual(orbitPreset("inner-planets").bodies);
    expect(bodyNames("inner-planets", 6)).toEqual(["Sun", "Mercury", "Venus", "Earth", "Mars", "Planet 6"]);
  });

  it("hostile links are ignored or clamped, never crash", () => {
    const s = decode("?mode=wormhole&preset=nope&speed=1e9&count=7&galaxy=x&integrator=rk4&bodies=1,2,3");
    expect(s).toEqual({ ...NBODY_DEFAULTS, speed: 4 });
    for (const bad of ["1,0,0,0", "x,0,0,0,0", "1,0,0,0,0;" + "1,1,1,1,1;".repeat(8), "-1,0,0,0,0", "1,1e9,0,0,0", "1,0,0,1e6,0", "1,0,0,,0"]) {
      expect(decodeBodies(bad), bad).toBeUndefined();
      expect(decode(`?bodies=${encodeURIComponent(bad)}`).bodies, bad).toBe("");
    }
  });
});
