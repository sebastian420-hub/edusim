import { describe, expect, it } from "vitest";
import { decodeParams, encodeParams } from "@/lib/urlState";
import { layersField, NET_DEFAULTS, NET_SCHEMA, planeSpec } from "./settings";
import type { NetSettings } from "./settings";

const decode = (search: string) => decodeParams(NET_SCHEMA, NET_DEFAULTS, search);
const encode = (s: NetSettings) => encodeParams(NET_SCHEMA, NET_DEFAULTS, s);

describe("neural network settings", () => {
  it("round-trip through a link, and the defaults make an empty link", () => {
    expect(encode(NET_DEFAULTS)).toBe("");
    const changed: NetSettings = { ...NET_DEFAULTS, view: "digits", data: "xor", points: 800, noise: 0.2, train: 0.1, features: 0b10011, layers: [], act: "sigmoid", lr: 0.000123, batch: 2000, mom: 0, l2: 0.001, init: "zeros", seed: 4242, speed: 20, dnet: "linear", dlr: 0.5, dbatch: 128 };
    expect(decode(encode(changed))).toEqual(changed);
    expect(encode({ ...NET_DEFAULTS, layers: [4, 2, 8] })).toBe("?layers=4-2-8");
  });

  it("hostile links are ignored or clamped, never crash", () => {
    const s = decode("?view=matrix&data=moons&points=1e9&noise=-1&features=0&layers=8-x&act=gelu&lr=1e6&batch=17&mom=2&seed=-5&speed=0.2&dnet=cnn&dbatch=7");
    expect(s).toEqual({ ...NET_DEFAULTS, points: 2000, noise: 0, lr: 30, mom: 0.99, seed: 1, speed: 1 });
    expect(decode("?features=128").features).toBe(NET_DEFAULTS.features);
    expect(decode("?layers=1-2-3-4-5-6-7").layers).toEqual(NET_DEFAULTS.layers); // too many layers
    expect(layersField.decode("99-0")).toEqual([32, 1]);
    expect(decode("?layers=none").layers).toEqual([]);
  });

  it("the network follows the features and layers", () => {
    expect(planeSpec({ features: 0b1111111, layers: [3], act: "tanh" })).toEqual({ inputs: 7, hidden: [3], outputs: 1, activation: "tanh" });
  });
});
