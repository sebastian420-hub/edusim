import { describe, expect, it, vi } from "vitest";
import { mayGoLive, SingleLive } from "./intent";

const env = { reducedMotion: false, hasWebGPU: true, pointerType: "mouse" };

describe("mayGoLive", () => {
  it("allows mouse, pen and keyboard users who permit motion and have WebGPU", () => {
    for (const pointerType of ["mouse", "pen", "keyboard"]) expect(mayGoLive({ ...env, pointerType })).toBe(true);
  });

  it("never starts on touch", () => {
    expect(mayGoLive({ ...env, pointerType: "touch" })).toBe(false);
  });

  it("respects prefers-reduced-motion and missing WebGPU", () => {
    expect(mayGoLive({ ...env, reducedMotion: true })).toBe(false);
    expect(mayGoLive({ ...env, hasWebGPU: false })).toBe(false);
  });

  it("ignores unknown pointer types", () => {
    expect(mayGoLive({ ...env, pointerType: "" })).toBe(false);
  });
});

describe("SingleLive", () => {
  it("allows only one live plate: activating a second stops the first", () => {
    const registry = new SingleLive();
    const stopA = vi.fn();
    const stopB = vi.fn();
    registry.activate("a", stopA);
    registry.activate("b", stopB);
    expect(stopA).toHaveBeenCalledTimes(1);
    expect(stopB).not.toHaveBeenCalled();
    expect(registry.activeId).toBe("b");
  });

  it("re-activating the same plate does not stop it", () => {
    const registry = new SingleLive();
    const stop = vi.fn();
    registry.activate("a", stop);
    registry.activate("a", stop);
    expect(stop).not.toHaveBeenCalled();
  });

  it("release stops only the live plate and is a no-op for others", () => {
    const registry = new SingleLive();
    const stopA = vi.fn();
    registry.activate("a", stopA);
    registry.release("b");
    expect(stopA).not.toHaveBeenCalled();
    registry.release("a");
    expect(stopA).toHaveBeenCalledTimes(1);
    expect(registry.activeId).toBeNull();
    registry.release("a");
    expect(stopA).toHaveBeenCalledTimes(1);
  });
});
