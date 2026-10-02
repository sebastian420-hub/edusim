import { clock, init, surface } from "vgpu";
import type { Clock, Gpu, Surface } from "vgpu";

export type GpuStatus =
  | { state: "loading" }
  | { state: "ready" }
  | { state: "unsupported" }
  | { state: "error"; message: string };

/** Everything a simulation needs from the GPU runtime. */
export interface SimContext {
  gpu: Gpu;
  canvas: HTMLCanvasElement;
  surface: Surface;
  clock: Clock;
}

/** Minimal contract every simulation handle fulfils. */
export interface SimHandle {
  dispose?(): void;
}

/** Builds a simulation on an initialised GPU. Must be a stable (module-level) function. */
export type SimFactory<H extends SimHandle> = (ctx: SimContext) => H | Promise<H>;

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && !!navigator.gpu;
}

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Initialises WebGPU, binds a surface to `canvas` and runs `factory`.
 *
 * Follows vgpu's documented lifecycle: the returned teardown is synchronous and safe to call at any
 * point — before `init()` resolves, while the factory is running, or after the sim is ready — so
 * React strict-mode double mounts never leak a device or render loop.
 */
export function launchSim<H extends SimHandle>(
  canvas: HTMLCanvasElement,
  factory: SimFactory<H>,
  onReady: (handle: H) => void,
  onStatus: (status: GpuStatus) => void,
): () => void {
  let disposed = false;
  let gpu: Gpu | undefined;
  let handle: H | undefined;

  void (async () => {
    if (!hasWebGPU()) {
      onStatus({ state: "unsupported" });
      return;
    }
    try {
      const g = await init();
      if (disposed) {
        g.dispose();
        return;
      }
      gpu = g;
      g.onError((err) => {
        console.error("[gpu]", err.message, { cause: err.cause, detail: (err as { detail?: unknown }).detail });
        if (!disposed) onStatus({ state: "error", message: err.message });
      });
      const ctx: SimContext = {
        gpu: g,
        canvas,
        surface: surface(g, canvas, { dpr: [1, 2] }),
        clock: clock(g),
      };
      const built = await factory(ctx);
      if (disposed) {
        built.dispose?.();
        return;
      }
      handle = built;
      onReady(built);
      onStatus({ state: "ready" });
    } catch (e) {
      if (!disposed) onStatus({ state: "error", message: messageOf(e) });
    }
  })();

  return () => {
    disposed = true;
    handle?.dispose?.();
    gpu?.dispose();
  };
}

/** Seconds since the previous frame, clamped so a backgrounded tab doesn't cause a huge jump. */
export function frameDelta(c: Clock, max = 0.1): number {
  return Math.min(Math.max(c.deltaTime, 0), max);
}

/**
 * PNG snapshot of a WebGPU canvas. The drawing buffer only holds valid pixels during the frame it was
 * presented in, so the read happens inside a requestAnimationFrame callback (which runs after the
 * render loop's callback for the same frame) rather than at an arbitrary time.
 */
export function captureCanvas(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    requestAnimationFrame(() =>
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Canvas capture failed"))), "image/png"),
    );
  });
}
