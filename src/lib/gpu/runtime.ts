import { clock, frameLoop, init, surface } from "vgpu";
import type { Clock, Frame, FrameLoopHandle, Gpu, Surface } from "vgpu";
import { FramePacer, QualityGovernor } from "./pacing";

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
  /**
   * The render loop. Prefer this over vgpu's `frameLoop`: it skips ticks while the GPU is still busy
   * (so a slow device never builds up a queue of stale frames), passes `dt` — seconds of real time
   * since the previous *rendered* tick, clamped — and feeds the adaptive-resolution governor.
   */
  loop(callback: (frame: Frame, dt: number) => void): FrameLoopHandle;
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

/** vgpu's CanvasSurface supports explicit resizing; the public `Surface` type just doesn't declare it. */
type ResizableSurface = Surface & { resize(size: readonly [number, number]): void };

const MAX_DPR = 2;

/** Key of the localStorage setting that pins full resolution (value "full"): crisp projector/screenshots. */
export const QUALITY_STORAGE_KEY = "edusim:quality";

function fullQualityForced(): boolean {
  try {
    return window.localStorage.getItem(QUALITY_STORAGE_KEY) === "full";
  } catch {
    return false;
  }
}

/**
 * A surface whose resolution we control: canvas pixels = CSS size x device pixel ratio (1..2) x `scale`.
 * Lowering `scale` is how the quality governor trades resolution for frame rate.
 */
function createAdaptiveSurface(g: Gpu, canvas: HTMLCanvasElement) {
  const pixelRatio = () => Math.min(MAX_DPR, Math.max(1, window.devicePixelRatio || 1));
  const sizeFor = (scale: number): [number, number] => [
    Math.max(1, Math.round(canvas.clientWidth * pixelRatio() * scale)),
    Math.max(1, Math.round(canvas.clientHeight * pixelRatio() * scale)),
  ];

  let scale = 1;
  const target = surface(g, canvas, { autoResize: false, size: sizeFor(1) }) as ResizableSurface;
  const apply = () => {
    if (!target.disposed) target.resize(sizeFor(scale));
  };
  const observer = new ResizeObserver(apply);
  observer.observe(canvas);

  return {
    surface: target as Surface,
    setScale(next: number) {
      scale = next;
      apply();
    },
    dispose: () => observer.disconnect(),
  };
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
  onQuality: (scale: number) => void = () => {},
): () => void {
  let disposed = false;
  let gpu: Gpu | undefined;
  let handle: H | undefined;
  let stopSurface: (() => void) | undefined;

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
      const adaptive = createAdaptiveSurface(g, canvas);
      stopSurface = adaptive.dispose;
      const frameClock = clock(g);
      const pacer = new FramePacer();
      const governor = new QualityGovernor();
      const ctx: SimContext = {
        gpu: g,
        canvas,
        surface: adaptive.surface,
        clock: frameClock,
        loop(callback) {
          let lastTime: number | undefined;
          return frameLoop(g, (frame) => {
            if (!pacer.shouldRender()) return; // GPU still busy with earlier frames: skip this tick
            const now = frameClock.time;
            const dt = lastTime === undefined ? 0 : Math.min(Math.max(now - lastTime, 0), 0.1);
            lastTime = now;
            callback(frame, dt);
            // vgpu submits the frame when this callback returns, so ask the queue about it right after.
            // (`frame.done` is no use here: it resolves before the GPU has actually finished.)
            queueMicrotask(() => {
              if (disposed) return;
              pacer.track(g.gpu.queue.onSubmittedWorkDone(), (costMs) => {
                if (disposed || fullQualityForced() || !governor.record(costMs)) return;
                adaptive.setScale(governor.scale);
                onQuality(governor.scale);
              });
            });
          });
        },
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
    stopSurface?.();
    handle?.dispose?.();
    gpu?.dispose();
  };
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
