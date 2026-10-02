/**
 * Frame pacing and adaptive resolution.
 *
 * Problem: a render loop that submits a frame every display tick keeps queuing work even when the GPU
 * can't keep up. The queue grows, input (Pause, sliders) takes seconds to show, and memory climbs.
 * Measuring CPU-side frame intervals doesn't help — they stay healthy while the GPU falls behind.
 *
 * Solution, in two independent, pure (and therefore unit-testable) parts:
 *  - `FramePacer` bounds the number of frames in flight and estimates the real GPU cost per frame
 *    from when each frame's work completes.
 *  - `QualityGovernor` turns those cost estimates into a render-resolution level: it steps down when
 *    frames are persistently too expensive and steps back up only when the *predicted* cost at the
 *    higher resolution fits comfortably, so it doesn't oscillate.
 */

export interface FramePacerOptions {
  /** Frames allowed to be queued on the GPU at once before ticks are skipped. */
  maxInFlight?: number;
  /** Millisecond clock; injectable for tests. */
  now?: () => number;
}

export class FramePacer {
  private inFlight = 0;
  private readonly maxInFlight: number;
  private readonly now: () => number;
  /** Ticks skipped because the GPU still had `maxInFlight` frames queued. */
  skipped = 0;
  /** Frames submitted and tracked. */
  submitted = 0;

  constructor({ maxInFlight = 2, now = () => performance.now() }: FramePacerOptions = {}) {
    this.maxInFlight = maxInFlight;
    this.now = now;
  }

  /** Whether to render on this tick. False means: the GPU is behind, skip and try next tick. */
  shouldRender(): boolean {
    if (this.inFlight >= this.maxInFlight) {
      this.skipped++;
      return false;
    }
    return true;
  }

  /**
   * Registers a submitted frame whose GPU work completes when `done` resolves. `onCost` receives an
   * estimate of that frame's own GPU time in ms: the wait since submission divided by the number of
   * frames that were ahead of (and including) it in the queue.
   */
  track(done: Promise<unknown>, onCost: (costMs: number) => void): void {
    const queued = ++this.inFlight;
    this.submitted++;
    const start = this.now();
    void done.then(
      () => this.complete(start, queued, onCost),
      () => this.complete(start, queued, onCost),
    );
  }

  private complete(start: number, queued: number, onCost: (costMs: number) => void) {
    this.inFlight = Math.max(0, this.inFlight - 1);
    onCost((this.now() - start) / queued);
  }

  get pending(): number {
    return this.inFlight;
  }
}

const WINDOW = 9;
const MIN_SAMPLES = 5;

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

export interface QualityOptions {
  /** Render scales from best to worst quality (fractions of full resolution per axis). */
  levels?: readonly number[];
  /** Target maximum GPU time per frame (ms); above this the resolution is lowered. */
  budgetMs?: number;
  /** Frames ignored at start (shader compilation and first-use costs are not representative). */
  warmupFrames?: number;
  /** Frames ignored after every change, while the new size settles. */
  cooldownFrames?: number;
  /** Consecutive over-budget frames needed before stepping down. */
  degradeFrames?: number;
  /** Consecutive frames with comfortable predicted headroom needed before stepping up. */
  upgradeFrames?: number;
  /** Fraction of the budget the *predicted* cost at the higher quality must stay under. */
  headroom?: number;
}

export class QualityGovernor {
  private readonly levels: readonly number[];
  private readonly budgetMs: number;
  private readonly warmupFrames: number;
  private readonly cooldownFrames: number;
  private readonly degradeFrames: number;
  private readonly upgradeFrames: number;
  private readonly headroom: number;

  level = 0;
  private seen = 0;
  private cooldown = 0;
  private over = 0;
  private under = 0;
  /** Recent frame costs; their median ignores isolated spikes (shader compiles, GC pauses). */
  private window: number[] = [];

  constructor(options: QualityOptions = {}) {
    this.levels = options.levels ?? [1, 0.75, 0.5, 0.35];
    this.budgetMs = options.budgetMs ?? 25;
    this.warmupFrames = options.warmupFrames ?? 20;
    this.cooldownFrames = options.cooldownFrames ?? 12;
    this.degradeFrames = options.degradeFrames ?? 6;
    this.upgradeFrames = options.upgradeFrames ?? 90;
    this.headroom = options.headroom ?? 0.6;
  }

  get scale(): number {
    return this.levels[this.level];
  }

  /** Feeds one frame's GPU cost (ms). Returns true when the resolution level changed. */
  record(costMs: number): boolean {
    this.seen++;
    if (this.seen <= this.warmupFrames) return false;
    if (this.cooldown > 0) {
      this.cooldown--;
      return false;
    }

    this.window.push(costMs);
    if (this.window.length > WINDOW) this.window.shift();
    if (this.window.length < MIN_SAMPLES) return false;
    const cost = median(this.window);

    if (cost > this.budgetMs && this.level < this.levels.length - 1) {
      this.under = 0;
      if (++this.over >= this.degradeFrames) return this.move(+1);
      return false;
    }

    this.over = 0;
    if (this.level > 0) {
      // Cost is roughly proportional to the number of pixels, i.e. scale squared.
      const predicted = cost * (this.levels[this.level - 1] / this.levels[this.level]) ** 2;
      if (predicted < this.budgetMs * this.headroom) {
        if (++this.under >= this.upgradeFrames) return this.move(-1);
      } else {
        this.under = 0;
      }
    }
    return false;
  }

  private move(delta: number): boolean {
    this.level += delta;
    this.cooldown = this.cooldownFrames;
    this.over = 0;
    this.under = 0;
    this.window = [];
    return true;
  }
}
