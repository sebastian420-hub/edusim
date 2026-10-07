import { energy, growthRate, PeriodTracker, separation } from "./pendulum";
import type { PendulumParams, State } from "./pendulum";

/** Samples kept for the graphs. */
const HISTORY = 600;

/** Which "turn" an angle is on: changes by one each time the arm passes over the top. */
const turn = (theta: number) => Math.floor((theta + Math.PI) / (2 * Math.PI));

export interface SwingStats {
  time: number;
  angles: [number, number];
  kinetic: number;
  potential: number;
  total: number;
  /** (E − E₀)/|E₀|. */
  drift: number;
  /** Times either arm has gone over the top, and when the first did. */
  flips: number;
  firstFlip?: number;
  /** The last measured periods of θ₁ (upward zero crossings), oldest first. */
  periods: number[];
  /** Recent (θ₁, θ₂) in radians, unwrapped, for the phase plot. */
  phase: { t1: number[]; t2: number[] };
}

/** Everything measured on a single pendulum, from its (64-bit) state. Used by the simulation and the tests alike. */
export class SwingObserver {
  private e0: number;
  private turns: [number, number];
  private flips = 0;
  private firstFlip: number | undefined;
  private readonly periods = new PeriodTracker();
  private t1: number[] = [];
  private t2: number[] = [];
  private last: { t: number; s: State };

  constructor(
    start: State,
    private readonly params: PendulumParams,
  ) {
    this.e0 = energy(start, params).total;
    this.turns = [turn(start[0]), turn(start[1])];
    this.last = { t: 0, s: start };
    this.periods.sample(0, start[0]);
  }

  /** Call after every step (or every few steps: flips and zero crossings need well under half a swing between samples). */
  sample(t: number, s: State): void {
    const k: [number, number] = [turn(s[0]), turn(s[1])];
    const crossed = Math.abs(k[0] - this.turns[0]) + Math.abs(k[1] - this.turns[1]);
    if (crossed > 0) {
      this.flips += crossed;
      this.firstFlip ??= t;
      this.turns = k;
    }
    this.periods.sample(t, s[0]);
    this.last = { t, s };
  }

  /** Adds the current angles to the phase plot (call at the display rate). */
  record(): void {
    this.t1.push(this.last.s[0]);
    this.t2.push(this.last.s[1]);
    if (this.t1.length > HISTORY) {
      this.t1.shift();
      this.t2.shift();
    }
  }

  stats(): SwingStats {
    const { t, s } = this.last;
    const e = energy(s, this.params);
    return {
      time: t,
      angles: [s[0], s[1]],
      kinetic: e.kinetic,
      potential: e.potential,
      total: e.total,
      drift: (e.total - this.e0) / Math.abs(this.e0 || 1),
      flips: this.flips,
      firstFlip: this.firstFlip,
      periods: this.periods.periods(8),
      phase: { t1: [...this.t1], t2: [...this.t2] },
    };
  }
}

export interface SpreadStats {
  time: number;
  /** Current gap between the two (θ₁, θ₂) runs, rad. */
  spread: number;
  maxSpread: number;
  /** When the gap first passed 1 rad: the runs look completely different. */
  visibleAt?: number;
  /** Exponential growth rate of the gap (Lyapunov exponent, 1/s), once measurable. */
  lambda?: number;
  history: { t: number[]; log10: number[] };
}

/** The gap between a pendulum and its nudged twin, its growth and its growth rate. */
export class SpreadObserver {
  private maxSpread = 0;
  private visibleAt: number | undefined;
  private times: number[] = [];
  private seps: number[] = [];
  private hist = { t: [] as number[], log10: [] as number[] };
  private last = { t: 0, d: 0 };

  /** Call every few milliseconds of simulated time (the growth-rate fit uses every sample). */
  sample(t: number, a: State, b: State): void {
    const d = separation(a, b);
    this.last = { t, d };
    this.maxSpread = Math.max(this.maxSpread, d);
    if (this.visibleAt === undefined && d > 1) this.visibleAt = t;
    this.times.push(t);
    this.seps.push(d);
    if (this.times.length > 20000) {
      this.times.splice(0, 10000);
      this.seps.splice(0, 10000);
    }
  }

  /** Adds the current gap to the graph (call at the display rate). */
  record(): void {
    this.hist.t.push(this.last.t);
    this.hist.log10.push(Math.log10(Math.max(1e-16, this.last.d)));
    if (this.hist.t.length > HISTORY) {
      // Keep the whole run visible: halve the resolution instead of dropping the start.
      this.hist.t = this.hist.t.filter((_, i) => i % 2 === 0);
      this.hist.log10 = this.hist.log10.filter((_, i) => i % 2 === 0);
    }
  }

  stats(): SpreadStats {
    return {
      time: this.last.t,
      spread: this.last.d,
      maxSpread: this.maxSpread,
      visibleAt: this.visibleAt,
      lambda: growthRate(this.times, this.seps),
      history: { t: [...this.hist.t], log10: [...this.hist.log10] },
    };
  }
}
