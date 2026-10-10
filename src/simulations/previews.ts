/**
 * How each simulation looks when it runs live inside its home-page plate.
 * Kept separate from the full simulation UIs: a preview is the bare canvas, small, in a fixed showpiece
 * state. Loaded only on intent (hover / keyboard focus), never on page load.
 */
import type { SimFactory, SimHandle } from "@/lib/gpu/runtime";
import { createCellularAutomata } from "./cs/cellular-automata/sim";
import type { CellularAutomataHandle } from "./cs/cellular-automata/sim";
import { createHodgkinHuxley } from "./biology/hodgkin-huxley/sim";
import type { HHHandle } from "./biology/hodgkin-huxley/sim";
import { createAxonSim } from "./biology/axon-propagation/sim";
import type { AxonHandle } from "./biology/axon-propagation/sim";
import { AXON_DEFAULTS } from "./biology/axon-propagation/settings";
import { createDoublePendulum } from "./physics/double-pendulum/sim";
import type { PendulumHandle } from "./physics/double-pendulum/sim";
import { DEFAULT_PENDULUM, FULL_WINDOW } from "./physics/double-pendulum/pendulum";
import { createNBody } from "./physics/n-body/sim";
import type { NBodyHandle } from "./physics/n-body/sim";
import { createWaveSim } from "./physics/wave-interference/sim";
import type { WaveHandle } from "./physics/wave-interference/sim";

export interface Preview {
  factory: SimFactory<SimHandle>;
  /** Puts the running simulation into its showpiece state; may return a cleanup function. */
  setup(handle: SimHandle, canvas: HTMLCanvasElement): void | (() => void);
}

/** Typed helper: declares a preview against its own handle type, stored under the common interface. */
function define<H extends SimHandle>(preview: { factory: SimFactory<H>; setup: (handle: H, canvas: HTMLCanvasElement) => void | (() => void) }): Preview {
  return preview as unknown as Preview;
}

const cellularAutomataFactory: SimFactory<CellularAutomataHandle> = (ctx) => createCellularAutomata(ctx, { gridSize: 256 });

export const PREVIEWS: Record<string, Preview> = {
  "wave-interference": define<WaveHandle>({
    factory: createWaveSim,
    setup(handle) {
      handle.setParams({ mode: "double-slit", view: "intensity", frequency: 5, separation: 0.7, slitWidth: 0.12, damping: 0.01 });
      // The intensity view is time-independent, so let the wavelength breathe: the fringes slowly widen and narrow.
      const start = performance.now();
      let frame = 0;
      const tick = () => {
        const t = (performance.now() - start) / 1000;
        handle.setParams({ frequency: 5 + 1.5 * Math.sin(t * 0.6) });
        frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
      return () => cancelAnimationFrame(frame);
    },
  }),

  "n-body": define<NBodyHandle>({
    factory: createNBody,
    setup(handle) {
      // Two galaxies colliding: the GPU showcase. Small enough for any GPU, at the most dramatic speed.
      handle.setGalaxy("collision", 2048);
      handle.setSpeed(3);
      handle.play();
    },
  }),

  "double-pendulum": define<PendulumHandle>({
    factory: (ctx) => createDoublePendulum(ctx),
    setup(handle) {
      // The flip-time fractal developing from dark: a quarter of a million pendulums, one per pixel.
      handle.configure({ view: "fractal", start: [0, 0], params: DEFAULT_PENDULUM, integrator: "rk4", count: 2, nudge: 1e-9, fractalSize: 512, window: FULL_WINDOW, boundary: false });
      handle.play();
    },
  }),

  "axon-propagation": define<AxonHandle>({
    factory: (ctx) => createAxonSim(ctx),
    setup(handle) {
      // Two spikes fired from both ends meet and annihilate: an X that never crosses, sweep after sweep.
      handle.configure({ ...AXON_DEFAULTS, pulses: "both" });
      handle.setSpeed(4);
      handle.play();
    },
  }),

  "hodgkin-huxley": define<HHHandle>({
    factory: createHodgkinHuxley,
    setup(handle) {
      handle.setParams({ I_inj: 12, temperature: 1, timeScale: 120, pulse_mode: 0 });
    },
  }),

  "cellular-automata": define<CellularAutomataHandle>({
    factory: cellularAutomataFactory,
    setup(handle, canvas) {
      handle.setParams({ theme: 3, speed: 14 }); // amber: the computer-science accent
      handle.randomize();
      const rect = canvas.getBoundingClientRect();
      handle.zoomAt(2.4, rect.width / 2, rect.height / 2);
      handle.play();
    },
  }),
};
