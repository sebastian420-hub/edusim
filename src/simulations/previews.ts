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
