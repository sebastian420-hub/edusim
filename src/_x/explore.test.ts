import { it } from "vitest";
import { runFibre, conductionVelocity, HH_TEMPERATURE, extracellular } from "@/simulations/biology/axon-propagation/cable";
import { nerveComposition, nerveFibre, nerveStimulus, NERVE_LENGTH } from "@/simulations/biology/axon-propagation/model";
it("explore", () => {
  const comp = nerveComposition(300);
  const fibres = comp.map(nerveFibre);
  console.log("total comps", fibres.reduce((a, f) => a + f.n, 0), "max", Math.max(...fibres.map((f) => f.n)), "min", Math.min(...fibres.map((f) => f.n)));
  for (const k of [0, 50, 100, 150, 200, 299]) {
    const f = fibres[k];
    const r = runFibre(f, HH_TEMPERATURE, [nerveStimulus(f, 12)], 30, 0.01);
    console.log(comp[k].kind, comp[k].diameter.toFixed(2), "n", f.n, "v", conductionVelocity(f, r.arrival)?.toFixed(2), "arrive end", r.arrival[f.n - 1].toFixed(2));
  }
  for (const S of [1, 2, 4, 6, 12]) {
    let n = 0; for (const [k, f] of fibres.entries()) { if (k % 10) continue; const r = runFibre(f, HH_TEMPERATURE, [nerveStimulus(f, S)], 3); n += Number(r.crossings[Math.floor(f.n / 3)] > 0 || r.crossings[f.n - 1] > 0 || !Number.isNaN(r.arrival[Math.floor(f.n * 0.1)])); }
    console.log("S", S, "recruited of 30", n);
  }
}, 900000);
