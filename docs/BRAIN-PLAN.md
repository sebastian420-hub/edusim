# A brain on the GPU — research and implementation plan

The flagship biology work: grow the single Hodgkin–Huxley neuron into **signals travelling down nerves** and
**networks of thousands of neurons firing in rhythms**, live in the browser, with measurements a test proves
correct. Same bar as every other simulation: a TypeScript twin of every shader, GPU-vs-twin tests, shareable
settings, guided challenges, e2e scenarios, a poster and a live home-page preview.

It ships as **two simulations**, built in this order:

1. **Axons & Nerves** (catalog id `axon-propagation`, already listed as planned): one axon, then a whole nerve
   of a thousand fibres.
2. **Brain Rhythms** (new catalog id `brain-rhythms`, route `/biology/brain-rhythms`): spiking networks from
   1 000 to 100 000 neurons, including a sheet of "cortex" where waves and spirals travel.

## 1. Why this, and what is genuinely missing

| What exists (free) | What it does | What it does not do |
|---|---|---|
| PhET *Neuron* | Qualitative picture of ion channels | No equations, no measurement, no propagation physics |
| HHsim (CMU), Neurons in Action | Proper HH single-neuron and propagation exercises (refractory period, TTX) | Desktop download / old HTML app; one axon; no networks |
| Neuronify (CINPLA) | Drag-and-drop integrate-and-fire circuits, aimed at students | A handful of neurons; app download (web embedding announced as a future version) |
| NEST Desktop (EBRAINS) | Browser GUI for the NEST research simulator | Needs a simulation server; built for university courses, not instant, not visual at scale |
| GeNN, NEST GPU, CARLsim, NeMo | Fast GPU spiking simulation | Research tools: Python/C++, no browser, no teaching layer |

**The gap:** nobody runs *thousands of HH fibres or 100 000 spiking neurons live in a browser tab*, with
measured conduction velocity, rhythms and spectra, and predict–observe–explain challenges. That is exactly the
combination EduSim already has the machinery for (WebGPU compute, twins, measurement, challenges, links).

## 2. What students should get out of it

From the physiology-education literature and the classic labs (HHsim, Neurons in Action, the frog sciatic nerve
compound action potential lab, Backyard Brains cockroach labs):

1. **The spike is regenerated, not conducted like current in a wire.** Each patch of membrane fires again; that
   is why the signal does not fade, and why it is slow (metres per second, not light speed).
2. **All-or-none and refractory.** A bigger stimulus does not make a bigger spike; a spike cannot fire again
   immediately, which is why it travels one way and two spikes that meet *annihilate*.
3. **Speed depends on geometry.** Thicker unmyelinated axons are faster, but only as √diameter; myelin makes
   speed proportional to diameter, which is how vertebrates got fast nerves without giant axons. The signal does
   not "jump through empty space": current flows under the myelin and the spike is regenerated at each node
   (a documented misconception).
4. **Disease and drugs are physics.** Demyelination (multiple sclerosis) slows and then blocks conduction; local
   anaesthetics (lidocaine) and pufferfish toxin (TTX) block sodium channels.
5. **Networks make rhythms.** Excitation and inhibition together produce brain waves (alpha ~10 Hz, gamma
   ~40 Hz); too little inhibition gives runaway synchrony (a seizure-like state); a sheet of neurons carries
   travelling waves and spirals.

## 3. Simulation 1 — Axons & Nerves

### Views

| View | What you see | What you do | What it measures |
|---|---|---|---|
| **Axon** (default) | One axon as a long glowing tube coloured by voltage; nodes of Ranvier when myelinated; a space–time plot (kymograph) underneath | Click anywhere to stimulate (one or two electrodes); set diameter, myelin, temperature; paint a stretch with TTX / lidocaine or *demyelinate* it | Conduction velocity (from the kymograph slope and from two recording electrodes), spike amplitude, refractory period, block / no block |
| **Nerve** | A nerve of 1 000+ fibres with a realistic spread of diameters (thin unmyelinated C fibres to thick myelinated A fibres), stimulated at one end | Set the stimulus strength and electrode distance | The **compound action potential** at a recording electrode: separate A and C humps, which spread apart with distance — the classic frog sciatic nerve lab, done on the GPU |

### Physics

- **Membrane:** the existing HH model (squid parameters, Q10 = 3 temperature scaling), unchanged and shared with
  the HH simulation.
- **Cable:** compartments of length Δx coupled by axial resistivity Rᵢ = 35.4 Ω·cm:
  C·∂V/∂t = (a / 2Rᵢ)·∂²V/∂x² − I_ion. Squid axon radius 238 µm (diameter 476 µm).
- **Myelinated fibres:** HH nodes (1 µm) separated by passive myelinated internodes (≈ 100 × fibre diameter)
  with membrane capacitance and leak reduced by the number of myelin wraps; demyelinating an internode restores
  bare-membrane capacitance and leak there. Kept deliberately simple (not the full double-cable McIntyre model);
  the claim tested is the scaling, not a specific nerve.
- **Reference numbers for tests:**
  - HH 1952 computed **18.8 m/s** (measured 21.2 m/s) for the squid axon at 18.3 °C; the twin must reproduce the
    computed value within 5 % once the grid is converged.
  - Unmyelinated speed ∝ √d: doubling the diameter multiplies the speed by √2 (tested within 3 %).
  - Myelinated speed ∝ d (Rushton), and myelinated ≫ unmyelinated at equal diameter.
  - Two spikes started at both ends annihilate; a second stimulus inside the refractory period fails to propagate.
  - A TTX-painted stretch longer than about one length constant blocks; a short one does not.

### Numerics (the hard part, decided up front)

- **Operator splitting** each step: (1) gates and ionic currents per compartment, Rush–Larsen as in the HH sim,
  fully parallel; (2) axial diffusion.
- Squid axon with Δx = 0.5 mm: diffusion coefficient a/(2RᵢC) ≈ 0.034 cm²/ms, so explicit diffusion is stable
  for Δt < Δx²/(2D) ≈ 0.037 ms; the HH step of 0.01 ms is well inside. **Explicit is fine for the squid axon.**
- Myelinated internodes (tiny capacitance) make explicit diffusion stiff. So diffusion is **implicit (backward
  Euler), solved with the Thomas algorithm, one GPU thread per fibre** — the nerve view has a thousand fibres, so
  the GPU stays busy, and each solve is exact. (NEURON solves the same tridiagonal system with the Hines
  algorithm.) The twin does the identical solve in 64-bit, and a test checks the explicit and implicit schemes
  agree where both are stable.
- Grid-convergence test: halving Δx and Δt changes the measured velocity by < 1 %.

## 4. Simulation 2 — Brain Rhythms

### Models

- **Izhikevich (2003)** neurons as the default: v′ = 0.04v² + 5v + 140 − u + I, u′ = a(bv − u), reset at 30 mV.
  Two equations per neuron, yet regular spiking, bursting and fast-spiking cells by parameter choice; his
  1 000-neuron network (800 excitatory, 200 inhibitory, random parameters per cell) produces alpha- and
  gamma-band rhythms — the reference experiment (also in Brian2 and ANNarchy examples, for cross-checks).
- **Leaky integrate-and-fire** for the **Brunel (2000)** network: 80 % excitatory, sparse random connections,
  relative inhibition g and external drive ν_ext; its phase diagram (synchronous regular, asynchronous
  irregular, synchronous irregular) is the textbook map of network states (Neuronal Dynamics, exercise 11).
- **PING gamma:** excitatory and inhibitory populations whose rhythm frequency is set by the inhibitory synapse
  decay time (Börgers & Kopell).
- Full HH neurons are *not* used for networks: the axon simulation already teaches the channels, and two-variable
  models let the GPU run 100× more cells.

### Views

| View | What you see | What it measures |
|---|---|---|
| **Raster** (default) | Every spike as a dot (neuron × time), excitatory and inhibitory in two colours, plus the population rate and an "EEG" trace | Mean rate, the power spectrum and its dominant frequency (alpha/beta/gamma band named), CV of inter-spike intervals, a synchrony index |
| **Cortex sheet** | 100 × 100 to 300 × 300 neurons on a sheet, mostly local connections, each glowing as it fires | Travelling waves and spiral waves; wave speed (measured from the sheet) |

### GPU design

- **Per step (0.5 ms, as Izhikevich):** (1) a neuron kernel integrates every cell and appends spikers to a
  compact list with an atomic counter; (2) a propagation kernel walks only the spikers' outgoing synapses (CSR
  sparse matrix) and adds their weights to the targets' input. Spike-driven ("push") costs rate × synapses, not
  N × synapses: 100 000 neurons × 10 Hz × 1 000 synapses ≈ 10⁹ adds/s, feasible on integrated graphics at a few
  hundred synapses per cell, and the reason this needs the GPU.
- **WGSL has no floating-point atomics**, so inputs accumulate as **fixed-point integers** (`atomicAdd` on i32).
  Integer addition is order-independent, which makes the GPU **deterministic** — and lets the tests compare it
  with the twin spike for spike over short runs.
- Synaptic delays as a ring buffer of input slots; connectivity generated from a seed, so a shared link
  reproduces the exact network.
- Read-backs are tiny: spike counts per millisecond (for the rate, spectrum and "EEG") and the raster's spikes
  for a sample of neurons, at ≤ 10 Hz.

### Measurements and how each is proven

| Readout | Proven by |
|---|---|
| Spike times | GPU vs twin: identical spike trains for the first ~200 ms of a 1 000-neuron network (fixed-point inputs, same seed) |
| Rhythm (dominant frequency) | Twin: the Izhikevich 2003 network's spectrum peaks in the alpha and gamma bands; PING frequency falls when the inhibitory decay time rises |
| Network state | Twin: Brunel network asynchronous irregular (CV > 0.8, low synchrony) vs synchronous regular (high synchrony) at the published (g, ν_ext) points |
| Wave speed | Twin: the sheet's wave speed scales with connection range and axonal delay as predicted |

## 5. Challenges (predict, observe, explain)

**Axons & Nerves**

1. **Bigger push, bigger spike?** Double the stimulus: the spike's height does not change (all-or-none).
2. **Thick or thin?** Double the diameter: the speed rises only √2 ≈ 1.4× (prediction options: 1×, 1.4×, 2×).
3. **Myelin.** Same diameter, myelinated: much faster, and the kymograph shows the spike regenerated at each node.
4. **Multiple sclerosis.** Demyelinate internodes one by one: conduction slows, then blocks; find the tipping point.
5. **Head-on.** Stimulate both ends: the spikes annihilate rather than pass through (refractory tail).
6. **The dentist's injection.** Paint lidocaine on a stretch: how long must it be to block?
7. **Read a real nerve.** In the compound action potential, which hump is the fast fibres, and why do the humps
   separate as the recording electrode moves away?

**Brain Rhythms**

1. **Find the gamma rhythm.** Measure the dominant frequency of the default network (≈ 40 Hz band).
2. **Without brakes.** Weaken inhibition: the network locks into synchrony (a seizure-like state); measure the
   synchrony index jump.
3. **Tune the clock.** Lengthen the inhibitory synapse decay: predict whether the rhythm speeds up or slows down.
4. **Order from noise.** Move a Brunel network from regular to irregular firing by changing g.
5. **Spiral waves.** On the cortex sheet, break a travelling wave to make a spiral, and measure its period.

## 6. Shareable settings

URL state for everything that defines the run: view, axon diameter, myelin, temperature, painted regions, electrode
positions; network model, size, seed, g, drive, synaptic time constants. Links reproduce the exact network
because connectivity comes from the seed.

## 7. Testing

- Twins (`cable.ts`, `network.ts`) with unit tests for every number above and in the challenge texts.
- GPU tests: the cable kernels vs the twin (explicit and implicit, including fibre counts around workgroup sizes);
  the network kernels spike-for-spike vs the twin; fixed-point accumulation exact; CSR generation identical on CPU
  and GPU.
- e2e: both simulations render without GPU errors; velocity measured in the browser; myelin and TTX effects;
  compound action potential humps; rhythm frequency in the browser; links and hostile links; phone layout; axe;
  reduced motion; challenge flows.

## 8. Phases and estimate (≈ 7–9 days)

| Phase | Work | Done when |
|---|---|---|
| 1. Cable physics (1½ days) | `cable.ts` twin (explicit + implicit), HH velocity, √d, myelin, collision, TTX; GPU kernels and tests | 18.8 m/s and the scalings proven, GPU = twin |
| 2. Axon view (1½ days) | Tube renderer, kymograph, electrodes, painting tools, readouts, URL state | velocity measured in the browser; a painted block reproduces from its link |
| 3. Nerve view (1 day) | Fibre-diameter distributions, compound action potential, recording electrode | A and C humps separate with distance |
| 4. Network core (1½ days) | `network.ts` twin (Izhikevich, LIF), CSR generation, neuron and push kernels with fixed-point inputs, GPU tests | spike-for-spike agreement; Izhikevich rhythms in the twin |
| 5. Brain Rhythms views (1½ days) | Raster, rate, spectrum, "EEG", cortex sheet with waves, URL state | rhythms and waves measured in the browser |
| 6. Challenges and polish (1–2 days) | 12 challenges, catalog/home plates, posters, previews, docs, e2e | `pnpm verify` and full e2e green on prod, dev and static |

## 9. Decisions needed

1. **Two simulations or one.** Recommended: two (Axons & Nerves, Brain Rhythms). They teach different levels, and
   each stays simple to use. Alternative: one "Neuroscience lab" with four views.
2. **Build order.** Recommended: Axons first (reuses the HH code, smaller, a strong standalone result), then Brain
   Rhythms.
3. **Default network size.** Recommended 10 000 neurons (smooth on integrated graphics), with 1 000 for the
   reference experiment and 100 000 as "high detail".
4. **Home page.** Seven plates in the scrolling row (already supports more than four).

## Sources

- Hodgkin & Huxley (1952), *J Physiol* 117:500; computed 18.8 m/s vs measured 21.2 m/s: <https://neurowiki.case.edu/wiki/Reading_Cables_II>; squid axon Rᵢ = 35.4 Ω·cm and 300–600 µm diameters: <https://en.wikipedia.org/wiki/Squid_giant_axon>
- Myelin and node geometry vs speed (Rushton scaling, internode and node length): <https://elifesciences.org/articles/23329>, <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5766232/>
- Numerical integration of cables (backward Euler, Crank–Nicolson, Hines): <https://www.neuron.yale.edu/ftp/ted/book/chap5.pdf>, <https://neuron.yale.edu/ftp/ted/neuron/numerical_integration.pdf>
- Izhikevich (2003), "Simple model of spiking neurons": <https://courses.cs.washington.edu/courses/cse528/07sp/izhi1.pdf>; Brian2 example: <https://brian2.readthedocs.io/en/2.9.0/examples/frompapers.Izhikevich_2003.html>
- Brunel (2000), sparse E–I networks: <https://pubmed.ncbi.nlm.nih.gov/10809012/>; teaching exercise: <https://neuronaldynamics-exercises.readthedocs.io/en/latest/exercises/brunel-network.html>
- PING gamma (Börgers & Kopell): <https://pmc.ncbi.nlm.nih.gov/articles/PMC3276541>
- GPU spiking simulators: <https://arxiv.org/pdf/2007.14236> (GeNN), NeMo: <https://centaur.reading.ac.uk/id/eprint/30301>
- Existing teaching tools: Neuronify <https://www.eneuro.org/content/4/2/ENEURO.0022-17.2017>; NEST Desktop <https://nest-desktop.readthedocs.io/en/latest/about/abstract.html>; HHsim / Neurons in Action exercises <https://www.st-andrews.ac.uk/~wjh/neurosim/TutorialV5_6/tutorials.html>, <https://nest-desktop.readthedocs.io/en/dev/lecturer/neuron-models/hodgkin-huxley-action-potential.html>
- Misconception that the spike "jumps" over myelin: <https://www.revisiondojo.com/student-faq/how-action-potential-travels-along-axon>
