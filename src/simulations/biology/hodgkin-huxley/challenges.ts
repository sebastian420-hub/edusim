import type { Challenge } from "@/lib/challenges";
import type { HHReadouts, HHSettings } from "./settings";

/** Guided experiments for the Hodgkin–Huxley neuron. Numbers come from the model (see hh.test.ts). */
export const HH_CHALLENGES: Challenge<HHSettings, HHReadouts>[] = [
  {
    id: "threshold",
    title: "Find the firing threshold",
    prompt: "Inject a steady current and slowly turn it up from zero. When does the neuron start firing repeatedly?",
    setup: { pulse_mode: 0, I_inj: 0, g_Na: 120, g_K: 36, temperature: 6.3, showCurve: true },
    prediction: {
      question: "As you raise the current from 0, the neuron will…",
      options: [
        { label: "fire as soon as there is any current at all" },
        { label: "stay silent until a threshold (about 6 µA/cm²), then jump to a fast rhythm", correct: true },
        { label: "fire slowly at first, speeding up smoothly from 0 Hz" },
      ],
    },
    goal: {
      description: "Set the current so the neuron fires, using as little current as you can (within 1.5 µA/cm² of the threshold).",
      check: ({ params, readouts }) =>
        params.pulse_mode === 0 && readouts.rate > 0 && readouts.fi.rheobase !== undefined && params.I_inj <= readouts.fi.rheobase + 1.5,
    },
    hint: "Watch the firing-rate curve in the sidebar: the dot shows where you are. Nothing happens until the curve leaves zero.",
    explanation:
      "Below the threshold the membrane just settles to a slightly depolarised level. Just above it, sodium channels open faster than potassium channels can catch up, so every cycle produces a full spike and the neuron jumps straight to roughly 50 spikes per second — it can’t fire slowly. This all-or-nothing onset is a hallmark of the Hodgkin–Huxley neuron.",
  },
  {
    id: "ttx",
    title: "Silence it with a toxin",
    prompt: "The neuron is firing steadily. Tetrodotoxin (pufferfish toxin) blocks sodium channels. Can you stop the spikes without touching the current?",
    setup: { pulse_mode: 0, I_inj: 15, g_Na: 120, g_K: 36, temperature: 6.3 },
    prediction: {
      question: "If you block most of the sodium channels but keep injecting the same current…",
      options: [
        { label: "the spikes get smaller but keep coming" },
        { label: "the spikes disappear — only a slow, small bump remains", correct: true },
        { label: "the neuron fires even faster" },
      ],
    },
    goal: {
      description: "Keep the current at 15 µA/cm² and K⁺ conductance at 36, and lower Na⁺ conductance until the neuron stops firing.",
      check: ({ params, readouts }) => params.pulse_mode === 0 && params.I_inj >= 15 && params.g_K === 36 && readouts.rate === 0,
    },
    hint: "Drag the Na⁺ conductance slider (TTX) down. Watch the green voltage trace flatten out.",
    explanation:
      "The upstroke of a spike is a sodium current: depolarisation opens Na⁺ channels, which depolarise the cell further. With most of those channels blocked the positive feedback can’t start, so the same current only produces a passive bump. That’s why TTX is so deadly — nerves can no longer carry signals.",
  },
  {
    id: "temperature",
    title: "Warm it up",
    prompt: "Same neuron, same current — only the temperature changes. What does heat do to the firing rate?",
    setup: { pulse_mode: 0, I_inj: 10, g_Na: 120, g_K: 36, temperature: 6.3 },
    prediction: {
      question: "Raising the temperature from 6°C to about 16°C will make the neuron fire…",
      options: [{ label: "more slowly" }, { label: "about the same" }, { label: "much faster (more than twice as fast)", correct: true }],
    },
    goal: {
      description: "Reach a firing rate of at least 150 Hz by changing only the temperature (current 10, conductances unchanged).",
      check: ({ params, readouts }) =>
        params.pulse_mode === 0 && params.I_inj === 10 && params.g_Na === 120 && params.g_K === 36 && readouts.rate >= 150,
    },
    hint: "Drag Temperature to the right. The firing rate readout in the corner shows the effect.",
    explanation:
      "Every channel gate opens and closes faster when it’s warmer — in the model, kinetics speed up threefold for each 10°C (Q10 = 3). Faster gates mean faster spikes and shorter recovery, so the rate climbs from about 68 Hz at 6°C to about 160 Hz at 16°C. This is why cold-blooded animals slow down in the cold.",
  },
];
