import type { Challenge } from "@/lib/challenges";
import type { AxonReadout } from "./measure";
import type { AxonSettings } from "./settings";
import { effectiveDiameter } from "./settings";
import type { AxonStats, CapReadout } from "./sim";

/** What challenge goals can observe: the readouts of the last *finished* sweep. */
export interface AxonReadouts {
  view: AxonSettings["view"] | null;
  axon: AxonReadout | null;
  nerve: CapReadout | null;
}

export function readoutsFor(stats: AxonStats | null): AxonReadouts {
  if (!stats) return { view: null, axon: null, nerve: null };
  return stats.view === "axon" ? { view: "axon", axon: stats.done, nerve: null } : { view: "nerve", axon: null, nerve: stats.done };
}

const squid = { view: "axon", diameter: 476, myelin: false, temp: 18.3, stim: 2, pulses: "single", drug: "none", myelinLeft: 1, from: 0.45, to: 0.55, e1: 0.3, e2: 0.7 } as const;

/** Guided experiments. Every number in the texts is checked against the CPU twin in challenges.test.ts. */
export const AXON_CHALLENGES: Challenge<AxonSettings, AxonReadouts>[] = [
  {
    id: "all-or-none",
    title: "Bigger push, bigger spike?",
    prompt: "The axon fires when it is stimulated just above its threshold. Now hit it at least twice as hard, and read the spike's height at electrode 2.",
    setup: { ...squid, stim: 1.2 },
    prediction: {
      question: "With a stimulus twice as strong, the spike that arrives at electrode 2 will be…",
      options: [{ label: "about twice as tall" }, { label: "taller, but not twice" }, { label: "exactly the same height", correct: true }],
    },
    goal: {
      description: "Fire a sweep with a stimulus of at least 2.4× threshold and measure the spike at electrode 2.",
      check: ({ params, readouts: r }) => params.view === "axon" && params.stim >= 2.4 && r.axon?.outcome === "conducted" && r.axon.peak2 !== undefined,
    },
    hint: "Raise Stimulus to 2.4 or more. Compare the peak readout with the one before: it barely moves.",
    explanation:
      "The stimulus only has to push the membrane past threshold; after that, the sodium channels take over and every spike is made by the axon itself, from its own ion gradients. So the spike is all-or-none: its height (about +25 mV here) does not depend on how hard you pushed. Nerves signal 'stronger' with more spikes per second, not bigger ones.",
  },
  {
    id: "diameter",
    title: "Thick or thin?",
    prompt: "The squid's giant axon is half a millimetre thick, which makes it fast. Measure its speed between the two electrodes, then double its diameter.",
    setup: { ...squid },
    prediction: {
      question: "Doubling the diameter, the conduction speed becomes…",
      options: [{ label: "the same" }, { label: "about 1.4× faster", correct: true }, { label: "2× faster" }],
    },
    goal: {
      description: "Measure the speed of a bare axon at least 900 µm thick.",
      check: ({ params, readouts: r }) => params.view === "axon" && !params.myelin && params.diameter >= 900 && r.axon?.velocity !== undefined && r.axon.velocity > 20,
    },
    hint: "Set Diameter to 952 µm (twice 476). The speed shows under Measure after the sweep.",
    explanation:
      "A thicker axon has less internal resistance, so the local currents reach further ahead and charge the membrane sooner. But it also has more membrane to charge. Together the speed grows only as the square root of the diameter: twice as thick, √2 ≈ 1.41× faster (18.5 → 26 m/s). That is why the squid needed a giant axon to escape quickly, and why our brains could not work that way.",
  },
  {
    id: "myelin",
    title: "Wrap it in myelin",
    prompt: "A bare 10 µm fibre is slow. Vertebrates wrap theirs in myelin, leaving short gaps (nodes of Ranvier). Switch myelin on and measure the speed.",
    setup: { ...squid, diameter: 10 },
    prediction: {
      question: "Same 10 µm fibre, now myelinated. Its speed becomes…",
      options: [{ label: "about the same: myelin is just insulation" }, { label: "about twice as fast" }, { label: "about ten times as fast", correct: true }],
    },
    goal: {
      description: "Measure a myelinated 10 µm fibre conducting faster than 20 m/s.",
      check: ({ params, readouts: r }) => params.view === "axon" && params.myelin && effectiveDiameter(params) === 10 && (r.axon?.velocity ?? 0) > 20,
    },
    hint: "Tick Myelin. Watch the kymograph: the spike line becomes a staircase, jumping from node to node.",
    explanation:
      "Myelin cuts the membrane's capacitance and leak about a hundredfold, so the local current racing inside the axon charges the next node almost at once instead of leaking out or charging every bit of membrane on the way. The spike is regenerated only at the nodes: saltatory conduction. It does not leap through empty space; current flows along the inside the whole time. Here: about 2.7 m/s bare, 25 m/s myelinated, from a fibre 50 times thinner than the squid's.",
  },
  {
    id: "multiple-sclerosis",
    title: "Multiple sclerosis",
    prompt: "In multiple sclerosis, the immune system strips myelin. Thin the myelin on the marked stretch step by step and find where the signal stops getting through.",
    setup: { ...squid, diameter: 10, myelin: true, from: 0.45, to: 0.55, myelinLeft: 1 },
    prediction: {
      question: "With half of the myelin gone on that stretch, the signal will…",
      options: [{ label: "stop dead" }, { label: "get through, a little later", correct: true }, { label: "get through faster" }],
    },
    goal: {
      description: "Find a myelin level on the stretch that blocks conduction.",
      check: ({ params, readouts: r }) => params.view === "axon" && params.myelin && params.drug === "none" && params.myelinLeft < 1 && r.axon?.outcome === "blocked",
    },
    hint: "Lower Myelin left: 50 %, 10 %, 5 %, 0 %. Watch when the spike arrives at electrode 2, and when it no longer arrives at all.",
    explanation:
      "Thinner myelin leaks more charge and has more capacitance to fill, so each node takes longer to reach threshold: conduction slows (here from 1.8 to about 3 ms to reach the end at 5 % myelin). Strip it completely and the nodes can no longer charge the bare stretch: the spike dies. Slowed and blocked conduction is what makes MS symptoms come and go, and why they often worsen in the heat.",
  },
  {
    id: "collision",
    title: "Head-on",
    prompt: "Stimulate the axon at both ends at the same moment. Two spikes race towards each other.",
    setup: { ...squid, pulses: "both" },
    prediction: {
      question: "When the two spikes meet in the middle, they…",
      options: [{ label: "pass through each other, like waves" }, { label: "annihilate", correct: true }, { label: "bounce back" }],
    },
    goal: {
      description: "Fire both ends and record exactly one spike at each electrode.",
      check: ({ params, readouts: r }) => params.view === "axon" && params.pulses === "both" && r.axon?.spikes1 === 1 && r.axon.spikes2 === 1,
    },
    hint: "Choose Both ends under Stimulus. Look at the kymograph: an X that does not cross.",
    explanation:
      "Behind every spike the membrane is refractory for a few milliseconds: the sodium channels are inactivated and the potassium channels still open. When two spikes meet, each runs into the other's refractory tail, and both die. That is also why a spike normally travels only forwards. Water waves pass through each other because nothing has to be recharged; a spike is not a wave in that sense.",
  },
  {
    id: "lidocaine",
    title: "The dentist's injection",
    prompt: "Lidocaine blocks most sodium channels where it is injected. Here a very short stretch is numbed and the signal still gets through. How long must the numbed stretch be to stop it?",
    setup: { ...squid, drug: "lidocaine", from: 0.45, to: 0.47 },
    prediction: {
      question: "To block the signal, the numbed stretch must be at least about…",
      options: [{ label: "a single point" }, { label: "one length constant (about 1 cm here)", correct: true }, { label: "the whole axon" }],
    },
    goal: {
      description: "Block conduction with lidocaine on a stretch no longer than a fifth of the axon.",
      check: ({ params, readouts: r }) => params.view === "axon" && params.drug === "lidocaine" && Math.abs(params.to - params.from) <= 0.2 && r.axon?.outcome === "blocked",
    },
    hint: "Drag the edges of the shaded stretch (or use the sliders) to lengthen it, a little at a time.",
    explanation:
      "Even where it cannot fire, the axon still conducts passively: the voltage spreads, fading over a length constant (about 1 cm for the squid axon). A short numbed stretch is bridged: the spike's current leaks across and re-ignites the healthy membrane beyond. Make it about a length constant long and too little arrives. Dentists inject around the nerve so that enough of it is bathed in the drug.",
  },
  {
    id: "compound",
    title: "Read a real nerve",
    prompt: "A nerve holds thousands of fibres of different sizes. Record the summed signal (the compound action potential) 0.5 cm from the stimulus, then move the electrode to 1.5 cm.",
    setup: { view: "nerve", fibres: 300, nerveStim: 6, distance: 0.5, temp: 18.3 },
    prediction: {
      question: "Further from the stimulus, the fast (A) and slow (C) waves will be…",
      options: [{ label: "closer together" }, { label: "further apart", correct: true }, { label: "the same distance apart" }],
    },
    goal: {
      description: "Record both the A and the C wave at least 1.5 cm from the stimulus.",
      check: ({ params, readouts: r }) => params.view === "nerve" && params.distance >= 1.5 && r.nerve?.aLatency !== undefined && r.nerve.cLatency !== undefined,
    },
    hint: "Move Recording distance to 1.5 cm. The latencies of both waves show under Measure.",
    explanation:
      "Every fibre carries its own spike at its own speed: myelinated A fibres at about 10–35 m/s, thin unmyelinated C fibres under 1 m/s. At the stimulus they all start together; the further you record, the more the fast spikes pull ahead, so the waves spread apart in time. From each wave's latency and the distance you can read off the speed of each fibre group, which is how physiologists classify nerve fibres. (In the body, A fibres carry touch and sharp pain, C fibres slow burning pain.)",
  },
];
