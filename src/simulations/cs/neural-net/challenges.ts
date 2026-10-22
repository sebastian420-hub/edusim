import type { Challenge } from "@/lib/challenges";
import type { NetSettings } from "./settings";
import type { CurvePoint, NetStats } from "./sim";

/** What challenge goals can observe. */
export interface NetReadouts {
  view: NetSettings["view"] | null;
  steps: number;
  /** Hidden layers of the network actually being trained. */
  hidden: number;
  now: CurvePoint | null;
  /** How different the hidden neurons are (0 = clones). */
  spread: number | null;
}

export function readoutsFor(stats: NetStats | null): NetReadouts {
  if (!stats) return { view: null, steps: 0, hidden: 0, now: null, spread: null };
  return { view: stats.view, steps: stats.steps, hidden: stats.hidden, now: stats.now, spread: stats.view === "playground" ? stats.spread : null };
}

/** The plain Playground training settings every challenge starts from. */
const plain = { view: "playground", points: 400, noise: 0, train: 0.5, features: 0b11, lr: 0.1, batch: 16, mom: 0.9, l2: 0, init: "random", seed: 1, speed: 5 } as const;

const playground = (p: NetSettings, r: NetReadouts) => p.view === "playground" && r.view === "playground" && r.now !== null;

/** Guided experiments. Every number in the texts is checked against the CPU twin in challenges.test.ts. */
export const NET_CHALLENGES: Challenge<NetSettings, NetReadouts>[] = [
  {
    id: "line-xor",
    title: "A straight line can't do it",
    prompt:
      "XOR: blue where x and y have the same sign, orange where they differ. The network has no hidden layer, so all it can do is draw one straight line across the plane. Press Play and watch.",
    setup: { ...plain, data: "xor", layers: [], act: "tanh" },
    prediction: {
      question: "Trained for as long as you like, a straight-line model gets this many test points right:",
      options: [{ label: "100 %: it just needs time" }, { label: "about 75 %: three quadrants out of four" }, { label: "about 50 %: no better than a coin toss", correct: true }],
    },
    goal: {
      description: "Keep only x and y as inputs, and reach 95 % test accuracy on XOR by giving the network a hidden layer.",
      check: ({ params, readouts: r }) => playground(params, r) && params.data === "xor" && params.features === 0b11 && r.hidden >= 1 && params.layers.length >= 1 && r.now!.testAcc >= 0.95,
    },
    hint: "Under Network, press “+ layer”. Two to four neurons are enough; press Play again.",
    explanation:
      "One line can never split XOR: whichever way it goes, it cuts through both colours. Worse, the loss pulls the line two ways at once, so it settles at ‘50 % everywhere’ (a loss of ln 2 ≈ 0.69, the loss of a coin toss). A hidden layer draws several lines and combines them: each hidden neuron in the diagram shows its own line, and the output adds them up into the corners.",
  },
  {
    id: "features",
    title: "Or give it the right input",
    prompt: "Same XOR, still no hidden layer. Instead of more neurons, give the straight-line model a better input to look at.",
    setup: { ...plain, data: "xor", layers: [], act: "tanh" },
    prediction: {
      question: "Which one extra input lets a straight-line model solve XOR?",
      options: [{ label: "x²" }, { label: "x·y", correct: true }, { label: "sin x" }],
    },
    goal: {
      description: "With no hidden layer, reach 95 % test accuracy on XOR.",
      check: ({ params, readouts: r }) => playground(params, r) && params.data === "xor" && params.layers.length === 0 && r.hidden === 0 && r.now!.testAcc >= 0.95,
    },
    hint: "Under Inputs, switch on x·y and press Play. Try the Circle data too, with x² and y².",
    explanation:
      "x·y is positive exactly where x and y share a sign, so a single weight on it separates XOR perfectly; for the circle, x² + y² is the distance from the centre squared. A straight line in the right inputs is a curve in the original ones. Choosing such inputs by hand is called feature engineering. What a hidden layer does is learn its own inputs from the data, which is why deep networks replaced most hand-made features.",
  },
  {
    id: "big-step",
    title: "Too big a step",
    prompt:
      "The learning rate is the size of each step downhill. Here a circle is learnt at a learning rate of 0.1. Press Play, then make the steps a hundred times bigger and watch the loss and the step probe.",
    setup: { ...plain, data: "circle", layers: [8], act: "tanh" },
    prediction: {
      question: "With a learning rate of 10 instead of 0.1, the network will…",
      options: [{ label: "learn a hundred times faster" }, { label: "learn about as well, just more jumpily" }, { label: "break: overshoot every time and end up guessing", correct: true }],
    },
    goal: {
      description: "With a random start and a learning rate of at least 3, train the circle for 300 steps or more and see training accuracy stuck at 60 % or below.",
      check: ({ params, readouts: r }) => playground(params, r) && params.data === "circle" && params.init === "random" && params.lr >= 3 && r.steps >= 300 && r.now!.trainAcc <= 0.6,
    },
    hint: "Set the learning rate slider to 10 and press Reset. In the step probe, the dot (where the step landed) sits higher than the start: every step goes uphill.",
    explanation:
      "Gradient descent only knows the slope where it stands. A small step follows the slope downhill; a step much longer than the valley is wide jumps over it and lands higher up the other side, again and again. For a bowl-shaped loss there is an exact limit: the step must be smaller than 2 divided by the bowl's steepest curvature, or every step makes things worse. The same rule limits the time step in the physics simulations: too large a step and the numbers explode. Here the neurons saturate and the network ends up confidently wrong.",
  },
  {
    id: "overfit",
    title: "Memorise or learn?",
    prompt:
      "Only 40 training points, a fifth of their labels wrong on purpose, and a big network. The rings are test points it never trains on. Press Play and compare the two loss curves.",
    setup: { ...plain, data: "spiral", noise: 0.2, train: 0.1, layers: [16, 16], act: "relu", lr: 0.03 },
    prediction: {
      question: "As training goes on, the test loss will…",
      options: [{ label: "keep falling with the training loss" }, { label: "level off a little higher" }, { label: "turn and rise while the training loss keeps falling", correct: true }],
    },
    goal: {
      description: "With at most 20 % of the points for training, reach 95 % training accuracy while test accuracy stays at least 20 points lower.",
      check: ({ params, readouts: r }) => playground(params, r) && params.train <= 0.2 && r.now!.trainAcc >= 0.95 && r.now!.testAcc <= r.now!.trainAcc - 0.2,
    },
    hint: "Just let it train (a few thousand steps). Watch the map grow little islands around single wrong points.",
    explanation:
      "With about 300 weights and only 40 points, the network can fit every point, including the ones labelled wrong: it memorises instead of finding the spiral. Its training score is nearly perfect and its test score is poor. This is overfitting, and it is why a model is always judged on data it never trained on. The cures are more data (raise the training share), a smaller network, or a pull towards small weights (the L2 setting).",
  },
  {
    id: "zeros",
    title: "Why start at random?",
    prompt: "Every weight in this network starts at exactly 0, a tidy-looking choice. Press Play and look at the hidden neurons in the diagram.",
    setup: { ...plain, data: "xor", layers: [4, 4], act: "tanh", init: "zeros" },
    prediction: {
      question: "Starting from all zeros, the network will learn XOR…",
      options: [{ label: "normally" }, { label: "slowly at first, then normally" }, { label: "not at all", correct: true }],
    },
    goal: {
      description: "Keep the same network (two layers of 4) and reach 95 % test accuracy on XOR.",
      check: ({ params, readouts: r }) => playground(params, r) && params.data === "xor" && params.layers.length === 2 && params.layers.every((n) => n === 4) && r.now!.testAcc >= 0.95,
    },
    hint: "Under Network, switch Start to Random.",
    explanation:
      "Neurons that start identical get identical updates, so they stay clones for ever (the spread readout stays exactly 0): four neurons doing the work of one. With all-zero weights it is worse still, since no error reaches the hidden layers at all. A random start breaks the symmetry, so each neuron can specialise. That is why every real network starts from random weights, scaled to the layer size (Glorot or He initialisation).",
  },
  {
    id: "deep",
    title: "Deep and stuck",
    prompt: "Six hidden layers with the sigmoid activation, the classic choice of the 1990s. Press Play.",
    setup: { ...plain, data: "circle", layers: [4, 4, 4, 4, 4, 4], act: "sigmoid", lr: 0.03 },
    prediction: {
      question: "Compared with one hidden layer, six layers of sigmoids will learn the circle…",
      options: [{ label: "better: more layers, more power" }, { label: "about the same" }, { label: "hardly at all", correct: true }],
    },
    goal: {
      description: "Keep six hidden layers and reach 90 % test accuracy on the circle.",
      check: ({ params, readouts: r }) => playground(params, r) && params.data === "circle" && params.layers.length === 6 && r.hidden === 6 && r.now!.testAcc >= 0.9,
    },
    hint: "Watch “Gradient by layer” under Measure: the first layers get almost nothing. Try tanh.",
    explanation:
      "Backpropagation multiplies the error by each layer's slope on its way back. A sigmoid's slope is at most ¼, so after six layers the first layer receives less than a thousandth of the error and barely moves: the vanishing gradient. tanh's slope reaches 1, and ReLU's is exactly 1 wherever it is active, which is a large part of why deep learning took off only once ReLU replaced the sigmoid.",
  },
  {
    id: "digits",
    title: "Read my handwriting",
    prompt:
      "Real handwriting: 10 000 digits to learn from and 2 000 never shown during training. Start with a straight-line model, which gives every pixel one vote per digit. Press Play.",
    setup: { view: "digits", dnet: "linear", dlr: 0.05, dbatch: 32, seed: 1, speed: 5 },
    prediction: {
      question: "A straight-line model reads this many digits out of ten correctly:",
      options: [{ label: "about 1 (it's just a line)" }, { label: "about 5" }, { label: "about 9", correct: true }],
    },
    goal: {
      description: "Reach 92 % test accuracy on the handwritten digits.",
      check: ({ params, readouts: r }) => params.view === "digits" && r.view === "digits" && r.now !== null && r.now.testAcc >= 0.92,
    },
    hint: "A straight line levels off near 88 %. Choose “One hidden layer” and let it train for a few passes over the data.",
    explanation:
      "Even a straight-line model reads about 9 digits in 10: the templates above show the blurry ‘average digit’ it learnt for each class. But one template per digit cannot cover all the ways people write a 7, so it levels off near 88 %. A hidden layer learns 128 smaller strokes and combines them, and passes 92 %. Draw your own digit in the pad: the same network, the same arithmetic, reads it.",
  },
];
