# How AI Learns (neural network) — research and implementation plan

> **Status: shipped** with both views and all seven challenges, using the recommended decisions (catalog entry renamed
> "How AI Learns" and the biology duplicate removed; a 12 000-digit MNIST subset with credit; up to 6 × 32 neurons on the
> plane). Changes from the plan, found while building it and all covered by tests: (1) a straight-line model on XOR
> ends near **50 %**, not 75 %: cross-entropy pulls it to "½ everywhere" (loss ln 2), so challenge 1 asks for that;
> (2) "too big a step" uses tanh, where η = 10 leaves the circle at ≤ 60 % for every seed tested (ReLU nets just die),
> and its goal reads accuracy instead of a 10× loss; (3) the digits measure ~88 % for the best straight line and
> 92–94 % for one hidden layer of 128 on this subset, so challenge 7's goal is 92 %; (4) "deep and stuck" is solved
> with tanh (six ReLU layers of 4 sometimes die); (5) the spiral has 1.25 turns and the default network is two ReLU
> layers of 16 at η = 0.03, which untangles it for every seed tested; (6) the decision map is 160² (and 24² per neuron),
> interpolated; (7) the gradient check uses h = 10⁻⁴ (10⁻⁵ is rounding-limited for tiny gradients); (8) the digit
> data ship as one gzip file decompressed in the browser (`DecompressionStream`) instead of a PNG sprite.

The seventh simulation: catalog id `neural-net-viz`, route `/cs/neural-net-viz`, already listed as planned ("Neural
Network Decision Boundary"). Proposed title **"How AI Learns"** (decision 1 below). Same bar as the other six: a 64-bit
TypeScript twin of every shader, GPU-vs-twin tests, shareable settings, measurements a test proves correct, guided
challenges, e2e scenarios, a poster and a live home-page preview.

Why this one (from the earlier research and discussion): it is the technology the user works with every day, it is the
most-used kind of simulation in computing education, and it carries the project's own thread — *a learning run is an
integration with a step size, and too large a step blows up here exactly as it did in the N-body and pendulum sims*.

## 1. What students should get out of it

From how neural networks are actually taught (TensorFlow Playground assignments in a dozen university courses, the
cs231n notes, the Boston University "gradients by hand, then check against finite differences" activity, the CTU
double-descent lab, 3Blue1Brown's series) and from the misconceptions those courses fight:

1. **Learning is error-shrinking.** A network is a function with adjustable numbers (weights). Training measures how
   wrong it is on examples (the loss) and nudges every weight downhill: *gradient descent*. Nothing is "understood";
   the error gets smaller.
2. **Backpropagation is the chain rule.** Each layer passes the blame for the error backwards. Whether it is right is
   checkable: wiggle one weight, see the loss change, compare with what the chain rule predicted (a *gradient check*).
3. **The step size decides everything.** Too small and nothing happens; too large and the loss *rises* every step and
   then explodes (the same instability as a too-large time step in the physics sims: the threshold is 2 / sharpness).
4. **Features and depth.** A straight line cannot separate XOR or a circle. Either give the model the right features
   (x², y², x·y) or give it a hidden layer and let it invent them. Depth with the wrong activation (six sigmoid layers)
   stops learning because the gradient fades on the way back: *vanishing gradients*.
5. **Fitting is not learning.** With too little data and too much network, the training error reaches zero while the
   error on *unseen* data gets worse: *overfitting*. The held-out test set is the only honest score.
6. **Random starts matter.** Start every weight at zero and every hidden neuron stays identical forever (symmetry);
   start them randomly and they specialise.
7. **It works on real data.** The same mechanism, trained on handwritten digits in seconds on the GPU, reads a digit the
   student draws, and says how sure it is. A plain linear model already reads about nine in ten; a hidden layer
   reads more and shows what it got wrong.

## 2. Product shape: one simulation, two views of the same engine

| View | What you see | What you do | What it measures |
|---|---|---|---|
| **Playground** (default) | Coloured points on the plane; the network's decision map underneath them (256², recomputed every frame); the network as a diagram; train and test loss curves; a *step probe* (the loss along the direction of the current step, with where the step actually landed) | Pick a dataset (two blobs, circle, XOR, spiral), noise, train/test split; choose input features; add or remove layers and neurons; choose activation, learning rate, batch size, momentum, L2; Play, Step, Reset (re-rolls the weights) | Train/test loss and accuracy, step count, gradient size per layer, whether the last step went uphill, per-neuron weight spread (symmetry) |
| **Digits** | A grid of real handwritten digits with the network's guess under each (wrong ones highlighted); accuracy curves; the ten "templates" of a linear model or the hidden-layer weight images; a confusion matrix; a draw pad | Choose the network (none / one hidden layer / two), learning rate, batch size; Play; draw a digit and watch the ten probability bars | Train/test accuracy, epochs, the confusion matrix, the probabilities for the drawn digit |

Both views run the **same training kernels** on the same engine; only the data (500–2 000 points in two features vs.
12 000 images of 784 pixels) and the renderer differ.

This is more than TensorFlow Playground in four ways: the GPU allows bigger networks (up to 6 × 32 neurons vs 6 × 8)
and a full-resolution decision map; real handwriting with a draw pad; measured predict-then-test challenges; and the
*step probe*, which makes the step-size lesson visible instead of leaving the student with a diverging loss curve.
The search found no browser playground combining live GPU training, a decision map and handwriting.

## 3. Model

- **Network:** a fully connected multilayer perceptron. Inputs: 2D features (any subset of x, y, x², y², x·y, sin x,
  sin y, as in Playground) or 784 pixels. Hidden layers: 0–6, each 1–32 neurons, activation tanh / ReLU / sigmoid /
  linear (one choice for the whole net). Output: one logit with sigmoid + binary cross-entropy for the plane (two
  classes), ten logits with softmax + cross-entropy for digits. Cross-entropy, not Playground's squared error, because it
  is what every real classifier uses and because its gradient on the logits (prediction − label) is the cleanest
  backprop lesson.
- **Initialisation:** Glorot-uniform (tanh, sigmoid) or He-uniform (ReLU) from a seeded PRNG; a "zeros" option for
  the symmetry lesson. The seed is part of the settings, so a link reproduces a training run bit for bit on the
  twin.
- **Optimiser:** mini-batch stochastic gradient descent with optional momentum and L2 weight decay. Adam is left out
  of v1 (it hides the step-size lesson); it can be added later as a second "optimiser" choice.
- **Batches:** the training set is shuffled once per epoch with the seeded PRNG on the CPU; the batch's sample indices
  are uploaded (a few hundred bytes). The twin does exactly the same, so GPU and twin see the same batches.
- **Datasets (plane):** Playground's four generators reimplemented with the seeded PRNG (two Gaussian blobs; circle
  and ring; XOR quadrants; two interleaved spirals), 100–2 000 points, label noise 0–50 %, train share 10–90 %.
- **Dataset (digits):** a fixed 12 000-image subset of MNIST (10 000 train, 2 000 test; 28 × 28 greyscale), shipped
  as one PNG sprite plus a label file (≈ 2 MB, estimated from the 10 MB tfjs sprite of 65 000 images) under
  `public/data/`, built by a script from the original files. MNIST is CC BY-SA 3.0 (Yann LeCun, Corinna Cortes,
  Christopher Burges), so the app credits it in the view and the README. Fashion-MNIST (MIT) is the fallback if the
  share-alike term is unwanted (decision 2).
- **Draw pad → MNIST form:** the drawing is cropped to its bounding box, scaled so the longer side is 20 px, and
  placed in a 28 × 28 image with its centre of mass at the centre, which is how MNIST itself was normalised; without
  this the network misreads off-centre digits (the Harley demo notes the same).

## 4. GPU design

A training step is a fixed sequence of small compute dispatches over the **whole mini-batch at once**; nothing is
drawn by the engine, and no step depends on an atomic (sums run in a fixed order, so the result is deterministic and
testable against the twin).

- **Buffers (compat-mode budget: 4 storage buffers per stage):** `params` (all weights and biases packed with
  per-layer offsets), `acts` (pre- and post-activation values for every sample in the batch and every layer),
  `deltas` (the backpropagated errors, same layout), `grads` (same layout as `params`, plus the momentum buffer behind
  it), `data` (the dataset: features as f32 for the plane, pixels packed four per `u32` for digits and unpacked with
  `unpack4x8unorm`; labels behind them), `batch` (the sample indices of the current batch). No kernel touches more than
  four: forward reads `params`, `data`, `batch` and writes `acts`; backward reads `params`, `acts` and writes `deltas`;
  the weight-gradient kernel reads `acts`, `deltas`, `batch`, `data` and writes `grads`; the update reads `grads` and
  writes `params`. Per-layer sizes and offsets live in one uniform per layer.
- **Kernels (`mlp.wgsl`):** `forward` (one invocation per sample × neuron: dot product over the previous layer);
  `outputDelta` (prediction − label, with sigmoid or softmax computed per sample); `backward` (one invocation per
  sample × neuron of layer l: Σ over layer l + 1, times the activation derivative); `weightGrad` (one invocation per
  weight: Σ over the batch in index order; no atomics); `update` (SGD with momentum and weight decay); `eval` (forward
  over an arbitrary set of samples, writing per-sample loss and correctness, reduced on the CPU for the curves).
  Workgroup size 64 throughout.
- **Decision map (`boundary.wgsl`, fragment):** every pixel evaluates the network from `params` (storage reads in
  fragment stage work everywhere), 256 × 256 at the default quality, so the map is exact and redrawn each frame at no
  CPU cost. The same shader with a "stop at layer l, neuron j" uniform renders the per-neuron thumbnails in the
  network diagram.
- **Step probe:** after a step, `eval` runs the training set at θ − s·η·g for s = 0 … 2 (33 samples along the step),
  giving the loss along the line the step took; the chart marks s = 1. For a linear model with squared loss this is a
  parabola and the step goes uphill exactly when η > 2 / λ_max of the Hessian (a tested fact); for a real network the
  student sees the same shape without the formula.
- **Cost:** digits, 784 → 128 → 10, batch 128: ≈ 80 MFLOP per step (≈ 8 dispatches); one epoch of 10 000 images is
  79 steps, well under a second on integrated graphics; the twin needs ~1 s per epoch in a worker and is used only for
  tests and the draw pad. Plane: negligible. Tests use 2D sets and a 64-image digit subset on the software renderer.
- **Read-backs:** `params` (≤ 500 KB) at ≤ 4 Hz for the diagram, the templates and the draw pad (inference on the
  twin); the eval results (≤ 12 000 × 8 B) for the curves and confusion matrix at the same rate.

## 5. Measurements and how each is proven

| Readout / claim | Proven by |
|---|---|
| The gradient is right | Twin: analytic gradient vs centred finite differences (h = 10⁻⁵, 64-bit) on random nets with every activation and both losses, relative error < 10⁻⁶ (cs231n's gradient-check recipe) |
| GPU = twin | Weights after 20 steps from the same seed and batches agree to f32 tolerance for every activation, both losses, 0–6 layers, widths not a multiple of 64; the decision map equals the twin's at sampled pixels; `eval` loss equals the twin's |
| A line cannot do XOR | Twin: a linear model on noise-free XOR never beats 75 % accuracy, and its cross-entropy cannot fall below ln 2 ≈ 0.69 (exact for the symmetric four-point XOR, by convexity and the (x, y) → (−x, −y) symmetry; asserted with a margin on generated sets over many seeds); with 2 tanh hidden units or the x·y feature it reaches 100 % |
| The step-size threshold | Twin: linear model, squared loss, fixed data: the loss falls every step for η < 2 / λ_max and rises for η > 2 / λ_max (λ_max from the data's Gram matrix) |
| Vanishing gradients | Twin: six sigmoid layers at Glorot init: the first layer's gradient norm is < 1 % of the last layer's; with ReLU/He it is within 10× |
| Symmetry | Twin and GPU: zero init keeps every hidden neuron of a layer identical after 200 steps (spread exactly 0); random init does not |
| Overfitting | Twin: spiral, 50 % noise, 10 % train share, 6 × 8 tanh: train loss < 0.05 while test loss > 0.3 after N steps from the challenge's seed |
| Digits accuracy | Twin on the shipped subset (fast enough in a test at a few epochs): linear ≥ 88 % test, one hidden layer of 128 ≥ 94 % (the exact figures are measured while building and asserted with a margin); the GPU reaches the same ± 0.5 % |
| Draw-pad normalisation | Unit test: a thick off-centre stroke ends up 20 px tall and centred by mass, matching the MNIST recipe |

## 6. Shareable settings and challenges

**URL state:** view, dataset and its noise/size/train share, features, layer widths, activation, learning rate,
batch size, momentum, L2, init (random/zeros), seed, speed (steps per frame), digits network choice. Reset re-rolls
the seed; a link freezes it, so a challenge's "setup" is a reproducible run.

**Challenges** (goals read the measured values, cannot be shortcut, every number tested on the twin):

1. **A straight line can't do it.** XOR, no hidden layer. Prediction: the accuracy will reach… 100 % / ~75 % /
   ~50 % (75 %: it gets three quadrants right at best). Goal: ≥ 95 % test accuracy with *at least one hidden layer*
   (the goal checks the layer count, so changing the dataset or picking the x·y feature does not count).
2. **Or give it the right feature.** Same XOR, no hidden layer allowed. Goal: ≥ 95 % with the x·y feature on (and
   the sibling: circle, with x² and y²). Explanation: a linear model in the right features *is* a nonlinear model in
   the original ones; that is what hidden layers learn to do by themselves.
3. **Too big a step.** Two blobs, η = 10. Prediction: the loss will… fall faster / fall then stall / rise and blow
   up. Goal: the loss exceeds 10 × its starting value or becomes NaN (the step probe shows every step landing uphill);
   then the hint sends the student to find the largest η that still falls.
4. **Memorise or learn?** Spiral, 50 % noise, 10 % train share, 6 × 8. Prediction: the test loss will follow the
   train loss down / level off / go *up* while train goes down. Goal: train loss < 0.05 while test loss > 0.3.
   Explanation: with 20 points and 300 weights the network memorises the noise.
5. **Why random?** XOR, init = zeros, 2 × 4. Prediction: it will learn… normally / slower / never. Goal: 200 steps
   with the per-neuron spread still exactly zero (readout), then switch to random init and reach 95 %.
6. **Deep and dark.** Circle, six layers of 4, sigmoid. Prediction: deeper is… better / the same / stuck. Goal: the
   measured first-layer gradient < 1 % of the last layer's (readout) and accuracy still ~50 % after 500 steps; then
   ReLU trains.
7. **Read my handwriting.** Digits, no hidden layer. Prediction: a straight-line model reads… 1 / 5 / 9 digits in
   ten (9). Goal: ≥ 95 % test accuracy (needs a hidden layer), then draw a digit the network gets wrong and look at
   where it sits in the confusion matrix.

## 7. Testing

- `net.ts` (twin): layer packing and offsets, activations and derivatives, both losses, forward, backprop, SGD with
  momentum and decay, seeded init, the dataset generators, the batch shuffler, finite-difference gradient check, the
  λ_max bound, and every claim in §5 and every number in the challenge texts.
- `digits.ts`: sprite decoding, the draw-pad normalisation, confusion matrix, accuracy.
- GPU tests (`gpu-shaders.test.ts`): each kernel vs the twin; a full step and 20 steps vs the twin; the decision map
  vs the twin; the step probe vs the twin; shapes that are not multiples of 64; the 4-buffer budget respected
  (`check:wgsl` validates all shaders against a real device).
- e2e: both views render without GPU errors; the loss curve falls on two blobs; adding a hidden layer solves XOR in
  the browser; the step probe shows an uphill step at η = 10; the digits view loads the sprite from the static export,
  trains to ≥ 88 % with the linear model within the test budget, and the draw pad returns ten probabilities summing
  to 1; URL state and hostile links; phone layout; axe; reduced motion starts paused; challenge flows.

## 8. Phases and estimate (≈ 5–6 days)

| Phase | Work | Done when |
|---|---|---|
| 1. Twin (1 day) | `net.ts` + tests, including the gradient check and the known results of §5 | every claim in §5 proven in 64-bit |
| 2. Kernels (1 day) | `mlp.wgsl`, `boundary.wgsl`, `engine.ts` (buffers, dispatch sequence, step probe, eval), GPU-vs-twin tests | 20 GPU steps = 20 twin steps to f32 tolerance |
| 3. Playground view (1–1½ days) | data generators, features, decision map, network diagram with thumbnails, loss curves, step probe, controls, URL state | XOR solved live; the step probe shows uphill steps |
| 4. Digits view (1–1½ days) | sprite build script and data pipeline, digit grid, accuracy curves, templates, confusion matrix, draw pad | ≥ 95 % in the browser in under a minute; a drawn digit is read |
| 5. Challenges and polish (1 day) | 7 challenges, home plate (7 plates), poster, preview, docs, e2e | `pnpm verify` and full e2e green on prod, dev and static |

## 9. Decisions needed

1. **Title.** Recommended: rename the catalog entry to **"How AI Learns"** (description: "Train a neural network live
   on the GPU: watch the decision boundary form, blow it up with a big step, overfit it, then read your own
   handwriting"). Also fold the planned biology entry "Neural Network Learning" into this one (remove it from the
   catalog), since it duplicates the subject.
2. **Digits data.** Recommended: a 12 000-image MNIST subset shipped with the app under CC BY-SA 3.0 with credit in
   the view and README. Alternative: Fashion-MNIST (MIT, ten clothing classes) — less iconic and the draw pad makes no
   sense for it.
3. **Network limits.** Recommended: up to 6 hidden layers × 32 neurons (the diagram draws lines up to 16 per layer
   and switches to weight heat-maps above). Larger is possible on the GPU but unreadable.
4. **Scope of v1.** Recommended: both views and all seven challenges. If time is short, Playground with challenges
   1–6 first; Digits as a follow-up PR.

## Sources

- TensorFlow Playground (what it offers and its limits: 6 layers × 8 neurons, 500 points, 100² map, squared loss),
  source: <https://github.com/tensorflow/playground/blob/master/src/state.ts> and
  <https://github.com/tensorflow/playground/blob/master/src/playground.ts>; course use, e.g.
  <https://www.cs.ucf.edu/~lboloni/Teaching/CAP4611_Fall2023/homeworks/CAP4611-HW6-NeuralNetworks.pdf> and
  <https://turcotte.xyz/teaching/csi-5180/lectures/16/index.html>
- Gradient checking (centred differences, relative error, do it in double precision): cs231n notes (copy at
  <https://www.scribd.com/document/869435143/CS231n-Deep-Learning-for-Computer-Vision-p-1>);
  <https://www.cs.purdue.edu/homes/dgleich/cs520-2026/claude-lectures/lecture-21/page1-finite-differences.html>;
  classroom activity <https://ds701.cds.bu.edu/slides/16-Neural-Networks-I-recap.html>
- Step-size threshold 2 / λ_max and the edge of stability: Arora, Li, Panigrahi (ICML 2022)
  <https://arxiv.org/pdf/2205.09745v2>; Lee & Jang (ICLR 2023) <https://mlanthology.org/iclr/2023/lee2023iclr-new/>
- Overfitting and double descent with small MLPs and label noise: <https://arxiv.org/pdf/2405.07679>,
  <https://arxiv.org/pdf/2411.00247>, lab <https://cw.fel.cvut.cz/wiki/courses/bev033dla/labs/ddescent/start>
- MNIST licence (CC BY-SA 3.0) and the browser sprite format:
  <https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/datasets/mnist/load_data>,
  <https://github.com/tensorflow/tfjs-examples/blob/master/mnist/data.js>; Fashion-MNIST (MIT)
  <https://arxiv.org/pdf/1708.07747>
- Baselines on MNIST (linear ≈ 88 %, hidden layers higher): <https://cs.calvin.edu/courses/cs/375/26sp/units/neural-computing/mlp/>
- Draw-your-digit demos and the centring caveat: Harley, "An Interactive Node-Link Visualization of Convolutional
  Neural Networks" (ISVC 2015) <https://inf-schule.de/assets/thirdparty/nn_vis/harley_vis_isvc15.pdf>
- Teaching neural networks without calculus (16–18 year olds): <https://arxiv.org/pdf/1904.00296>
- In-browser GPU training exists only as frameworks or inference demos: TabGPT <https://github.com/jbingen/tabgpt>,
  jax-js, CoViz <https://github.com/tillife/neural-network-playground>
