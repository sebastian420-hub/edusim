# How AI Writes (a tiny language model) — research and implementation plan

The eighth simulation, proposed title **"How AI Writes"** (decision 1), as the sequel to *How AI Learns*. A language
model trained live on the GPU, one letter at a time, from gibberish to recognisable Shakespeare (or invented names)
in a minute or two. Same bar as the other seven: a 64-bit TypeScript twin of every shader, GPU-vs-twin tests,
shareable settings, measurements a test proves correct, guided challenges, e2e scenarios, a poster and a live
home-page preview.

Why this one: it is the most direct bridge from the neural-network sim to the tools the user works with every day.
ChatGPT and Claude are trained exactly as here: a loss, gradient descent, and a model that predicts the next piece of
text. The practical ideas (temperature, context, memorisation, in-context learning, tokens) are the ones that
explain how chatbots behave, and every one of them can be *measured* in a model small enough to train in a tab.

## 1. What students should get out of it

1. **A language model only predicts the next letter.** It outputs a probability for every possible next character;
   "writing" is sampling one, appending it, and repeating. Everything else follows from that.
2. **Loss is surprise, measured in bits.** Guessing uniformly among the 65 characters of the Shakespeare text costs
   ln 65 = 4.17 nats (6.0 bits) per letter; knowing letter frequencies, 3.31 nats; knowing the previous letter,
   2.45 nats. Lower loss means the model is less surprised by real text. (Measured on the shipped data, §5.)
3. **More context helps, but counting can't use it.** A table of counts that looks at the last 4 letters gets the
   held-out loss down to 1.75 nats; with 5 or 6 letters it *gets worse* (1.98, 2.34) because most long contexts were
   never seen: it memorises instead of generalising. A neural network shares what it learns between similar contexts,
   which is why neural models replaced count tables.
4. **Temperature trades safety for variety.** Near 0 the model always picks its favourite letter and falls into
   loops ("the the the"); at 1 it writes in the style it learnt; at 2 it writes noise.
5. **Models can memorise their training text.** A big model on a short text reproduces it word for word: the
   reason chatbots can regurgitate training data.
6. **Attention lets a model copy from its context.** A two-layer transformer learns to continue a pattern it has only
   just seen (an *induction head*), which a one-layer model provably cannot do. That is the mechanism behind
   in-context learning: why examples in a prompt work.
7. **You are still a better predictor.** In Shannon's 1951 guessing game people need about 1 bit per letter of
   English; a model you can train in a tab needs about 2–3. Big models closed that gap with data and size, not new
   principles.

## 2. Product shape: one simulation, four views of the same model

| View | What you see | What you do | What it measures |
|---|---|---|---|
| **Train** (default) | The text it writes *now*, regenerated as it trains (gibberish → words → verse); the loss curve with reference lines (guessing, letter frequencies, previous letter, best count table) | Pick the text (Shakespeare, names, a sonnet, a copying game, your own) and the model (count table, neural bigram, MLP, transformer); size, learning rate, batch; Play | Training and held-out loss in nats and bits per letter, perplexity, parameters, letters seen, speed |
| **Predict** | A prompt and the model's next-letter probabilities as bars; for a transformer, each head's attention over the prompt; the letters' embeddings in 2D | Type a prompt, click a bar to append it, slide the temperature, "write 200 letters" | The probabilities and their entropy, the attention weights, repetition in generated text |
| **Count tables** | Held-out and training loss for count tables looking back 0 … 6 letters, and samples from each (Shannon's 1948 "approximations to English", live) | Pick how far back to look | The exact entropies of §5; how many contexts were ever seen |
| **Shannon game** | A hidden passage of held-out text | Guess the next letter until right, for 50 letters | Your guesses per letter and Shannon's bounds on your entropy, beside the model's on the same passage |

## 3. Models (a ladder, each one a lesson)

All models work on characters (the vocabulary is the set of characters in the text: 65 for Shakespeare, 27 for names).

1. **Count table, context k = 0 … 6** (CPU, exact): probabilities are counts, with add-α smoothing for unseen
   pairs. The training-set loss of the counts is the minimum any model with that context can reach on that set.
2. **Neural bigram**: a V × V table of logits trained by gradient descent. Its loss is convex, so it must converge to
   the k = 1 count table: a tested theorem, and the bridge from counting to learning.
3. **MLP (Bengio et al. 2003)**: embeddings of the last k letters (k = 1 … 16), concatenated, one tanh hidden layer,
   softmax. Shows what learnt embeddings buy over counts.
4. **Transformer (GPT)**: learned token and position embeddings, 1–4 pre-norm blocks (multi-head causal
   self-attention, then a GELU MLP of 4× width), final layer norm, a linear head. 1–4 heads, width 32–128, context
   16–128. Default: 2 layers, 4 heads, width 64, context 64 (about 0.1 M parameters); "bigger": 4 layers, width 128
   (about 0.8 M, the size of nanoGPT's CPU configuration).

Training: AdamW (β = 0.9, 0.99, weight decay on matrices only) with a short warm-up and gradient clipping (global
norm 1), as nanoGPT does; SGD is the wrong lesson here (the step-size lesson already lives in *How AI Learns*).
Batches are random windows of the training text from a seeded generator, the same for the GPU and the twin.
Training/held-out split: the first 90 % / last 10 % of Shakespeare; names are shuffled first (the file is sorted by
popularity) and split 90/10.

Sampling: the model's logits divided by the temperature, softmax, then a draw from a seeded generator on the CPU;
T = 0 means the most likely letter.

## 4. GPU design

Same principles as *How AI Learns*: training is a fixed sequence of dispatches; every sum runs in a fixed order (no
atomics, so GPU and twin agree step by step); compatibility-mode limits (at most 4 storage buffers per stage,
workgroups of 64).

- **Buffers:** `params` (all weights), `opt` (gradients, then Adam's m and v), `acts` (every forward activation of
  the batch, by offset), `dacts` (their gradients), `tokens` (the text, a u32 per character, uploaded once), `batch`
  (the B window starts of this step). A kernel binds at most four; offsets and shapes come in a uniform, as in
  `mlp.wgsl`. A forward matmul reads `params` and reads and writes different regions of `acts` through one binding
  (a buffer cannot be bound writable twice).
- **Kernels (`gpt.wgsl`, one entry point each):** embed; layer norm (and its backward, with dγ, dβ summed per column
  in row order); matmul with optional transposes, bias and residual add (tiled: 8 × 8 invocations each computing 4 × 4
  outputs, the tiles in workgroup memory); GELU; causal attention forward (one invocation per batch × head × query:
  scores, softmax, weighted sum; the probabilities kept for the backward pass and the attention view); attention
  backward (dV and dK summed over queries, dQ over keys: gathers, not scatters); cross-entropy forward and backward
  per position; embedding backward (one invocation per vocabulary row × channel, gathering the positions that use it);
  global gradient norm (two-stage fixed-order reduction); AdamW. Count tables run on the CPU (a Web Worker for k ≥ 4
  on Shakespeare), the neural bigram and MLP reuse the embed, matmul, tanh and loss kernels.
- **Cost (default model, batch 16 × 64 letters):** about 6 × 0.1 M × 1 024 ≈ 0.6 GFLOP per step plus attention,
  around 50 dispatches. With simple tiled kernels on integrated graphics, roughly 10–30 ms a step: 2 000 steps in about
  a minute. The bigger model is about 8× the work. The page sets steps per frame from the measured frame time.
- **Generation:** a forward pass of one window per letter; the last position's logits (65 floats) are read back and
  sampled on the CPU with the twin's sampler. A few hundred letters a second, streamed into the text panel.
- **Held-out loss:** a fixed set of 64 held-out windows, evaluated every ~2 s with the forward kernels only.

## 5. Measurements and how each is proven

Numbers measured on the actual files with a short script while researching (count tables, exact):

| Tiny Shakespeare (1 115 394 characters, 65 symbols) | Training-set minimum | Held-out (add-α tuned) |
|---|---|---|
| Uniform guessing | 4.174 nats (6.02 bits) | 4.174 |
| Letter frequencies (k = 0) | 3.309 | 3.347 |
| Previous letter (k = 1, the bigram) | 2.452 | 2.482 |
| k = 2 | 1.903 | 2.046 |
| k = 3 | 1.491 | 1.779 |
| k = 4 | 1.216 | **1.750** (best) |
| k = 5 | 0.983 | 1.979 |
| k = 6 | 0.761 | 2.342 (memorised: 264 623 contexts) |

Names (32 032 names, 27 symbols with the line break as the end of a name): uniform 3.296, letter frequencies 2.823,
bigram 2.454 nats over the whole file.
For comparison, nanoGPT's CPU configuration (4 layers, width 128) reaches 1.88 held-out after 2 000 steps and its GPU
"baby GPT" (6 layers, width 384) 1.47; makemore's 0.2 M-parameter transformer gets about 1.92 on names.

| Readout / claim | Proven by |
|---|---|
| Backpropagation through every part (attention, layer norm, GELU, embeddings, loss) | Twin: centred finite differences in 64-bit, relative error < 10⁻⁶ |
| GPU = twin | Each kernel against the twin; 20 full training steps from the same seed and batches agree to f32 tolerance (Adam included); attention probabilities and generated logits agree |
| The reference lines | The count-table entropies above, recomputed from the shipped files in a test |
| A neural bigram converges to the count table | Twin: its loss approaches 2.452 on Shakespeare (convex problem, unique optimum) |
| Count tables memorise | Twin: held-out loss is lowest at k = 4 and rises for k = 5, 6 while training loss keeps falling |
| Temperature | Twin: the entropy of the next-letter distribution rises with T; T → 0 is the argmax; at low T generated text repeats (a measured repetition score), at T = 2 its own loss under the model is high |
| Memorisation | Twin/GPU: a bigger transformer on the 14-line sonnet reproduces a run of ≥ 80 letters verbatim, measured as the longest common substring with the training text |
| Induction (in-context copying) | GPU and twin: on the copying game, a 2-layer transformer's loss on the repeated half falls below 0.5 nats while the first half stays near ln V; a 1-layer model's repeated half stays near ln V. Exact thresholds measured while building |
| Shannon's bounds | Unit tests of the bounds on sources with known entropy: Σ r (p_r − p_{r+1}) log₂ r ≤ H ≤ −Σ p_r log₂ p_r, where p_r is the share of letters needing r guesses |

## 6. Shareable settings and challenges

**URL state:** view, text, model and its size (context, layers, heads, width), learning rate, batch, temperature,
seed, speed. Your own text is kept in the browser only (it would not fit in a link, and it is private).

**Challenges** (goals read measured values, cannot be shortcut, every number tested on the twin):

1. **How surprised?** A model that sees only the previous letter. Prediction: its loss can get down to about 4.2
   (no better than guessing) / about 2.5 / about 1.0. (2.45: the bigram limit.) Goal: train the neural bigram to
   within 0.02 of the count table, and see that the two are the same model.
2. **More context, until it memorises.** Count tables. Prediction: as the table looks further back, its loss on unseen
   text keeps falling / falls then rises / stays flat. Goal: find the context with the lowest held-out loss (4 letters,
   1.75), having seen it rise beyond.
3. **Beat the table.** On names, where training is quick: reach a held-out loss below the best count table with a
   neural model. Explanation: embeddings let similar contexts share what they learnt. (Exact threshold measured while
   building; makemore's figures suggest roughly 2.1–2.2 against the table's 2.3 or so.)
4. **Turn down the temperature.** Prediction: at a temperature near 0 the model writes its best text / the same
   phrase over and over / nothing. Goal: generate 200 letters at T ≤ 0.2 and see the repetition score pass a threshold;
   then 200 at T = 2 and see its own loss on them rise above the guessing line.
5. **Learn by heart.** The 14-line Sonnet 18 and the bigger transformer. Goal: a generated passage with 80 or more
   letters copied verbatim. Explanation: memorisation, and why it matters for privacy and copyright.
6. **Copy what you just saw.** The copying game: random letters, then the same letters again. Prediction: a one-layer
   transformer will learn to copy the second half: easily / slowly / never. Goal: with two layers, get the loss on
   the repeated half below 0.5 nats, then find the head that looks back to the letter after the last match.
7. **You against the machine.** The Shannon game on a held-out passage. Prediction: who needs fewer bits per letter?
   Goal: play 50 letters; compare your upper bound with the model's loss on the same letters.

## 7. Testing

- `text.ts` (data: encoding, splits, seeded windows), `ngram.ts` (counts, entropies, smoothing, samples), `gpt.ts`
  (twin: every layer forward and backward, AdamW, clipping), `sample.ts` (temperature, repetition score, longest
  verbatim copy), `shannon.ts` (bounds): unit tests for every claim in §5 and every number in the challenge texts.
- GPU tests: each kernel against the twin, shapes not multiples of the tile size; 20 full steps; the attention
  probabilities; generation logits; the induction result on a small configuration. Throughput logged on the software
  renderer to keep the step budget honest.
- e2e: each view renders without GPU errors; the loss falls below the bigram line on Shakespeare within the test
  budget; the generated text changes as it trains; the count-table chart shows the minimum at 4; temperature 0
  repeats; the Shannon game scores; URL state and hostile links; your own text stays local; phone layout; axe; reduced
  motion; a challenge flow.

## 8. Phases and estimate (≈ 7–8 days)

| Phase | Work | Done when |
|---|---|---|
| 1. Twin and numbers (2 days) | data, count tables, neural bigram, MLP, transformer forward/backward, AdamW, sampling, Shannon bounds; measure the challenge thresholds (names, memorisation, induction) | every claim in §5 proven in 64-bit |
| 2. Kernels (2 days) | `gpt.wgsl`, `engine.ts`, GPU-vs-twin tests, throughput | 20 GPU steps = 20 twin steps; the default model trains a step in tens of milliseconds |
| 3. Train view (1 day) | text panel, loss chart with reference lines, readouts, settings, URL state | gibberish becomes words live; loss below the bigram line |
| 4. Predict view (1 day) | prompt, probability bars, temperature, attention maps, embeddings in 2D | attention matches the twin for a prompt |
| 5. Count tables and Shannon game (1 day) | the exact ladder and samples; the guessing game and its bounds | the minimum at 4 letters; bounds bracket known entropies |
| 6. Challenges and polish (1 day) | 7 challenges, home plate (8 plates), poster, preview, docs, e2e | `pnpm verify` and full e2e green on prod, dev and static |

## 9. Decisions needed

1. **Title and place.** Recommended: **"How AI Writes"**, a new Computer Science entry (`/cs/language-model`) next to
   *How AI Learns*.
2. **Texts shipped.** Recommended: tiny Shakespeare (public domain, 434 KB gzipped) and the makemore names (from the
   US Social Security baby-name data, CC0; 103 KB gzipped), Sonnet 18 (public domain) and the generated copying game;
   plus your own pasted text, kept in the browser. Not recommended for v1: TinyStories (simple English that tiny models
   write fluently, but CDLA-Sharing-1.0 and itself generated by GPT models).
3. **Tokens.** Recommended: characters throughout, plus a small read-only "tokens" panel in the Predict view that
   learns a byte-pair tokenizer on the text (CPU, a few hundred merges) and shows why a model that sees "str·aw·berry"
   cannot simply count its r's. About half a day; drop it if time is short.
4. **Model sizes.** Recommended: default 2 × 64 (about 0.1 M parameters), "bigger" 4 × 128 (about 0.8 M), with steps
   per frame set from the measured frame time.
5. **Scope of v1.** Recommended: all four views and seven challenges. If time is short: Train and Predict with
   challenges 1–6 first; Count tables and the Shannon game as a follow-up PR.

## 10. What already exists (and how this differs)

- **Transformer Explainer** (Georgia Tech Polo Club, AAAI-25; about 125 000 users): GPT-2 running in the browser
  with a temperature slider and animated attention, inference only.
- **LLM Visualization** (Brendan Bycroft): a 3D walk-through of a tiny GPT's inference, no training.
- **Sequence Toy** (Vin Howe, built on the Piston WebGPU autodiff library): trains transformers, LSTMs, GRUs and RNNs
  in the browser as a playground. The closest relative; its site could not be opened from this environment, so its
  features were not checked.
- **makemore / nanoGPT** (Karpathy): the reference code and videos (Python, not in the browser).

None of them pairs live training with *measured* lessons: exact information-theory reference lines, count tables
against learning, temperature and memorisation scored, the induction mechanism shown as a measurable before/after,
and a human-versus-model Shannon game, all checked against a 64-bit twin.

## Sources

- Tiny Shakespeare (Karpathy, char-rnn): <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>;
  makemore and its names (the most common 32K names from ssa.gov, 2018): <https://github.com/karpathy/makemore>;
  SSA baby-name data, CC0: <https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data>
- nanoGPT shakespeare_char results (1.4697 baby GPT; 1.88 CPU configuration): <https://github.com/karpathy/nanoGPT>
- Bengio et al. 2003, *A Neural Probabilistic Language Model*: <https://www.jmlr.org/papers/v3/bengio03a.html>
- Shannon 1951, *Prediction and Entropy of Printed English* (0.6–1.3 bits per letter; the guessing game):
  <https://www.nokia.com/bell-labs/publications-and-media/publications/prediction-and-entropy-of-printed-english/>;
  the bounds restated by Mahoney: <https://mattmahoney.net/dc/entropy1.html>; a 1960s replication (lower bound near
  1 bit, upper 1.6–1.7): <https://dspace.mit.edu/bitstream/1721.1/51270/1/RLE_QPR_037_IX.pdf>
- Induction heads and in-context learning: Olsson et al. 2022, <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>;
  two layers needed, proven: <https://arxiv.org/html/2508.07208v2>
- Temperature, greedy repetition and nucleus sampling: Holtzman et al. (ICLR 2020), as summarised in
  <https://arxiv.org/pdf/2012.14660>
- Tokens and letter counting: <https://arxiv.org/pdf/2410.19730>
- Transformer Explainer: <https://arxiv.org/html/2408.04619v1>, <https://ojs.aaai.org/index.php/AAAI/article/view/35347>;
  LLM Visualization: <https://github.com/bbycroft/llm-viz>; Sequence Toy / Piston: <https://github.com/vinhowe/piston>
- TinyStories (CDLA-Sharing-1.0): <https://huggingface.co/datasets/roneneldan/TinyStories>
- Students' misconceptions about how chatbots work (overtrust, mental models): <https://arxiv.org/pdf/2505.24126>,
  <https://www.arxiv.org/abs/2309.03087>
