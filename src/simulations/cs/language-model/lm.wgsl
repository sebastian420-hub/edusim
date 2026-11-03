// Kernels of "How AI Writes": every operation of training, evaluating and writing with a character-level language
// model (neural bigram, MLP or transformer). One entry point per op kind; the programs that chain them are built in
// model.ts and run, unchanged, by the 64-bit CPU twin in cpu.ts, which mirrors each kernel line for line.
//
// Every sum runs in a fixed order and nothing uses atomics, so a step is deterministic and matches the twin.
// All four storage buffers are bound to every kernel (an explicit layout); the op's fields arrive as a uniform at a
// dynamic offset, so a whole training step is one command buffer.
//
// params: the weights.  work: activations, their gradients, evaluation losses.
// opt: gradient [0, P), Adam's m and v, the gradient-norm scratch, the step's learning rate and bias corrections.
// tokens: the text, the writing lines, the prompt, window starts, and the writing counter.

struct Op {
  M: u32, N: u32, K: u32, flags: u32,
  a: u32, b: u32, c: u32, d: u32,
  e: u32, f: u32, g: u32, h: u32,
  B: u32, T: u32, H: u32, D: u32,
  V: u32, ctx: u32, starts: u32, gen: u32,
  s0: f32, s1: f32, s2: f32, s3: f32,
}

@group(0) @binding(0) var<storage, read_write> params: array<f32>;
@group(0) @binding(1) var<storage, read_write> work: array<f32>;
@group(0) @binding(2) var<storage, read_write> opt: array<f32>;
@group(0) @binding(3) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(4) var<uniform> u: Op;

const LN_EPS = 1e-5;
const NORM_PARTS = 256u;
const MAX_HEAD = 64u;
const SQRT_2_OVER_PI = 0.7978845608028654;

fn flatId(gid: vec3u, nwg: vec3u) -> u32 {
  return gid.x + gid.y * nwg.x * 64u;
}

fn windowStart(b: u32) -> u32 {
  var s = tokens[u.starts + b];
  if (u.gen > 0u) {
    s += tokens[u.gen];
  }
  return s;
}

fn safeTanh(x: f32) -> f32 {
  return tanh(clamp(x, -15.0, 15.0));
}

/// Letter (and, for the transformer, position) embeddings of B windows of T letters; ctx > 1 concatenates the
/// embeddings of the last ctx letters (the MLP): out[row][j·K + i] = table[letter t − j][i] (+ pos[t][i] for j = 0).
@compute @workgroup_size(64)
fn embed(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let id = flatId(gid, nwg);
  if (id >= u.B * u.T * u.N) { return; }
  let i = id % u.K;
  let j = (id / u.K) % u.ctx;
  let r = id / u.N;
  let b = r / u.T;
  let t = r % u.T;
  let tok = tokens[windowStart(b) + t - j];
  var v = params[u.a + tok * u.K + i];
  if ((u.flags & 1u) != 0u && j == 0u) {
    v += params[u.b + t * u.K + i];
  }
  work[u.c + r * u.N + j * u.K + i] = v;
}

/// The embeddings' gradients, gathered: one invocation per (letter, channel) sums the rows that used it
/// (in the order b, t, j); then one per (position, channel) for the position table.
@compute @workgroup_size(64)
fn embedBack(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let id = flatId(gid, nwg);
  let nTable = u.V * u.K;
  if (id < nTable) {
    let v = id / u.K;
    let i = id % u.K;
    var sum = 0.0;
    for (var b = 0u; b < u.B; b++) {
      let s = windowStart(b);
      for (var t = 0u; t < u.T; t++) {
        for (var j = 0u; j < u.ctx; j++) {
          if (tokens[s + t - j] == v) {
            sum += work[u.c + (b * u.T + t) * u.N + j * u.K + i];
          }
        }
      }
    }
    opt[u.a + v * u.K + i] = sum;
    return;
  }
  if ((u.flags & 1u) == 0u || id >= nTable + u.T * u.K) { return; }
  let p = id - nTable;
  let t = p / u.K;
  let i = p % u.K;
  var sum = 0.0;
  for (var b = 0u; b < u.B; b++) {
    sum += work[u.c + (b * u.T + t) * u.N + i];
  }
  opt[u.b + t * u.K + i] = sum;
}

/// Layer norm of M rows of N: y = (x − mean) / √(var + ε) · γ + β, keeping (mean, 1/√(var + ε)) per row.
@compute @workgroup_size(64)
fn layernorm(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let r = flatId(gid, nwg);
  if (r >= u.M) { return; }
  let x = u.a + r * u.N;
  var mean = 0.0;
  for (var i = 0u; i < u.N; i++) { mean += work[x + i]; }
  mean /= f32(u.N);
  var v = 0.0;
  for (var i = 0u; i < u.N; i++) { v += (work[x + i] - mean) * (work[x + i] - mean); }
  let rstd = 1.0 / sqrt(v / f32(u.N) + LN_EPS);
  for (var i = 0u; i < u.N; i++) {
    work[u.c + r * u.N + i] = (work[x + i] - mean) * rstd * params[u.b + i] + params[u.b + u.N + i];
  }
  work[u.d + 2u * r] = mean;
  work[u.d + 2u * r + 1u] = rstd;
}

/// Layer norm backward: invocations [0, M) give dx per row (plus the residual stream's gradient when flags & 1);
/// [M, M + N) give dγ and dβ per column, summed over rows in order.
@compute @workgroup_size(64)
fn layernormBack(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let id = flatId(gid, nwg);
  let N = u.N;
  if (id < u.M) {
    let r = id;
    let mean = work[u.d + 2u * r];
    let rstd = work[u.d + 2u * r + 1u];
    var m1 = 0.0;
    var m2 = 0.0;
    for (var i = 0u; i < N; i++) {
      let dyg = work[u.e + r * N + i] * params[u.b + i];
      m1 += dyg;
      m2 += dyg * (work[u.a + r * N + i] - mean) * rstd;
    }
    m1 /= f32(N);
    m2 /= f32(N);
    for (var i = 0u; i < N; i++) {
      let xhat = (work[u.a + r * N + i] - mean) * rstd;
      let dyg = work[u.e + r * N + i] * params[u.b + i];
      var dx = rstd * (dyg - m1 - xhat * m2);
      if ((u.flags & 1u) != 0u) { dx += work[u.f + r * N + i]; }
      work[u.c + r * N + i] = dx;
    }
    return;
  }
  if (id >= u.M + N) { return; }
  let i = id - u.M;
  var dg = 0.0;
  var db = 0.0;
  for (var r = 0u; r < u.M; r++) {
    let dy = work[u.e + r * N + i];
    dg += dy * (work[u.a + r * N + i] - work[u.d + 2u * r]) * work[u.d + 2u * r + 1u];
    db += dy;
  }
  opt[u.b + i] = dg;
  opt[u.b + N + i] = db;
}

// ── Matrix product: C (M × N) = op(A) · op(B) (+ bias) (+ residual), tiled 32 × 32 per workgroup, 4 × 4 per thread.
// flags: 1 Aᵀ, 2 Bᵀ, 4 B in work (else params), 8 C in opt (else work), 16 bias, 32 residual, 64 accumulate.
// Leading dimensions: f (A), g (B), h (C and the residual).

var<workgroup> tileA: array<f32, 512>; // 32 rows × 16 of K
var<workgroup> tileB: array<f32, 512>; // 16 of K × 32 columns

fn loadA(m: u32, k: u32) -> f32 {
  if (m >= u.M || k >= u.K) { return 0.0; }
  if ((u.flags & 1u) != 0u) { return work[u.a + k * u.f + m]; }
  return work[u.a + m * u.f + k];
}

fn loadB(k: u32, n: u32) -> f32 {
  if (k >= u.K || n >= u.N) { return 0.0; }
  var at = u.b + k * u.g + n;
  if ((u.flags & 2u) != 0u) { at = u.b + n * u.g + k; }
  if ((u.flags & 4u) != 0u) { return work[at]; }
  return params[at];
}

@compute @workgroup_size(64)
fn matmul(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u) {
  let tx = li % 8u;
  let ty = li / 8u;
  let m0 = wg.y * 32u;
  let n0 = wg.x * 32u;
  var acc: array<f32, 16>;
  for (var k0 = 0u; k0 < u.K; k0 += 16u) {
    for (var q = 0u; q < 8u; q++) {
      let idx = li + q * 64u;
      tileA[idx] = loadA(m0 + idx / 16u, k0 + idx % 16u);
      tileB[idx] = loadB(k0 + idx / 32u, n0 + idx % 32u);
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      for (var i = 0u; i < 4u; i++) {
        let a = tileA[(ty * 4u + i) * 16u + kk];
        for (var j = 0u; j < 4u; j++) {
          acc[i * 4u + j] += a * tileB[kk * 32u + tx * 4u + j];
        }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = m0 + ty * 4u + i;
    if (m >= u.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let n = n0 + tx * 4u + j;
      if (n >= u.N) { continue; }
      var s = acc[i * 4u + j];
      if ((u.flags & 16u) != 0u) { s += params[u.d + n]; }
      if ((u.flags & 32u) != 0u) { s += work[u.e + m * u.h + n]; }
      let at = u.c + m * u.h + n;
      if ((u.flags & 8u) != 0u) {
        if ((u.flags & 64u) != 0u) { s += opt[at]; }
        opt[at] = s;
      } else {
        if ((u.flags & 64u) != 0u) { s += work[at]; }
        work[at] = s;
      }
    }
  }
}

/// Column sums of M rows (leading dimension f): the bias gradients.
@compute @workgroup_size(64)
fn colsum(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let n = flatId(gid, nwg);
  if (n >= u.N) { return; }
  var s = 0.0;
  for (var m = 0u; m < u.M; m++) { s += work[u.a + m * u.f + n]; }
  if ((u.flags & 8u) != 0u) {
    if ((u.flags & 64u) != 0u) { s += opt[u.c + n]; }
    opt[u.c + n] = s;
  } else {
    if ((u.flags & 64u) != 0u) { s += work[u.c + n]; }
    work[u.c + n] = s;
  }
}

/// Elementwise activation: flags 0 GELU (tanh form), 1 its backward (a = input, e = dy), 2 tanh, 3 its backward
/// (a = output, e = dy).
@compute @workgroup_size(64)
fn act(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = flatId(gid, nwg);
  if (i >= u.M) { return; }
  let x = work[u.a + i];
  switch u.flags {
    case 0u: {
      work[u.c + i] = 0.5 * x * (1.0 + safeTanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
    }
    case 1u: {
      let t = safeTanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x));
      let slope = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x);
      work[u.c + i] = work[u.e + i] * slope;
    }
    case 2u: {
      work[u.c + i] = safeTanh(x);
    }
    default: {
      work[u.c + i] = work[u.e + i] * (1.0 - x * x);
    }
  }
}

/// Causal self-attention, one invocation per (window, head, query): scores against the letters so far, softmax
/// (kept at d for the backward pass and the attention view), then the weighted sum of the values.
/// qkv rows are [q | k | v], each N wide; head h uses columns h·D … (h + 1)·D − 1 of each.
@compute @workgroup_size(64)
fn attention(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let id = flatId(gid, nwg);
  let T = u.T;
  let N = u.N;
  let D = u.D;
  if (id >= u.B * u.H * T) { return; }
  let i = id % T;
  let h = (id / T) % u.H;
  let b = id / (T * u.H);
  let q = u.a + (b * T + i) * 3u * N + h * D;
  let p = u.d + ((b * u.H + h) * T + i) * T;
  var mx = -3.0e38;
  for (var j = 0u; j <= i; j++) {
    let k = u.a + (b * T + j) * 3u * N + N + h * D;
    var dot = 0.0;
    for (var c = 0u; c < D; c++) { dot += work[q + c] * work[k + c]; }
    work[p + j] = dot * u.s0;
    mx = max(mx, work[p + j]);
  }
  var sum = 0.0;
  for (var j = 0u; j <= i; j++) {
    let e = exp(work[p + j] - mx);
    work[p + j] = e;
    sum += e;
  }
  for (var j = 0u; j < T; j++) {
    if (j <= i) { work[p + j] = work[p + j] / sum; } else { work[p + j] = 0.0; }
  }
  for (var c = 0u; c < D; c++) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j++) { acc += work[p + j] * work[u.a + (b * T + j) * 3u * N + 2u * N + h * D + c]; }
    work[u.c + (b * T + i) * N + h * D + c] = acc;
  }
}

fn headDot(x: u32, y: u32) -> f32 {
  var s = 0.0;
  for (var c = 0u; c < u.D; c++) { s += work[x + c] * work[y + c]; }
  return s;
}

/// Attention backward, gathered: invocations [0, BHT) give dq for each query; [BHT, 2BHT) give dk and dv for each
/// key. With dO the output's gradient and O the output: D_i = dO_i · O_i, ds_ij = p_ij (dO_i · v_j − D_i).
@compute @workgroup_size(64)
fn attentionBack(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let id0 = flatId(gid, nwg);
  let T = u.T;
  let N = u.N;
  let D = u.D;
  let n = u.B * u.H * T;
  if (id0 >= 2u * n) { return; }
  let keys = id0 >= n;
  let id = select(id0, id0 - n, keys);
  let x = id % T;
  let h = (id / T) % u.H;
  let b = id / (T * u.H);
  let pRow = u.d + (b * u.H + h) * T * T;
  if (!keys) {
    let i = x;
    let dOi = u.e + (b * T + i) * N + h * D;
    let Di = headDot(dOi, u.b + (b * T + i) * N + h * D);
    var dq: array<f32, MAX_HEAD>;
    for (var j = 0u; j <= i; j++) {
      let ds = work[pRow + i * T + j] * (headDot(dOi, u.a + (b * T + j) * 3u * N + 2u * N + h * D) - Di);
      let k = u.a + (b * T + j) * 3u * N + N + h * D;
      for (var c = 0u; c < D; c++) { dq[c] += ds * work[k + c]; }
    }
    for (var c = 0u; c < D; c++) { work[u.c + (b * T + i) * 3u * N + h * D + c] = dq[c] * u.s0; }
    return;
  }
  let j = x;
  let vj = u.a + (b * T + j) * 3u * N + 2u * N + h * D;
  var dk: array<f32, MAX_HEAD>;
  var dv: array<f32, MAX_HEAD>;
  for (var i = j; i < T; i++) {
    let dOi = u.e + (b * T + i) * N + h * D;
    let Di = headDot(dOi, u.b + (b * T + i) * N + h * D);
    let p = work[pRow + i * T + j];
    let ds = p * (headDot(dOi, vj) - Di);
    let q = u.a + (b * T + i) * 3u * N + h * D;
    for (var c = 0u; c < D; c++) {
      dk[c] += ds * work[q + c];
      dv[c] += p * work[dOi + c];
    }
  }
  for (var c = 0u; c < D; c++) {
    work[u.c + (b * T + j) * 3u * N + N + h * D + c] = dk[c] * u.s0;
    work[u.c + (b * T + j) * 3u * N + 2u * N + h * D + c] = dv[c];
  }
}

/// Cross-entropy of each position's scores against the letter that follows it: the loss to c, and (flags & 1) the
/// gradient (softmax − one-hot) · s0 to e.
@compute @workgroup_size(64)
fn crossEntropy(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let r = flatId(gid, nwg);
  if (r >= u.B * u.T) { return; }
  let b = r / u.T;
  let t = r % u.T;
  let row = u.a + r * u.V;
  let next = tokens[windowStart(b) + t + 1u];
  var mx = -3.0e38;
  for (var v = 0u; v < u.V; v++) { mx = max(mx, work[row + v]); }
  var sum = 0.0;
  for (var v = 0u; v < u.V; v++) { sum += exp(work[row + v] - mx); }
  let lse = mx + log(sum);
  work[u.c + r] = lse - work[row + next];
  if ((u.flags & 1u) != 0u) {
    for (var v = 0u; v < u.V; v++) {
      work[u.e + r * u.V + v] = (exp(work[row + v] - lse) - select(0.0, 1.0, v == next)) * u.s0;
    }
  }
}

/// Squared gradient, summed in NORM_PARTS contiguous chunks.
@compute @workgroup_size(64)
fn normPartial(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = flatId(gid, nwg);
  if (i >= NORM_PARTS) { return; }
  let chunk = (u.K + NORM_PARTS - 1u) / NORM_PARTS;
  var s = 0.0;
  let end = min(u.K, (i + 1u) * chunk);
  for (var p = i * chunk; p < end; p++) { s += opt[u.a + p] * opt[u.a + p]; }
  opt[u.c + i] = s;
}

/// The gradient's norm and the factor that clips it to at most s0.
@compute @workgroup_size(64)
fn normFinish(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  if (flatId(gid, nwg) != 0u) { return; }
  var s = 0.0;
  for (var i = 0u; i < NORM_PARTS; i++) { s += opt[u.c + i]; }
  let norm = sqrt(s);
  opt[u.c + NORM_PARTS] = norm;
  opt[u.c + NORM_PARTS + 1u] = min(1.0, u.s0 / (norm + 1e-6));
}

/// AdamW on K parameters, the first M with decoupled weight decay. The clip factor is opt[d]; the step's learning
/// rate, bias corrections and decay are opt[e … e + 3].
@compute @workgroup_size(64)
fn adam(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let p = flatId(gid, nwg);
  if (p >= u.K) { return; }
  let g = opt[u.a + p] * opt[u.d];
  let lr = opt[u.e];
  let bc1 = opt[u.e + 1u];
  let bc2 = opt[u.e + 2u];
  let wd = opt[u.e + 3u];
  let m = u.s0 * opt[u.b + p] + (1.0 - u.s0) * g;
  let v = u.s1 * opt[u.c + p] + (1.0 - u.s1) * g * g;
  opt[u.b + p] = m;
  opt[u.c + p] = v;
  var w = params[p];
  if (p < u.M) { w *= 1.0 - lr * wd; }
  params[p] = w - lr * (m / bc1) / (sqrt(v / bc2) + u.s2);
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

/// Writing: for each of B lines, the next letter from the last position's scores (temperature s0; 0 = the most
/// likely), drawn with a hashed uniform number (seed K, line, counter), written just after the window. With flags & 1,
/// the model's surprise at the letter (−ln p at temperature 1) goes to work[c + b·h + counter].
@compute @workgroup_size(64)
fn sample(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let b = flatId(gid, nwg);
  if (b >= u.B) { return; }
  var n = 0u;
  if (u.gen > 0u) { n = tokens[u.gen]; }
  let s = windowStart(b);
  let row = u.a + (b * u.T + u.T - 1u) * u.V;
  var mx = -3.0e38;
  var best = 0u;
  for (var v = 0u; v < u.V; v++) {
    if (work[row + v] > mx) {
      mx = work[row + v];
      best = v;
    }
  }
  var pick = best;
  if (u.s0 > 0.0) {
    var total = 0.0;
    for (var v = 0u; v < u.V; v++) { total += exp((work[row + v] - mx) / u.s0); }
    let uniform01 = f32(pcg(u.K ^ pcg(b ^ pcg(n))) >> 8u) / 16777216.0;
    let goal = uniform01 * total;
    var cum = 0.0;
    pick = u.V - 1u;
    for (var v = 0u; v < u.V; v++) {
      cum += exp((work[row + v] - mx) / u.s0);
      if (cum > goal) {
        pick = v;
        break;
      }
    }
  }
  tokens[s + u.T] = pick;
  if ((u.flags & 1u) != 0u && n < u.h) {
    var sum = 0.0;
    for (var v = 0u; v < u.V; v++) { sum += exp(work[row + v] - mx); }
    work[u.c + b * u.h + n] = mx + log(sum) - work[row + pick];
  }
}

/// Moves the writing counter on by one letter.
@compute @workgroup_size(64)
fn advance(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x == 0u) { tokens[u.gen] += 1u; }
}
