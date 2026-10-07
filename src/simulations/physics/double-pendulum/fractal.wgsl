// Colours the flip-time map: one pendulum per pixel, colour = how soon it first flipped (a perceptually uniform,
// colour-blind-safe "viridis" ramp on a log scale), near-black = not flipped (yet). The square map is centred in
// the canvas. The energy boundary is outlined: inside it flips are impossible (2 cos θ₁ + cos θ₂ > 1 for equal
// masses and lengths).

struct MapParams {
  resolution: vec2f,
  n: u32,
  showBoundary: u32,
  cx: f32,
  cy: f32,
  span: f32,
  tMax: f32,
  /** Energy boundary a·cos θ₁ + b·cos θ₂ = c (see flipBoundary in pendulum.ts). */
  a: f32,
  b: f32,
  c: f32,
  pad: f32,
}

@group(0) @binding(0) var<storage, read> flip: array<f32>;
@group(0) @binding(1) var<uniform> params: MapParams;

// Polynomial fit of matplotlib's viridis (Mattias Hejl / Inigo Quilez style approximation).
fn viridis(t: f32) -> vec3f {
  let c0 = vec3f(0.2777, 0.0054, 0.3341);
  let c1 = vec3f(0.1050, 1.4046, 1.3846);
  let c2 = vec3f(-0.3309, 0.2148, 0.0951);
  let c3 = vec3f(-4.6342, -5.7991, -19.3324);
  let c4 = vec3f(6.2283, 14.1799, 56.6906);
  let c5 = vec3f(4.7764, -13.7451, -65.3530);
  let c6 = vec3f(-5.4355, 4.6459, 26.3124);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  // Fit the square map into the canvas.
  let res = params.resolution;
  let side = min(res.x, res.y);
  let p = (uv * res - 0.5 * (res - vec2f(side))) / side; // [0,1]² inside the square
  if (p.x < 0.0 || p.y < 0.0 || p.x >= 1.0 || p.y >= 1.0) {
    return vec4f(0.02, 0.025, 0.035, 1.0);
  }
  let n = params.n;
  let i = min(u32(p.x * f32(n)), n - 1u);
  let j = min(u32(p.y * f32(n)), n - 1u);
  let t = flip[j * n + i];
  var color = vec3f(0.03, 0.035, 0.05);
  if (t >= 0.0) {
    // Early flips bright yellow, late flips deep blue: log scale from 0.1 s to tMax.
    let x = clamp(log(max(t, 0.1) / 0.1) / log(params.tMax / 0.1), 0.0, 1.0);
    color = viridis(1.0 - x);
  }
  if (params.showBoundary == 1u) {
    let t1 = params.cx + (p.x - 0.5) * params.span;
    let t2 = params.cy + (0.5 - p.y) * params.span;
    let f = params.a * cos(t1) + params.b * cos(t2) - params.c;
    // How much f changes per pixel, from its gradient (fwidth is not allowed after the early return above).
    let w = length(vec2f(params.a * sin(t1), params.b * sin(t2))) * params.span / side;
    let line = 1.0 - smoothstep(0.0, 1.5 * w, abs(f));
    color = mix(color, vec3f(0.95, 0.95, 0.95), 0.8 * line);
  }
  return vec4f(color, 1.0);
}
