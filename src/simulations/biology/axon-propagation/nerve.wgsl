// The Nerve view: every fibre of the nerve as one row (largest at the top), coloured by its voltage along the
// nerve now. A stimulus sends a front of spikes across: fast fibres race ahead, thin C fibres trail far behind.

struct View {
  resolution: vec2f,
  length: f32, // cm
  fibres: u32,
  top: f32, // uv band of the picture
  bottom: f32,
  margin: f32,
  pad: f32,
}

struct Fibre {
  offset: u32,
  count: u32,
  pad0: u32,
  pad1: u32,
}

struct Comp {
  a: vec4f,
  b: vec4f,
}

struct Dyn {
  s: vec4f,
  r: vec4f,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> fibres: array<Fibre>;
@group(0) @binding(2) var<storage, read> comps: array<Comp>;
@group(0) @binding(3) var<storage, read> dyn: array<Dyn>;

fn voltageColor(v: f32) -> vec3f {
  let t = clamp((v + 80.0) / 120.0, 0.0, 1.0);
  let c0 = vec3f(0.03, 0.05, 0.16);
  let c1 = vec3f(0.12, 0.22, 0.55);
  let c2 = vec3f(0.75, 0.2, 0.55);
  let c3 = vec3f(0.98, 0.55, 0.2);
  let c4 = vec3f(1.0, 0.95, 0.7);
  if (t < 0.15) { return mix(c0, c1, t / 0.15); }
  if (t < 0.45) { return mix(c1, c2, (t - 0.15) / 0.3); }
  if (t < 0.75) { return mix(c2, c3, (t - 0.45) / 0.3); }
  return mix(c3, c4, (t - 0.75) / 0.25);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let bg = vec3f(0.016, 0.02, 0.03);
  let u = (uv.x - view.margin) / (1.0 - 2.0 * view.margin);
  if (u < 0.0 || u > 1.0 || uv.y < view.top || uv.y > view.bottom || view.fibres == 0u) { return vec4f(bg, 1.0); }
  let row = min(u32((uv.y - view.top) / (view.bottom - view.top) * f32(view.fibres)), view.fibres - 1u);
  let first = fibres[row].offset;
  let n = fibres[row].count;
  let x = u * view.length;
  var lo = 0u;
  var hi = n - 1u;
  while (lo < hi) {
    let mid = (lo + hi + 1u) / 2u;
    let edge = 0.5 * (comps[first + mid - 1u].a.z + comps[first + mid].a.z);
    if (x >= edge) { lo = mid; } else { hi = mid - 1u; }
  }
  return vec4f(voltageColor(dyn[first + lo].s.x), 1.0);
}
