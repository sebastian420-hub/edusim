// 2D wave interference and diffraction, evaluated analytically per pixel.
//
// Every source is monochromatic, so the field is a phasor (C, S): the instantaneous height is
//   h(p, t) = C(p) cos(wt) + S(p) sin(wt)
// and the time-averaged intensity is (C^2 + S^2) / 2 -- the stationary interference pattern.
//
// Slits are modelled with the Huygens-Fresnel principle: each slit is a row of secondary point
// sources, driven by a plane wave arriving from the left. Left of the barrier we show the incident
// plane wave itself.

struct Uniforms {
  resolution: vec2f,
  time: f32,
  frequency: f32,
  amplitude: f32,
  damping: f32,
  waveSpeed: f32,
  separation: f32,
  slitWidth: f32,
  mode: u32,     // 0 = one point, 1 = two points, 2 = single slit, 3 = double slit
  viewMode: u32, // 0 = amplitude, 1 = intensity, 2 = 3D water
}

@group(0) @binding(0) var<uniform> u: Uniforms;

const TAU = 6.28318530718;
const SOURCE_X = -1.5;       // x of point sources and of the barrier
const BARRIER_HALF = 0.04;   // half thickness of the barrier
const SLIT_SAMPLES = 24;     // secondary sources per slit

fn wavenumber() -> f32 {
  return TAU * u.frequency / u.waveSpeed;
}

// Phasor of a cylindrical wave from `src`, weighted by `weight`.
fn cylindrical(p: vec2f, src: vec2f, weight: f32) -> vec2f {
  let r = length(p - src);
  let a = weight * u.amplitude * exp(-u.damping * r) / (sqrt(r) + 0.1);
  let phase = wavenumber() * r;
  return a * vec2f(cos(phase), sin(phase));
}

// Phasor of a single slit centred at (SOURCE_X, centreY), as a row of Huygens sources.
// Each slit is normalised to emit a unit-strength cylindrical wave whatever its width, so the slit
// width only controls the diffraction envelope (narrow slit = wide spreading) and the pattern stays
// bright enough to read.
fn slit(p: vec2f, centreY: f32) -> vec2f {
  let dy = u.slitWidth / f32(SLIT_SAMPLES);
  var sum = vec2f(0.0);
  for (var i = 0; i < SLIT_SAMPLES; i++) {
    let y = centreY + (f32(i) + 0.5 - 0.5 * f32(SLIT_SAMPLES)) * dy;
    let r = length(p - vec2f(SOURCE_X, y));
    let a = u.amplitude * exp(-u.damping * r) / (f32(SLIT_SAMPLES) * sqrt(1.0 + r));
    let phase = wavenumber() * r;
    sum += a * vec2f(cos(phase), sin(phase));
  }
  return sum;
}

fn slitCentres() -> vec2f {
  // Centres of the (up to two) slits; the second is only used in double-slit mode.
  if (u.mode == 3u) {
    return vec2f(0.5 * u.separation, -0.5 * u.separation);
  }
  return vec2f(0.0, 0.0);
}

fn isBarrier(p: vec2f) -> bool {
  if (u.mode < 2u || abs(p.x - SOURCE_X) > BARRIER_HALF) {
    return false;
  }
  let c = slitCentres();
  let half = 0.5 * u.slitWidth;
  var open = abs(p.y - c.x) < half;
  if (u.mode == 3u) {
    open = open || abs(p.y - c.y) < half;
  }
  return !open;
}

fn phasor(p: vec2f) -> vec2f {
  if (u.mode == 0u) {
    return cylindrical(p, vec2f(SOURCE_X, 0.0), 1.0);
  }
  if (u.mode == 1u) {
    return cylindrical(p, vec2f(SOURCE_X, 0.5 * u.separation), 1.0)
         + cylindrical(p, vec2f(SOURCE_X, -0.5 * u.separation), 1.0);
  }
  // Slit modes: incident plane wave on the left, Huygens sources on the right.
  if (p.x < SOURCE_X) {
    let phase = wavenumber() * (p.x - SOURCE_X);
    return u.amplitude * vec2f(cos(phase), sin(phase));
  }
  let c = slitCentres();
  var sum = slit(p, c.x);
  if (u.mode == 3u) {
    sum += slit(p, c.y);
  }
  return sum;
}

fn height(p: vec2f, wt: f32) -> f32 {
  let ph = phasor(p);
  return ph.x * cos(wt) + ph.y * sin(wt);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let aspect = u.resolution.x / u.resolution.y;
  // y is up in simulation space; uv.y points down.
  let p = vec2f((uv.x - 0.5) * aspect * 4.0, (0.5 - uv.y) * 4.0);
  let wt = TAU * u.frequency * u.time;

  if (isBarrier(p)) {
    return vec4f(0.35, 0.37, 0.42, 1.0);
  }

  var color: vec3f;
  if (u.viewMode == 0u) {
    let h = height(p, wt);
    let t = clamp(h * 0.35 + 0.5, 0.0, 1.0);
    color = mix(vec3f(0.05, 0.1, 0.6), vec3f(0.85, 0.15, 0.1), t);
  } else if (u.viewMode == 1u) {
    let ph = phasor(p);
    let intensity = 0.5 * dot(ph, ph) / (u.amplitude * u.amplitude);
    let t = 1.0 - exp(-1.2 * intensity);
    color = mix(vec3f(0.01, 0.02, 0.1), vec3f(0.1, 0.8, 1.0), t) + vec3f(0.9) * t * t * t;
  } else {
    let eps = 0.01;
    let h0 = height(p, wt);
    let dx = (height(p + vec2f(eps, 0.0), wt) - h0) / eps;
    let dy = (height(p + vec2f(0.0, eps), wt) - h0) / eps;
    let normal = normalize(vec3f(-dx * 0.25, -dy * 0.25, 1.0));
    let light = normalize(vec3f(0.3, 0.5, 1.0));
    let diffuse = max(dot(normal, light), 0.0);
    let specular = pow(max(dot(reflect(-light, normal), vec3f(0.0, 0.0, 1.0)), 0.0), 64.0);
    color = vec3f(0.05, 0.2, 0.4) + vec3f(0.1, 0.4, 0.8) * diffuse + vec3f(1.0) * specular * 0.8;
  }
  return vec4f(color, 1.0);
}
