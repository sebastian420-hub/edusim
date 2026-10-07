struct State {
    V: f32,
    m: f32,
    h: f32,
    n: f32,
}

struct HistoryMeta {
    index: u32,
    step_count: u32,
    pad0: u32,
    pad1: u32,
}

struct RenderParams {
    resolution: vec2f,
    time: f32,
    pad: f32,
}

@group(0) @binding(0) var<storage, read> history: array<State>;
@group(0) @binding(1) var<storage, read> histMeta: HistoryMeta;
@group(0) @binding(2) var<uniform> params: RenderParams;

fn sdLine(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

const NUM_SAMPLES = 2048u;

// Sample `k` (0 = oldest) of the ring buffer.
fn sample_at(k: u32) -> State {
    let start_idx = histMeta.index; // the head is the oldest sample once the ring is full
    return history[(start_idx + min(k, NUM_SAMPLES - 1u)) % NUM_SAMPLES];
}

// Plot position (in pixels) of channel value `y` (0..1, up) at sample k.
fn plot(k: u32, y: f32) -> vec2f {
    return vec2f(f32(k) / f32(NUM_SAMPLES) * params.resolution.x, (1.0 - y) * params.resolution.y);
}

fn volt01(v: f32) -> f32 {
    return (v + 100.0) / 160.0; // -100..60 mV -> 0..1
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    var color = vec3f(0.02, 0.05, 0.02); // Dark green bg

    // Grid lines
    let grid = fract(uv * vec2f(10.0, 8.0));
    if (grid.x < 0.02 || grid.y < 0.02) {
        color = vec3f(0.1, 0.2, 0.1);
    }

    // Distance to each trace, measured in pixels and over the few segments around this pixel so that
    // steep spike edges are drawn as continuous lines.
    let p = uv * params.resolution;
    let centre = u32(clamp(uv.x * f32(NUM_SAMPLES), 0.0, f32(NUM_SAMPLES - 1u)));
    var dV = 1e9;
    var dm = 1e9;
    var dh = 1e9;
    var dn = 1e9;
    for (var o = -3; o <= 3; o++) {
        let k0 = u32(clamp(i32(centre) + o, 0, i32(NUM_SAMPLES) - 2));
        let s0 = sample_at(k0);
        let s1 = sample_at(k0 + 1u);
        dV = min(dV, sdLine(p, plot(k0, volt01(s0.V)), plot(k0 + 1u, volt01(s1.V))));
        dm = min(dm, sdLine(p, plot(k0, s0.m), plot(k0 + 1u, s1.m)));
        dh = min(dh, sdLine(p, plot(k0, s0.h), plot(k0 + 1u, s1.h)));
        dn = min(dn, sdLine(p, plot(k0, s0.n), plot(k0 + 1u, s1.n)));
    }

    // Okabe–Ito colours (distinguishable with every common colour-vision deficiency); the voltage, the main
    // trace, is also drawn thicker so it never depends on colour alone. Keep in sync with LEGEND in controls.tsx.
    let line_width = 1.2; // px
    let glow = 3.0;       // px
    color += vec3f(0.337, 0.706, 0.914) * smoothstep(glow, 0.0, dV - 2.0 * line_width); // voltage: sky blue
    color += vec3f(0.902, 0.624, 0.0) * smoothstep(glow, 0.0, dm - line_width);   // m gate: orange
    color += vec3f(0.8, 0.475, 0.655) * smoothstep(glow, 0.0, dh - line_width);   // h gate: reddish purple
    color += vec3f(0.941, 0.894, 0.259) * smoothstep(glow, 0.0, dn - line_width); // n gate: yellow

    return vec4f(color, 1.0);
}
