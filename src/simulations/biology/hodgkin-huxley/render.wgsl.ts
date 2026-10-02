export const shaderSource = `
struct State {
    V: f32,
    m: f32,
    h: f32,
    n: f32,
}

struct HistoryMeta {
    index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct RenderParams {
    resolution: vec2f,
    time: f32,
    pad: f32,
}

@group(0) @binding(0) var<storage, read> history: array<State>;
@group(0) @binding(1) var<storage, read> meta: HistoryMeta;
@group(0) @binding(2) var<uniform> params: RenderParams;

fn sdLine(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    // Background and Grid
    var color = vec3f(0.02, 0.05, 0.02); // Dark green bg
    
    // Grid lines
    let grid_uv = uv * vec2f(10.0, 8.0);
    let grid = fract(grid_uv);
    if (grid.x < 0.02 || grid.y < 0.02) {
        color = vec3f(0.1, 0.2, 0.1);
    }
    
    // Map screen x to history buffer (0 to 2048)
    let num_samples = 2048.0;
    
    // Find sample indices for this pixel
    let sample_x = uv.x * num_samples;
    let idx0_offset = u32(clamp(sample_x, 0.0, num_samples - 1.0));
    let idx1_offset = min(idx0_offset + 1u, 2047u);
    
    let head = meta.index;
    let start_idx = (head + 2048u - u32(num_samples)) % 2048u;
    
    let i0 = (start_idx + idx0_offset) % 2048u;
    let i1 = (start_idx + idx1_offset) % 2048u;
    
    let s0 = history[i0];
    let s1 = history[i1];
    
    let y_v0 = (s0.V + 100.0) / 160.0; // -100 to 60 -> 0 to 1
    let y_v1 = (s1.V + 100.0) / 160.0;
    
    let y_m0 = s0.m; let y_m1 = s1.m;
    let y_h0 = s0.h; let y_h1 = s1.h;
    let y_n0 = s0.n; let y_n1 = s1.n;
    
    // Screen coordinates of segments
    let p0_x = f32(idx0_offset) / num_samples;
    let p1_x = f32(idx1_offset) / num_samples;
    
    let v_dist = sdLine(uv, vec2f(p0_x, 1.0 - y_v0), vec2f(p1_x, 1.0 - y_v1));
    let m_dist = sdLine(uv, vec2f(p0_x, 1.0 - y_m0), vec2f(p1_x, 1.0 - y_m1));
    let h_dist = sdLine(uv, vec2f(p0_x, 1.0 - y_h0), vec2f(p1_x, 1.0 - y_h1));
    let n_dist = sdLine(uv, vec2f(p0_x, 1.0 - y_n0), vec2f(p1_x, 1.0 - y_n1));
    
    // Render traces with glow
    let v_color = vec3f(0.2, 1.0, 0.4); // Voltage (green)
    let m_color = vec3f(1.0, 0.2, 0.2); // m gate (red)
    let h_color = vec3f(0.2, 0.4, 1.0); // h gate (blue)
    let n_color = vec3f(1.0, 0.8, 0.2); // n gate (yellow)
    
    let line_width = 0.002;
    let glow = 0.005;
    
    color += v_color * smoothstep(glow, 0.0, v_dist - line_width);
    color += m_color * smoothstep(glow, 0.0, m_dist - line_width);
    color += h_color * smoothstep(glow, 0.0, h_dist - line_width);
    color += n_color * smoothstep(glow, 0.0, n_dist - line_width);
    
    return vec4f(color, 1.0);
}
`;
