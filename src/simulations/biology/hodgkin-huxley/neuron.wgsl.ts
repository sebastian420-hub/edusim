export const shaderSource = `
struct Params {
    g_Na: f32,
    g_K: f32,
    g_L: f32,
    E_Na: f32,
    E_K: f32,
    E_L: f32,
    C_m: f32,
    I_inj: f32,
    temperature: f32,
    dt: f32,
    substeps: u32,
    pulse_mode: u32,
    time: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> state: State;
@group(0) @binding(2) var<storage, read_write> history: array<State>;
@group(0) @binding(3) var<storage, read_write> meta: HistoryMeta;

fn get_current(t_ms: f32) -> f32 {
    let t_mod = t_ms - 25.0 * floor(t_ms / 25.0); // Repeat every 25ms
    if (params.pulse_mode == 0u) {
        return params.I_inj;
    } else if (params.pulse_mode == 1u) {
        if (t_mod >= 5.0 && t_mod <= 6.0) {
            return params.I_inj;
        }
        return 0.0;
    } else { // Twin pulse
        if ((t_mod >= 5.0 && t_mod <= 6.0) || (t_mod >= 13.0 && t_mod <= 14.0)) {
            return params.I_inj;
        }
        return 0.0;
    }
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var V = state.V;
    var m = state.m;
    var h = state.h;
    var n = state.n;
    
    let dt = params.dt;
    let phi = pow(3.0, (params.temperature - 6.3) / 10.0);
    
    var time = params.time;

    for (var i = 0u; i < params.substeps; i = i + 1u) {
        let I_app = get_current(time);
        
        let V_shift = V + 65.0;
        
        // Alpha/Beta for m (Na+ activation)
        var am = 0.1 * (25.0 - V_shift) / (exp((25.0 - V_shift) / 10.0) - 1.0);
        if (abs(V_shift - 25.0) < 0.001) { am = 1.0; } 
        let bm = 4.0 * exp(-V_shift / 18.0);
        
        // Alpha/Beta for h (Na+ inactivation)
        let ah = 0.07 * exp(-V_shift / 20.0);
        let bh = 1.0 / (exp((30.0 - V_shift) / 10.0) + 1.0);
        
        // Alpha/Beta for n (K+ activation)
        var an = 0.01 * (10.0 - V_shift) / (exp((10.0 - V_shift) / 10.0) - 1.0);
        if (abs(V_shift - 10.0) < 0.001) { an = 0.1; }
        let bn = 0.125 * exp(-V_shift / 80.0);
        
        // Apply temperature factor
        let am_t = am * phi; let bm_t = bm * phi;
        let ah_t = ah * phi; let bh_t = bh * phi;
        let an_t = an * phi; let bn_t = bn * phi;
        
        // Rush-Larsen integration
        let m_inf = am_t / (am_t + bm_t);
        let tau_m = 1.0 / (am_t + bm_t);
        m = m_inf + (m - m_inf) * exp(-dt / tau_m);
        
        let h_inf = ah_t / (ah_t + bh_t);
        let tau_h = 1.0 / (ah_t + bh_t);
        h = h_inf + (h - h_inf) * exp(-dt / tau_h);
        
        let n_inf = an_t / (an_t + bn_t);
        let tau_n = 1.0 / (an_t + bn_t);
        n = n_inf + (n - n_inf) * exp(-dt / tau_n);
        
        // Currents
        let I_Na = params.g_Na * m * m * m * h * (V - params.E_Na);
        let I_K = params.g_K * n * n * n * n * (V - params.E_K);
        let I_L = params.g_L * (V - params.E_L);
        
        V = V + dt * (I_app - I_Na - I_K - I_L) / params.C_m;
        time = time + dt;
    }
    
    state.V = V;
    state.m = m;
    state.h = h;
    state.n = n;
    
    let idx = meta.index;
    history[idx] = state;
    meta.index = (idx + 1u) % 2048u;
}
`;
