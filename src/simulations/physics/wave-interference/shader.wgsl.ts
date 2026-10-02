export const shaderSource = `
struct Uniforms {
  resolution: vec2f,
  time: f32,
  frequency: f32,
  amplitude: f32,
  damping: f32,
  waveSpeed: f32,
  sourceCount: f32,
  source1: vec2f,
  source2: vec2f,
  slitWidth: f32,
  slitSeparation: f32,
  viewMode: f32,
  pad: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let aspect = u.resolution.x / u.resolution.y;
  let p = (uv - 0.5) * vec2f(aspect, 1.0) * 4.0;
  
  let k = 6.28318 * u.frequency / u.waveSpeed;
  let omega = 6.28318 * u.frequency;
  
  var wave = 0.0;
  var inBarrier = false;
  
  // Fake barrier logic: if sources are at x=0 (double slit preset), we draw a barrier.
  // We'll say if source1.x == 0.0 and it's a double slit, draw barrier.
  let isDoubleSlit = u.sourceCount > 1.5 && abs(u.source1.x) < 0.01;
  let barrierThickness = 0.05;
  
  if (isDoubleSlit && abs(p.x) < barrierThickness) {
    let distToSlit1 = abs(p.y - u.source1.y);
    let distToSlit2 = abs(p.y - u.source2.y);
    if (distToSlit1 > u.slitWidth * 0.5 && distToSlit2 > u.slitWidth * 0.5) {
      inBarrier = true;
    }
  }
  
  if (inBarrier) {
    return vec4f(0.2, 0.2, 0.2, 1.0);
  }
  
  // Source 1
  let r1 = length(p - u.source1);
  wave += u.amplitude * cos(k * r1 - omega * u.time) / (sqrt(r1) + 0.1) * exp(-u.damping * max(r1, 0.0));
  
  // Source 2
  if (u.sourceCount > 1.5) {
    let r2 = length(p - u.source2);
    wave += u.amplitude * cos(k * r2 - omega * u.time) / (sqrt(r2) + 0.1) * exp(-u.damping * max(r2, 0.0));
  }
  
  // Color mapping based on view mode
  var color: vec3f;
  if (u.viewMode < 0.5) {
    let t = clamp(wave * 0.5 + 0.5, 0.0, 1.0);
    color = mix(vec3f(0.05, 0.1, 0.6), vec3f(0.8, 0.15, 0.1), t);
  } else if (u.viewMode < 1.5) {
    let intensity = wave * wave;
    color = vec3f(intensity * 0.4, intensity * 0.8, intensity * 1.2);
  } else {
    let eps = 0.01;
    let r1x = length(p + vec2f(eps, 0.0) - u.source1);
    let r1y = length(p + vec2f(0.0, eps) - u.source1);
    var wx = u.amplitude * cos(k * r1x - omega * u.time) / (sqrt(r1x) + 0.1) * exp(-u.damping * max(r1x, 0.0));
    var wy = u.amplitude * cos(k * r1y - omega * u.time) / (sqrt(r1y) + 0.1) * exp(-u.damping * max(r1y, 0.0));
    
    if (u.sourceCount > 1.5) {
      let r2x = length(p + vec2f(eps, 0.0) - u.source2);
      let r2y = length(p + vec2f(0.0, eps) - u.source2);
      wx += u.amplitude * cos(k * r2x - omega * u.time) / (sqrt(r2x) + 0.1) * exp(-u.damping * max(r2x, 0.0));
      wy += u.amplitude * cos(k * r2y - omega * u.time) / (sqrt(r2y) + 0.1) * exp(-u.damping * max(r2y, 0.0));
    }
    
    let dx = (wx - wave) / eps;
    let dy = (wy - wave) / eps;
    let normal = normalize(vec3f(-dx, -dy, 1.0));
    let light = normalize(vec3f(0.3, 0.5, 1.0));
    let diffuse = max(dot(normal, light), 0.0);
    let specular = pow(max(dot(reflect(-light, normal), vec3f(0.0, 0.0, 1.0)), 0.0), 64.0);
    let baseColor = vec3f(0.05, 0.2, 0.4);
    color = baseColor + vec3f(0.1, 0.4, 0.8) * diffuse + vec3f(1.0) * specular * 0.8;
  }
  
  return vec4f(color, 1.0);
}
`;
