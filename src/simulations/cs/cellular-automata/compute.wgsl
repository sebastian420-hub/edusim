@group(0) @binding(0) var<storage, read> cellsIn: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellsOut: array<u32>;
@group(0) @binding(2) var<uniform> params: SimParams;

struct SimParams {
  width: u32,
  height: u32,
  rule_birth: u32,   // bitmask for birth rules (e.g., 0b000001000 = B3)
  rule_survive: u32, // bitmask for survive rules (e.g., 0b000001100 = S23)
}

@compute @workgroup_size(8, 8) // keep in sync with WORKGROUP_SIZE in workgroup.ts
fn main(@builtin(global_invocation_id) id: vec3u) {
  let x = id.x;
  let y = id.y;
  if (x >= params.width || y >= params.height) { return; }
  
  // Count 8 neighbors with toroidal wrapping
  var neighbors = 0u;
  for (var dy = -1i; dy <= 1i; dy++) {
    for (var dx = -1i; dx <= 1i; dx++) {
      if (dx == 0i && dy == 0i) { continue; }
      let nx = (i32(x) + dx + i32(params.width)) % i32(params.width);
      let ny = (i32(y) + dy + i32(params.height)) % i32(params.height);
      neighbors += cellsIn[u32(ny) * params.width + u32(nx)];
    }
  }
  
  let idx = y * params.width + x;
  let alive = cellsIn[idx];
  let neighborBit = 1u << neighbors;
  
  // Apply birth/survive rules via bitmask
  if (alive == 1u) {
    cellsOut[idx] = select(0u, 1u, (params.rule_survive & neighborBit) != 0u);
  } else {
    cellsOut[idx] = select(0u, 1u, (params.rule_birth & neighborBit) != 0u);
  }
}
