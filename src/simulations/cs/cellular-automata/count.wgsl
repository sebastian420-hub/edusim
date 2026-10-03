// Counts the live cells of a generation and computes an order-independent fingerprint of the grid
// (the wrapping sum of a per-cell hash). Results go into slot `params.slot` of a ring buffer of
// (count, hash) pairs; the CPU reads the ring back a few times a second. Each workgroup reduces its
// 8x8 tile locally first, so only one global atomic per workgroup is needed.
// Keep cell_hash in sync with cellHash() in life.ts.

struct CountParams {
  width: u32,
  height: u32,
  slot: u32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: CountParams;

var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_hash: atomic<u32>;

fn cell_hash(index: u32) -> u32 {
  var h = (index + 1u) * 0x9e3779b1u;
  h ^= h >> 16u;
  h *= 0x85ebca6bu;
  h ^= h >> 13u;
  h *= 0xc2b2ae35u;
  h ^= h >> 16u;
  return h;
}

@compute @workgroup_size(8, 8) // keep in sync with WORKGROUP_SIZE in workgroup.ts
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) local_index: u32) {
  if (local_index == 0u) {
    atomicStore(&wg_count, 0u);
    atomicStore(&wg_hash, 0u);
  }
  workgroupBarrier();

  if (id.x < params.width && id.y < params.height) {
    let idx = id.y * params.width + id.x;
    if (cells[idx] == 1u) {
      atomicAdd(&wg_count, 1u);
      atomicAdd(&wg_hash, cell_hash(idx));
    }
  }
  workgroupBarrier();

  if (local_index == 0u) {
    atomicAdd(&stats[params.slot * 2u], atomicLoad(&wg_count));
    atomicAdd(&stats[params.slot * 2u + 1u], atomicLoad(&wg_hash));
  }
}
