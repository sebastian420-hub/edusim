/**
 * CPU twin of the cellular-automaton shaders: one generation on a toroidal grid, plus the
 * population count and state fingerprint that count.wgsl computes on the GPU. Used for tests (the
 * GPU results are compared against this) and to check that challenges are solvable.
 */

/** Advances `cells` (row-major, 1 = alive) by one generation with birth/survive bitmasks (bit n = n neighbours). */
export function stepLife(cells: Uint32Array, width: number, height: number, birthMask: number, surviveMask: number): Uint32Array<ArrayBuffer> {
  const next = new Uint32Array(cells.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let neighbours = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          neighbours += cells[((y + dy + height) % height) * width + ((x + dx + width) % width)];
        }
      }
      const mask = cells[y * width + x] === 1 ? surviveMask : birthMask;
      next[y * width + x] = (mask >> neighbours) & 1;
    }
  }
  return next;
}

/**
 * Per-cell hash: the murmur3 32-bit finaliser applied to (index + 1). Must match `cell_hash` in count.wgsl.
 * (A plain linear function of the index would only depend on the pattern's centre of mass, so a horizontal and
 * a vertical blinker would collide.)
 */
export function cellHash(index: number): number {
  let h = Math.imul(index + 1, 0x9e3779b1);
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/**
 * Order-independent fingerprint of a grid: the wrapping (mod 2^32) sum of `cellHash` over live cells.
 * Equal grids always give equal fingerprints; a pattern that merely moves does not.
 */
export function stateHash(cells: Uint32Array): number {
  let hash = 0;
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] === 1) hash = (hash + cellHash(i)) >>> 0;
  }
  return hash;
}

export function countAlive(cells: Uint32Array): number {
  let n = 0;
  for (let i = 0; i < cells.length; i++) n += cells[i];
  return n;
}
