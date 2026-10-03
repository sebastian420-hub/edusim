/** Slots in the GPU ring buffer of per-generation (count, fingerprint) pairs. */
export const STATS_SLOTS = 1024;

/** Selectable grid edge lengths (the grid is always square). */
export const GRID_SIZES = [256, 512, 1024, 2048] as const;
