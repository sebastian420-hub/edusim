export type Pattern = {
  name: string;
  points: [number, number][];
};

export const patterns: Pattern[] = [
  {
    name: "Block",
    points: [[0, 0], [1, 0], [0, 1], [1, 1]],
  },
  {
    name: "Blinker",
    points: [[0, 0], [1, 0], [2, 0]],
  },
  {
    name: "Toad",
    points: [[1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1]],
  },
  {
    name: "Glider",
    points: [
      [1, 0],
      [2, 1],
      [0, 2], [1, 2], [2, 2]
    ]
  },
  {
    name: "Glider Gun (Gosper)",
    points: [
      [24, 0],
      [22, 1], [24, 1],
      [12, 2], [13, 2], [20, 2], [21, 2], [34, 2], [35, 2],
      [11, 3], [15, 3], [20, 3], [21, 3], [34, 3], [35, 3],
      [0, 4], [1, 4], [10, 4], [16, 4], [20, 4], [21, 4],
      [0, 5], [1, 5], [10, 5], [14, 5], [16, 5], [17, 5], [22, 5], [24, 5],
      [10, 6], [16, 6], [24, 6],
      [11, 7], [15, 7],
      [12, 8], [13, 8]
    ]
  },
  {
    name: "Pulsar",
    points: [
      [2, 0], [3, 0], [4, 0], [8, 0], [9, 0], [10, 0],
      [0, 2], [5, 2], [7, 2], [12, 2],
      [0, 3], [5, 3], [7, 3], [12, 3],
      [0, 4], [5, 4], [7, 4], [12, 4],
      [2, 5], [3, 5], [4, 5], [8, 5], [9, 5], [10, 5],
      [2, 7], [3, 7], [4, 7], [8, 7], [9, 7], [10, 7],
      [0, 8], [5, 8], [7, 8], [12, 8],
      [0, 9], [5, 9], [7, 9], [12, 9],
      [0, 10], [5, 10], [7, 10], [12, 10],
      [2, 12], [3, 12], [4, 12], [8, 12], [9, 12], [10, 12]
    ]
  },
  {
    name: "Acorn",
    points: [
      [1, 0],
      [3, 1],
      [0, 2], [1, 2], [4, 2], [5, 2], [6, 2]
    ]
  },
  {
    name: "R-Pentomino",
    points: [
      [1, 0], [2, 0],
      [0, 1], [1, 1],
      [1, 2]
    ]
  },
  {
    name: "Spaceship (LWSS)",
    points: [
      [1, 0], [4, 0],
      [0, 1],
      [0, 2], [4, 2],
      [0, 3], [1, 3], [2, 3], [3, 3]
    ]
  }
];

export function getPatternBounds(pattern: Pattern): { width: number, height: number } {
  if (pattern.points.length === 0) return { width: 0, height: 0 };
  let maxX = 0;
  let maxY = 0;
  for (const [x, y] of pattern.points) {
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { width: maxX + 1, height: maxY + 1 };
}

/** Rasterises `pattern` centred on a `width` x `height` grid; cells outside the grid are dropped. */
export function rasterizePattern(pattern: Pattern, width: number, height: number): Uint32Array<ArrayBuffer> {
  const cells = new Uint32Array(width * height);
  const bounds = getPatternBounds(pattern);
  const startX = Math.floor(width / 2 - bounds.width / 2);
  const startY = Math.floor(height / 2 - bounds.height / 2);
  for (const [px, py] of pattern.points) {
    const x = startX + px;
    const y = startY + py;
    if (x >= 0 && x < width && y >= 0 && y < height) cells[y * width + x] = 1;
  }
  return cells;
}

export const DEFAULT_PATTERN = "Glider Gun (Gosper)";
