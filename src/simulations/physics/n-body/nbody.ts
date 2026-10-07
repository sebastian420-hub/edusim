/**
 * CPU twin of the N-body shaders, plus everything measured from a simulation state: energies, momentum,
 * orbital elements and measured orbital periods. The GPU tests compare the shaders against these functions,
 * and the unit tests check the physics (Kepler's laws, conservation) against textbook results.
 *
 * Layout matches the GPU buffers: `pos[4i..4i+3] = (x, y, z, mass)`, `vel[4i..4i+3] = (vx, vy, vz, 0)`,
 * `acc[4i..4i+3] = (ax, ay, az, φ)` where φ is the gravitational potential at body i (G included).
 */

/** Orbit-lab units: astronomical units, years, solar masses. Then G = 4π², and Earth's period is 1 year. */
export const G_ORBIT = 4 * Math.PI * Math.PI;

export interface Body {
  /** Mass (solar masses in the orbit lab). */
  m: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export interface NBodyState {
  n: number;
  pos: Float64Array;
  vel: Float64Array;
  acc: Float64Array;
}

export function stateFromBodies(bodies: readonly Body[]): NBodyState {
  const n = bodies.length;
  const pos = new Float64Array(n * 4);
  const vel = new Float64Array(n * 4);
  bodies.forEach((b, i) => {
    pos.set([b.x, b.y, 0, b.m], i * 4);
    vel.set([b.vx, b.vy, 0, 0], i * 4);
  });
  return { n, pos, vel, acc: new Float64Array(n * 4) };
}

/** Read-only view of a state: the CPU twin's Float64Arrays or a GPU read-back's Float32Arrays. */
export interface StateLike {
  n: number;
  pos: ArrayLike<number>;
  vel: ArrayLike<number>;
  acc: ArrayLike<number>;
}

export function bodiesFromState(s: Pick<StateLike, "n" | "pos" | "vel">): Body[] {
  return Array.from({ length: s.n }, (_, i) => ({ m: s.pos[i * 4 + 3], x: s.pos[i * 4], y: s.pos[i * 4 + 1], vx: s.vel[i * 4], vy: s.vel[i * 4 + 1] }));
}

/**
 * Plummer-softened gravity, exactly as gravity.wgsl: a_i = G Σ_j m_j d / (|d|² + ε²)^{3/2},
 * φ_i = −G Σ_j m_j / √(|d|² + ε²), self-interaction skipped.
 */
export function computeAccelerations(s: NBodyState, G: number, eps2: number): void {
  const { n, pos, acc } = s;
  for (let i = 0; i < n; i++) {
    let ax = 0;
    let ay = 0;
    let az = 0;
    let phi = 0;
    const xi = pos[i * 4];
    const yi = pos[i * 4 + 1];
    const zi = pos[i * 4 + 2];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = pos[j * 4] - xi;
      const dy = pos[j * 4 + 1] - yi;
      const dz = pos[j * 4 + 2] - zi;
      const m = pos[j * 4 + 3];
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz + eps2);
      const inv3 = inv * inv * inv;
      ax += m * inv3 * dx;
      ay += m * inv3 * dy;
      az += m * inv3 * dz;
      phi -= m * inv;
    }
    acc.set([G * ax, G * ay, G * az, G * phi], i * 4);
  }
}

export type Integrator = "leapfrog" | "euler";

/**
 * One step. Leapfrog = kick–drift–kick (velocity Verlet): symplectic, bounded energy error. Euler = explicit
 * Euler (position from the old velocity): energy grows, orbits spiral outwards — kept for the numerics lesson.
 * Expects `acc` to hold the accelerations of the current positions, and leaves it holding those of the new ones.
 */
export function step(s: NBodyState, dt: number, G: number, eps2: number, integrator: Integrator = "leapfrog"): void {
  const { n, pos, vel, acc } = s;
  if (integrator === "leapfrog") {
    for (let i = 0; i < n * 4; i += 4) {
      for (let k = 0; k < 3; k++) {
        vel[i + k] += 0.5 * dt * acc[i + k];
        pos[i + k] += dt * vel[i + k];
      }
    }
    computeAccelerations(s, G, eps2);
    for (let i = 0; i < n * 4; i += 4) for (let k = 0; k < 3; k++) vel[i + k] += 0.5 * dt * acc[i + k];
  } else {
    for (let i = 0; i < n * 4; i += 4) {
      for (let k = 0; k < 3; k++) {
        pos[i + k] += dt * vel[i + k];
        vel[i + k] += dt * acc[i + k];
      }
    }
    computeAccelerations(s, G, eps2);
  }
}

export interface Conserved {
  kinetic: number;
  potential: number;
  total: number;
  /** z component of the total angular momentum. */
  angularMomentum: number;
  momentum: [number, number, number];
  centreOfMass: [number, number, number];
  mass: number;
}

/** Energies and momenta. `acc` must hold the potentials of the current positions (w component). */
export function conserved(s: StateLike): Conserved {
  const { n, pos, vel, acc } = s;
  let kinetic = 0;
  let potential = 0;
  let angularMomentum = 0;
  let mass = 0;
  const p: [number, number, number] = [0, 0, 0];
  const c: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < n * 4; i += 4) {
    const m = pos[i + 3];
    const [vx, vy, vz] = [vel[i], vel[i + 1], vel[i + 2]];
    kinetic += 0.5 * m * (vx * vx + vy * vy + vz * vz);
    potential += 0.5 * m * acc[i + 3]; // each pair counted twice
    angularMomentum += m * (pos[i] * vy - pos[i + 1] * vx);
    p[0] += m * vx;
    p[1] += m * vy;
    p[2] += m * vz;
    c[0] += m * pos[i];
    c[1] += m * pos[i + 1];
    c[2] += m * pos[i + 2];
    mass += m;
  }
  return { kinetic, potential, total: kinetic + potential, angularMomentum, momentum: p, centreOfMass: [c[0] / mass, c[1] / mass, c[2] / mass], mass };
}

/** Removes the centre-of-mass velocity, so the system as a whole does not drift off screen. */
export function toCentreOfMassFrame(bodies: readonly Body[]): Body[] {
  const mass = bodies.reduce((a, b) => a + b.m, 0);
  if (mass <= 0) return bodies.map((b) => ({ ...b }));
  const vx = bodies.reduce((a, b) => a + b.m * b.vx, 0) / mass;
  const vy = bodies.reduce((a, b) => a + b.m * b.vy, 0) / mass;
  return bodies.map((b) => ({ ...b, vx: b.vx - vx, vy: b.vy - vy }));
}

// ───────────────────────────── orbits ─────────────────────────────

export interface OrbitalElements {
  /** Distance to the primary. */
  r: number;
  /** Speed relative to the primary. */
  v: number;
  /** Specific orbital energy v²/2 − μ/r: negative = bound, ≥ 0 = escapes. */
  energy: number;
  /** Semi-major axis (Infinity when unbound). */
  a: number;
  /** Eccentricity: 0 circle, < 1 ellipse, ≥ 1 escapes. */
  e: number;
  /** Kepler period 2π√(a³/μ) (Infinity when unbound). */
  period: number;
}

/** Two-body (osculating) orbit of a body around a primary: exact for two bodies, a good guide for planets. */
export function orbitalElements(body: Body, primary: Body, G = G_ORBIT): OrbitalElements {
  const mu = G * (primary.m + body.m);
  const rx = body.x - primary.x;
  const ry = body.y - primary.y;
  const vx = body.vx - primary.vx;
  const vy = body.vy - primary.vy;
  const r = Math.hypot(rx, ry);
  const v2 = vx * vx + vy * vy;
  const energy = v2 / 2 - mu / r;
  const rv = rx * vx + ry * vy;
  const ex = ((v2 - mu / r) * rx - rv * vx) / mu;
  const ey = ((v2 - mu / r) * ry - rv * vy) / mu;
  const e = Math.hypot(ex, ey);
  const a = energy < 0 ? -mu / (2 * energy) : Infinity;
  return { r, v: Math.sqrt(v2), energy, a, e, period: a < Infinity ? 2 * Math.PI * Math.sqrt((a * a * a) / mu) : Infinity };
}

/** Speed for a circular orbit at distance r around mass M (plus the body's own mass m). */
export const circularSpeed = (r: number, M: number, m = 0, G = G_ORBIT) => Math.sqrt((G * (M + m)) / r);
/** Escape speed at distance r: √2 × the circular speed. */
export const escapeSpeed = (r: number, M: number, m = 0, G = G_ORBIT) => Math.SQRT2 * circularSpeed(r, M, m, G);

/** Index of the most massive body (the "star" that planets are measured around). */
export function primaryIndex(bodies: readonly Pick<Body, "m">[]): number {
  let best = 0;
  bodies.forEach((b, i) => {
    if (b.m > bodies[best].m) best = i;
  });
  return best;
}

/**
 * Measures orbital periods from samples of a body's angle around the primary: every full turn (±2π of
 * unwrapped angle since the first sample) is timed, with linear interpolation between samples. Samples must be
 * close enough that the angle moves less than half a turn between them.
 */
export class OrbitTracker {
  private start: number | undefined;
  private last = 0;
  private total = 0;
  private lastTime = 0;
  private turns = 0;
  private crossings: number[] = [];

  reset(): void {
    this.start = undefined;
    this.total = 0;
    this.turns = 0;
    this.crossings = [];
  }

  sample(time: number, angle: number): void {
    if (this.start === undefined) {
      this.start = time;
      this.last = angle;
      this.lastTime = time;
      this.crossings = [time];
      return;
    }
    let delta = angle - this.last;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta <= -Math.PI) delta += 2 * Math.PI;
    const before = this.total;
    const after = before + delta;
    // Each new full turn (in either direction) between the two samples is a crossing.
    const nextTurn = this.turns + 1;
    const target = nextTurn * 2 * Math.PI;
    if (Math.abs(after) >= target && Math.abs(before) < target) {
      const f = (target - Math.abs(before)) / (Math.abs(after) - Math.abs(before));
      this.crossings.push(this.lastTime + f * (time - this.lastTime));
      this.turns = nextTurn;
    }
    this.total = after;
    this.last = angle;
    this.lastTime = time;
  }

  /** Completed orbits since the first sample. */
  get orbits(): number {
    return this.turns;
  }

  /** Duration of the most recent completed orbit, or undefined before the first one. */
  get period(): number | undefined {
    const c = this.crossings;
    return c.length >= 2 ? c[c.length - 1] - c[c.length - 2] : undefined;
  }
}

// ───────────────────────────── galaxies ─────────────────────────────

/** Small deterministic PRNG (mulberry32), so a preset + seed always gives the same initial conditions. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Cloud {
  n: number;
  pos: Float64Array;
  vel: Float64Array;
  /** Per body: 0 = first galaxy/cluster, 1 = second, 2 = a central black hole. */
  group: Uint8Array;
}

function emptyCloud(n: number): Cloud {
  return { n, pos: new Float64Array(n * 4), vel: new Float64Array(n * 4), group: new Uint8Array(n) };
}

/** Moves the cloud into its centre-of-mass frame: no net momentum, centred on the origin. */
function centred(c: Cloud): Cloud {
  let m = 0;
  const sum = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < c.n * 4; i += 4) {
    const w = c.pos[i + 3];
    m += w;
    for (let k = 0; k < 3; k++) {
      sum[k] += w * c.pos[i + k];
      sum[3 + k] += w * c.vel[i + k];
    }
  }
  for (let i = 0; i < c.n * 4; i += 4) {
    for (let k = 0; k < 3; k++) {
      c.pos[i + k] -= sum[k] / m;
      c.vel[i + k] -= sum[3 + k] / m;
    }
  }
  return c;
}

function randomDirection(random: () => number): [number, number, number] {
  const z = 2 * random() - 1;
  const t = 2 * Math.PI * random();
  const s = Math.sqrt(1 - z * z);
  return [s * Math.cos(t), s * Math.sin(t), z];
}

/**
 * Plummer star cluster (G = 1, total mass 1, scale radius 1), sampled as in Aarseth, Hénon & Wielen (1974):
 * radii from the cumulative mass profile, speeds by rejection from the distribution function. In equilibrium.
 */
export function plummerCluster(n: number, seed = 1): Cloud {
  const random = rng(seed);
  const c = emptyCloud(n);
  for (let i = 0; i < n; i++) {
    let r: number;
    do r = 1 / Math.sqrt(Math.pow(random() * 0.999 + 1e-6, -2 / 3) - 1);
    while (r > 12);
    const [dx, dy, dz] = randomDirection(random);
    let q = 0;
    for (;;) {
      q = random();
      if (0.1 * random() < q * q * Math.pow(1 - q * q, 3.5)) break;
    }
    const speed = q * Math.SQRT2 * Math.pow(1 + r * r, -0.25);
    const [ux, uy, uz] = randomDirection(random);
    c.pos.set([r * dx, r * dy, r * dz, 1 / n], i * 4);
    c.vel.set([speed * ux, speed * uy, speed * uz, 0], i * 4);
  }
  return centred(c);
}

interface DiskOptions {
  centre: [number, number];
  velocity: [number, number];
  /** +1 counter-clockwise, −1 clockwise. */
  spin: 1 | -1;
  /** Mass of the central black hole / bulge. */
  coreMass: number;
  /** Mass of all disk stars together. */
  diskMass: number;
  radius: number;
  group: 0 | 1;
}

/** Disk mass inside radius r for an exponential disk of scale length h truncated at R (fraction of the disk mass). */
function exponentialEnclosed(r: number, h: number, R: number): number {
  const m = (x: number) => 1 - (1 + x / h) * Math.exp(-x / h);
  return m(Math.min(r, R)) / m(R);
}

/**
 * Writes a rotating disk galaxy into `c` at indices [from, from + count): a heavy core plus stars on circular
 * orbits in an exponential disk (bright centre, fading edge), like real spiral galaxies.
 */
function writeDisk(c: Cloud, from: number, count: number, o: DiskOptions, random: () => number): void {
  const [cx, cy] = o.centre;
  const [vx, vy] = o.velocity;
  c.pos.set([cx, cy, 0, o.coreMass], from * 4);
  c.vel.set([vx, vy, 0, 0], from * 4);
  c.group[from] = 2;
  const stars = count - 1;
  const h = o.radius / 3.5;
  const inner = 0.12 * o.radius;
  for (let k = 0; k < stars; k++) {
    // Radius from the exponential surface-density profile (rejection on r·e^(−r/h)).
    let r: number;
    do r = inner + (o.radius - inner) * random();
    while (random() * (h / Math.E) > r * Math.exp(-r / h));
    const t = 2 * Math.PI * random();
    const enclosed = o.coreMass + o.diskMass * exponentialEnclosed(r, h, o.radius);
    const v = Math.sqrt(enclosed / Math.sqrt(r * r + 0.05 * 0.05));
    const i = from + 1 + k;
    c.pos.set([cx + r * Math.cos(t), cy + r * Math.sin(t), (random() - 0.5) * 0.02 * o.radius, o.diskMass / stars], i * 4);
    c.vel.set([vx - o.spin * v * Math.sin(t), vy + o.spin * v * Math.cos(t), 0, 0], i * 4);
    c.group[i] = o.group;
  }
}

/** One rotating disk galaxy (G = 1). */
export function diskGalaxy(n: number, seed = 1): Cloud {
  const c = emptyCloud(n);
  writeDisk(c, 0, n, { centre: [0, 0], velocity: [0, 0], spin: 1, coreMass: 1, diskMass: 0.3, radius: 3, group: 0 }, rng(seed));
  return centred(c);
}

/** Two disk galaxies on a close, nearly parabolic, prograde encounter (G = 1): the classic recipe for long tidal tails. */
export function collidingGalaxies(n: number, seed = 1): Cloud {
  const c = emptyCloud(n);
  const random = rng(seed);
  const half = Math.floor(n / 2);
  // Parabolic speed for two unit cores at separation 10 is √(2·G·2/10) ≈ 0.63; split between the two.
  writeDisk(c, 0, half, { centre: [-5, -1.5], velocity: [0.3, 0.06], spin: 1, coreMass: 1, diskMass: 0.2, radius: 2.5, group: 0 }, random);
  writeDisk(c, half, n - half, { centre: [5, 1.5], velocity: [-0.3, -0.06], spin: 1, coreMass: 1, diskMass: 0.2, radius: 2.5, group: 1 }, random);
  return centred(c);
}
