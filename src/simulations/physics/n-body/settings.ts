import { boolField, enumField, numberField } from "@/lib/urlState";
import type { FieldCodec, Schema } from "@/lib/urlState";
import type { Body, Integrator } from "./nbody";
import { DEFAULT_ORBIT_PRESET, ORBIT_PRESETS, orbitPreset } from "./presets";

export type NBodyMode = "orbit" | "galaxy";
export type GalaxyPreset = "cluster" | "disk" | "collision";
export const GALAXY_COUNTS = [1024, 4096, 16384] as const;
export type GalaxyCount = (typeof GALAXY_COUNTS)[number];

/** Most bodies the orbit lab holds (each gets a trail). */
export const MAX_ORBIT_BODIES = 8;

/** Everything a shared link reproduces. In the orbit lab that includes the exact starting bodies. */
export interface NBodySettings {
  mode: NBodyMode;
  /** Orbit-lab system the setup started from. */
  preset: string;
  /** Edited orbit-lab setup ("m,x,y,vx,vy;…"), or "" for the preset's own bodies. */
  bodies: string;
  /** Orbit lab: years per second of real time. */
  speed: number;
  integrator: Integrator;
  trails: boolean;
  showGraph: boolean;
  galaxy: GalaxyPreset;
  count: GalaxyCount;
  /** Galaxy mode: time units per second. */
  galaxySpeed: number;
}

export const NBODY_DEFAULTS: NBodySettings = {
  mode: "orbit",
  preset: DEFAULT_ORBIT_PRESET,
  bodies: "",
  speed: 0.5,
  integrator: "leapfrog",
  trails: true,
  showGraph: true,
  galaxy: "collision",
  count: 4096,
  galaxySpeed: 2,
};

/** Limits that keep any link physically sane (and the f32 GPU maths accurate). */
const LIMITS = { mass: [1e-15, 10], position: 100, speed: 300 } as const;

/** Six significant digits: enough for any hand-made setup, short in a URL. */
const num = (v: number) => String(Number(v.toPrecision(6)));

export function encodeBodies(bodies: readonly Body[]): string {
  return bodies.map((b) => [b.m, b.x, b.y, b.vx, b.vy].map(num).join(",")).join(";");
}

/** Parses an encoded setup; undefined when anything is malformed or out of range (the link is then ignored). */
export function decodeBodies(raw: string): Body[] | undefined {
  if (raw === "") return undefined;
  const parts = raw.split(";");
  if (parts.length < 1 || parts.length > MAX_ORBIT_BODIES) return undefined;
  const bodies: Body[] = [];
  for (const part of parts) {
    const values = part.split(",").map((s) => (s.trim() === "" ? NaN : Number(s)));
    if (values.length !== 5 || values.some((v) => !Number.isFinite(v))) return undefined;
    const [m, x, y, vx, vy] = values;
    if (m < LIMITS.mass[0] || m > LIMITS.mass[1]) return undefined;
    if (Math.abs(x) > LIMITS.position || Math.abs(y) > LIMITS.position) return undefined;
    if (Math.hypot(vx, vy) > LIMITS.speed) return undefined;
    bodies.push({ m, x, y, vx, vy });
  }
  return bodies;
}

const bodiesField: FieldCodec<string> = {
  decode(raw) {
    if (raw === "") return "";
    const bodies = decodeBodies(raw);
    return bodies ? encodeBodies(bodies) : undefined;
  },
  encode: (value) => value,
};

const countField: FieldCodec<GalaxyCount> = {
  decode: (raw) => GALAXY_COUNTS.find((c) => String(c) === raw),
  encode: String,
};

export const NBODY_SCHEMA: Schema<NBodySettings> = {
  mode: enumField(["orbit", "galaxy"] as const),
  preset: enumField(ORBIT_PRESETS.map((p) => p.id)),
  bodies: bodiesField,
  speed: numberField(0.02, 4),
  integrator: enumField(["leapfrog", "euler"] as const),
  trails: boolField,
  showGraph: boolField,
  galaxy: enumField(["cluster", "disk", "collision"] as const),
  count: countField,
  galaxySpeed: numberField(0.1, 4),
};

/** The orbit-lab bodies a settings object describes: the edited setup, or else the preset's. */
export function setupBodies(s: Pick<NBodySettings, "preset" | "bodies">): Body[] {
  return decodeBodies(s.bodies) ?? orbitPreset(s.preset).bodies.map((b) => ({ ...b }));
}

/** Display names: the preset's names by position, then generic ones for added bodies. */
export function bodyNames(preset: string, count: number): string[] {
  const names = orbitPreset(preset).names;
  return Array.from({ length: count }, (_, i) => names[i] ?? `Planet ${i + 1}`);
}
