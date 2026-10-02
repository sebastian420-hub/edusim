/**
 * Typed, forgiving (de)serialisation of simulation settings to a URL query string.
 *
 * Only values that differ from the defaults are written, so links stay short; anything missing,
 * malformed or out of range falls back to the default (numbers are clamped), so a hand-edited or
 * outdated link can never put a simulation into an invalid state.
 */

export interface FieldCodec<T> {
  decode(raw: string): T | undefined;
  encode(value: T): string;
}

export type Schema<P> = { [K in keyof P]: FieldCodec<P[K]> };

export const numberField = (min: number, max: number): FieldCodec<number> => ({
  decode(raw) {
    const n = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(n)) return undefined;
    return Math.min(max, Math.max(min, n));
  },
  encode: (value) => String(Number(value.toFixed(4))),
});

export const enumField = <T extends string>(values: readonly T[]): FieldCodec<T> => ({
  decode: (raw) => ((values as readonly string[]).includes(raw) ? (raw as T) : undefined),
  encode: (value) => value,
});

export const boolField: FieldCodec<boolean> = {
  decode: (raw) => (raw === "1" ? true : raw === "0" ? false : undefined),
  encode: (value) => (value ? "1" : "0"),
};

const keysOf = <P extends object>(schema: Schema<P>) => Object.keys(schema) as (keyof P & string)[];

/** True when `search` sets at least one known field (so it should win over remembered settings). */
export function hasKnownParams<P extends object>(schema: Schema<P>, search: string): boolean {
  const query = new URLSearchParams(search);
  return keysOf(schema).some((key) => query.has(key));
}

export function decodeParams<P extends object>(schema: Schema<P>, defaults: P, search: string): P {
  const query = new URLSearchParams(search);
  const result = { ...defaults };
  for (const key of keysOf(schema)) {
    const raw = query.get(key);
    if (raw === null) continue;
    const value = schema[key].decode(raw);
    if (value !== undefined) result[key] = value as P[typeof key];
  }
  return result;
}

/** "" when everything equals the defaults, otherwise "?a=1&b=2" in schema order. */
export function encodeParams<P extends object>(schema: Schema<P>, defaults: P, params: P): string {
  const query = new URLSearchParams();
  for (const key of keysOf(schema)) {
    const codec = schema[key];
    const encoded = codec.encode(params[key]);
    if (encoded !== codec.encode(defaults[key])) query.set(key, encoded);
  }
  const text = query.toString();
  return text ? `?${text}` : "";
}
