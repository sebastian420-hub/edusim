"use client";
import { useCallback, useEffect, useState } from "react";
import { decodeParams, encodeParams, hasKnownParams } from "./urlState";
import type { Schema } from "./urlState";

const storageKey = (simId: string) => `edusim:params:${simId}`;

function readStored(simId: string): string {
  try {
    return window.localStorage.getItem(storageKey(simId)) ?? "";
  } catch {
    return ""; // storage can be unavailable (private mode, blocked site data)
  }
}

/**
 * Simulation settings that live in the URL (shareable link) and, as a convenience, in localStorage.
 * Priority on load: URL > last used settings > defaults. Must run client-side only.
 */
export function usePersistedParams<P extends object>(simId: string, schema: Schema<P>, defaults: P) {
  const [params, setParams] = useState<P>(() => {
    const search = window.location.search;
    if (hasKnownParams(schema, search)) return decodeParams(schema, defaults, search);
    return decodeParams(schema, defaults, readStored(simId));
  });

  useEffect(() => {
    const search = encodeParams(schema, defaults, params);
    const { pathname, hash } = window.location;
    window.history.replaceState(window.history.state, "", `${pathname}${search}${hash}`);
    try {
      window.localStorage.setItem(storageKey(simId), search);
    } catch {
      // ignore: persistence is best-effort
    }
  }, [simId, schema, defaults, params]);

  const reset = useCallback(() => setParams(defaults), [defaults]);
  return [params, setParams, reset] as const;
}
