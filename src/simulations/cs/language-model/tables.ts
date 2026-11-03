/**
 * The count tables of a text, counted in a Web Worker (ngram.worker.ts), or on the main thread between frames where
 * workers are unavailable. Results arrive one table at a time, k = 0 first.
 */
import { MAX_K, tableResult } from "./ngram";
import type { TableResult } from "./ngram";
import type { Corpus } from "./text";

export interface TableRequest {
  id: number;
  corpus: Corpus;
  targets: Uint32Array;
  sampleLength: number;
}
export interface TableReply {
  id: number;
  result: TableResult;
}

let nextId = 1;

function startWorker(): Worker | null {
  try {
    return typeof Worker !== "undefined" ? new Worker(new URL("./ngram.worker.ts", import.meta.url), { type: "module" }) : null;
  } catch {
    return null;
  }
}

/**
 * Counts every table of `corpus`; `onTable` gets each result. Returns a function that cancels the rest (a worker
 * per request, so a cancelled request stops at once instead of holding up the next).
 */
export function countTables(corpus: Corpus, targets: Uint32Array, sampleLength: number, onTable: (r: TableResult) => void): () => void {
  const id = nextId++;
  let cancelled = false;
  const w = startWorker();
  if (w) {
    w.onmessage = (e: MessageEvent<TableReply>) => {
      if (e.data.id !== id || cancelled) return;
      onTable(e.data.result);
      if (e.data.result.k === MAX_K) w.terminate();
    };
    w.onerror = () => {
      w.terminate();
      if (!cancelled) fallback();
    };
    const request: TableRequest = { id, corpus, targets, sampleLength };
    w.postMessage(request);
  } else fallback();

  let timer: ReturnType<typeof setTimeout> | null = null;
  function fallback() {
    let k = 0;
    const next = () => {
      if (cancelled || k > MAX_K) return;
      onTable(tableResult(corpus, k, sampleLength, 1, targets));
      k++;
      timer = setTimeout(next, 0);
    };
    timer = setTimeout(next, 0);
  }
  return () => {
    cancelled = true;
    w?.terminate();
    if (timer) clearTimeout(timer);
  };
}
