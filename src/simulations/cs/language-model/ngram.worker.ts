/// <reference lib="webworker" />
// Counts the tables k = 0 … 6 off the main thread (a second or so each on Shakespeare) and posts each as it is done.
import { MAX_K, tableResult } from "./ngram";
import type { TableRequest, TableReply } from "./tables";

self.onmessage = (e: MessageEvent<TableRequest>) => {
  const { id, corpus, targets, sampleLength } = e.data;
  for (let k = 0; k <= MAX_K; k++) {
    const reply: TableReply = { id, result: tableResult(corpus, k, sampleLength, 1, targets) };
    self.postMessage(reply);
  }
};
