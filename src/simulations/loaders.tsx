"use client";
import dynamic from "next/dynamic";
import type { ComponentType } from "react";

// Simulations touch WebGPU, so they are loaded client-side only. To add one: build it under
// src/simulations/<subject>/<id>/, mark it `implemented: true` in lib/subjects.ts, and register it here.
function loadSim(importer: () => Promise<{ default: ComponentType }>) {
  return dynamic(importer, {
    ssr: false,
    loading: () => (
      <div className="flex h-dvh items-center justify-center bg-slate-950 text-sm text-slate-300">
        Loading simulation…
      </div>
    ),
  });
}

export const SIM_LOADERS: Record<string, ComponentType> = {
  "wave-interference": loadSim(() => import("./physics/wave-interference/controls")),
  "double-pendulum": loadSim(() => import("./physics/double-pendulum/controls")),
  "n-body": loadSim(() => import("./physics/n-body/controls")),
  "cellular-automata": loadSim(() => import("./cs/cellular-automata/controls")),
  "axon-propagation": loadSim(() => import("./biology/axon-propagation/controls")),
  "hodgkin-huxley": loadSim(() => import("./biology/hodgkin-huxley/controls")),
};
