"use client";
import dynamic from "next/dynamic";

const HHControls = dynamic(
  () => import("@/simulations/biology/hodgkin-huxley/controls"),
  { ssr: false, loading: () => <div className="p-8 text-white">Loading WebGPU Simulation...</div> }
);

export default function ClientBiology() {
  return <HHControls />;
}
