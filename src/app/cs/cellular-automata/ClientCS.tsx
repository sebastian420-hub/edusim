"use client";
import dynamic from "next/dynamic";

const CellularAutomataControls = dynamic(
  () => import("@/simulations/cs/cellular-automata/controls"),
  { ssr: false, loading: () => <div className="p-8 text-white">Loading WebGPU Simulation...</div> }
);

export default function ClientCS() {
  return <CellularAutomataControls />;
}
