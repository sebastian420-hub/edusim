"use client";
import dynamic from "next/dynamic";

const WaveInterferenceSim = dynamic(
  () => import("@/simulations/physics/wave-interference/controls"),
  { ssr: false, loading: () => <div className="p-8 text-center">Loading Simulation...</div> }
);

export default function ClientPhysics() {
  return <WaveInterferenceSim />;
}
