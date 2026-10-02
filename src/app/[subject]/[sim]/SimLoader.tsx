"use client";
import { notFound } from "next/navigation";
import { SIM_LOADERS } from "@/simulations/loaders";

export default function SimLoader({ simId }: { simId: string }) {
  const Sim = SIM_LOADERS[simId];
  if (!Sim) notFound();
  return <Sim />;
}
