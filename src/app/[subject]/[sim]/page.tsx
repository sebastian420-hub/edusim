import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { findSimulation, implementedSimulations } from "@/lib/subjects";
import SimLoader from "./SimLoader";

// Only implemented simulations are generated; anything else is a 404.
export const dynamicParams = false;

export function generateStaticParams() {
  return implementedSimulations().map((s) => ({ subject: s.subject, sim: s.id }));
}

export async function generateMetadata({ params }: PageProps<"/[subject]/[sim]">): Promise<Metadata> {
  const { subject, sim } = await params;
  const meta = findSimulation(subject, sim);
  return meta ? { title: meta.title, description: meta.description } : {};
}

export default async function SimulationPage({ params }: PageProps<"/[subject]/[sim]">) {
  const { subject, sim } = await params;
  const meta = findSimulation(subject, sim);
  if (!meta?.implemented) notFound();
  return <SimLoader simId={meta.id} />;
}
