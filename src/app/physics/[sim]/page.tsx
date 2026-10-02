import { notFound } from "next/navigation";
import ClientPhysics from "./ClientPhysics";

export default async function PhysicsSimPage({ params }: { params: Promise<{ sim: string }> }) {
  const resolvedParams = await params;
  if (resolvedParams.sim !== "wave-interference") {
    notFound();
  }

  return (
    <main className="min-h-screen bg-zinc-950 text-white">
      <div className="container mx-auto py-8">
        <h1 className="text-3xl font-bold mb-6 px-4">Wave Interference Simulation</h1>
        <ClientPhysics />
      </div>
    </main>
  );
}
