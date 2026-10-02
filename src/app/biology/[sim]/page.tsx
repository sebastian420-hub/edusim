import { notFound } from "next/navigation";
import ClientBiology from "./ClientBiology";

export default async function BiologySimPage({ params }: { params: Promise<{ sim: string }> }) {
  const resolvedParams = await params;
  if (resolvedParams.sim !== "hodgkin-huxley") {
    notFound();
  }

  return (
    <main className="w-screen h-screen">
      <ClientBiology />
    </main>
  );
}
