import { SimCard } from "@/components/SimCard";
import { SUBJECTS, simulationsFor } from "@/lib/subjects";
import type { Subject } from "@/lib/subjects";

const SUBJECT_KEYS = Object.keys(SUBJECTS) as Subject[];

export default function Home() {
  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-100 selection:bg-blue-500/30">
      <section className="relative mx-auto flex max-w-5xl flex-col items-center px-6 py-24 text-center md:py-32">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-blue-900/20 via-slate-950 to-slate-950" />
        <h1 className="mb-6 bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-5xl font-extrabold tracking-tight text-transparent md:text-7xl">
          EduSim
        </h1>
        <p className="mb-12 max-w-2xl text-xl font-light text-slate-400 md:text-2xl">
          GPU-Powered Interactive Science Simulations
        </p>

        <nav aria-label="Subjects" className="grid w-full grid-cols-2 gap-4 md:grid-cols-4">
          {SUBJECT_KEYS.map((key) => {
            const meta = SUBJECTS[key];
            const sims = simulationsFor(key);
            const available = sims.filter((s) => s.implemented).length;
            return (
              <a
                key={key}
                href={`#${key}`}
                className="flex flex-col items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/50 p-6 outline-none transition-colors hover:bg-slate-800/50 focus-visible:ring-2 focus-visible:ring-blue-500"
              >
                <span className="text-4xl" aria-hidden>
                  {meta.icon}
                </span>
                <h2 className="font-semibold">{meta.label}</h2>
                <span className="text-xs text-slate-500">
                  {available} of {sims.length} available
                </span>
              </a>
            );
          })}
        </nav>
      </section>

      <main className="mx-auto max-w-7xl space-y-16 px-6 pb-24">
        {SUBJECT_KEYS.map((key) => (
          <section key={key} id={key} className="scroll-mt-8">
            <h2 className="mb-8 flex items-center gap-3 text-2xl font-bold">
              <span className="h-1 w-8 rounded-full bg-blue-500" />
              {SUBJECTS[key].label}
            </h2>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
              {simulationsFor(key).map((sim) => (
                <SimCard key={sim.id} sim={sim} />
              ))}
            </div>
          </section>
        ))}
      </main>
    </div>
  );
}
