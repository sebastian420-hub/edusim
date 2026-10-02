import Link from "next/link";
import { SUBJECTS } from "@/lib/subjects";
import type { SimMeta } from "@/lib/subjects";

const DIFFICULTY_STYLES = {
  easy: "border-green-500/20 text-green-400",
  medium: "border-amber-500/20 text-amber-400",
  hard: "border-red-500/20 text-red-400",
} as const;

function CardBody({ sim }: { sim: SimMeta }) {
  const subject = SUBJECTS[sim.subject];
  return (
    <div
      className={`group flex h-full flex-col rounded-2xl border bg-slate-900 p-6 transition-all ${
        sim.implemented
          ? "border-slate-700 hover:border-slate-500 hover:shadow-lg hover:shadow-blue-900/20"
          : "border-slate-800 opacity-60"
      }`}
    >
      <div className="mb-4 flex items-start justify-between">
        <span className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-800 text-4xl shadow-inner" aria-hidden>
          {sim.icon}
        </span>
        <div className="flex flex-col items-end gap-2">
          <span
            className={`rounded-lg px-2 py-1 text-[10px] font-bold uppercase tracking-wider ${
              sim.implemented ? "border border-blue-500/20 bg-blue-500/10 text-blue-400" : "bg-slate-800 text-slate-400"
            }`}
          >
            {sim.implemented ? "Available" : "Coming Soon"}
          </span>
          <span className={`rounded-lg border px-2 py-1 text-[10px] font-bold uppercase tracking-wider ${DIFFICULTY_STYLES[sim.difficulty]}`}>
            {sim.difficulty}
          </span>
        </div>
      </div>
      <h3 className={`mb-2 text-xl font-bold text-slate-100 transition-colors ${sim.implemented ? "group-hover:text-blue-400" : ""}`}>
        {sim.title}
      </h3>
      <p className="mb-6 flex-1 text-sm text-slate-400">{sim.description}</p>
      <span className="mt-auto flex items-center gap-2 text-sm font-medium text-slate-500">
        <span className="text-lg" aria-hidden>
          {subject.icon}
        </span>
        {subject.label}
      </span>
    </div>
  );
}

export function SimCard({ sim }: { sim: SimMeta }) {
  if (!sim.implemented) {
    return (
      <div className="h-full cursor-not-allowed" aria-disabled="true">
        <CardBody sim={sim} />
      </div>
    );
  }
  return (
    <Link href={sim.path} className="block h-full rounded-2xl outline-none focus-visible:ring-2 focus-visible:ring-blue-500">
      <CardBody sim={sim} />
    </Link>
  );
}
