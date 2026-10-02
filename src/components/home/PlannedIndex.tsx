import { plannedBySubject, SUBJECTS } from "@/lib/subjects";
import type { SimMeta, Subject } from "@/lib/subjects";
import { Difficulty } from "./Difficulty";

function Rows({ sims }: { sims: SimMeta[] }) {
  return (
    <ul>
      {sims.map((sim) => (
        <li key={sim.id} className="flex items-center justify-between gap-2.5 border-b border-hair py-[7px] text-[13px] leading-tight text-dim">
          <span>{sim.title}</span>
          <Difficulty level={sim.difficulty} />
        </li>
      ))}
    </ul>
  );
}

const labelClass = "font-mono text-label tracking-[0.1em] uppercase";

/**
 * The simulations still to come, as a quiet typographic index.
 * Desktop: four columns, always visible. Phone: four collapsed accordions. Two renderings of the same
 * data switched by CSS — no JavaScript, no layout shift, and the hidden one is out of the a11y tree.
 */
export function PlannedIndex() {
  const groups = plannedBySubject();
  const total = groups.reduce((n, g) => n + g.sims.length, 0);

  return (
    <section aria-labelledby="planned-heading" className="rise border-t border-hair-strong pt-5 pb-7" style={{ "--i": 5 } as React.CSSProperties}>
      <div className="mb-3.5 flex items-baseline justify-between">
        <h2 id="planned-heading" className={`${labelClass} font-medium text-ink`}>
          In preparation
        </h2>
        <p className="font-mono text-label tracking-normal text-dim max-lg:hidden">
          <Difficulty level="easy" /> easy &nbsp; <Difficulty level="medium" /> medium &nbsp; <Difficulty level="hard" /> hard
        </p>
        <p className="font-mono text-label tracking-normal text-dim lg:hidden">{total}</p>
      </div>

      <div className="grid grid-cols-4 gap-8 max-lg:hidden">
        {groups.map(({ subject, sims }: { subject: Subject; sims: SimMeta[] }) => (
          <div key={subject}>
            <h3 className={`${labelClass} flex justify-between border-b border-hair pb-2.5 text-mut`}>
              {SUBJECTS[subject].label}
              <span className="text-dim">{sims.length}</span>
            </h3>
            <Rows sims={sims} />
          </div>
        ))}
      </div>

      <div className="lg:hidden">
        {groups.map(({ subject, sims }) => (
          <details key={subject} className="group border-b border-hair">
            <summary className={`${labelClass} flex cursor-pointer list-none items-center justify-between py-3.5 text-mut outline-none focus-visible:ring-2 focus-visible:ring-live [&::-webkit-details-marker]:hidden`}>
              {SUBJECTS[subject].label}
              <span className="text-dim">
                {sims.length} <span aria-hidden className="ml-1 inline-block transition-transform group-open:rotate-90">›</span>
              </span>
            </summary>
            <Rows sims={sims} />
          </details>
        ))}
      </div>
    </section>
  );
}
