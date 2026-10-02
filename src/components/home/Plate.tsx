import Link from "next/link";
import { accentOf, SUBJECTS } from "@/lib/subjects";
import type { SimMeta } from "@/lib/subjects";
import { Difficulty } from "./Difficulty";
import { PlateArt } from "./PlateArt";
import { roman } from "./format";
import { POSTER } from "./poster";

/** One working simulation, presented like a numbered figure: art on top, caption below. */
export function Plate({ sim, number, index }: { sim: SimMeta; number: number; index: number }) {
  const subject = SUBJECTS[sim.subject];
  return (
    <Link
      href={sim.path}
      data-plate={sim.id}
      style={{ "--acc": accentOf(sim.subject), "--i": index + 2 } as React.CSSProperties}
      className="rise group block rounded-[3px] border border-hair bg-white/[0.015] transition-[border-color,background-color] duration-300 outline-none hover:border-(--acc) hover:bg-white/[0.035] focus-visible:border-(--acc) focus-visible:ring-2 focus-visible:ring-(--acc)"
    >
      <div
        className="relative aspect-[4/4.4] overflow-hidden border-b border-hair"
        style={{ background: "radial-gradient(120% 90% at 25% 20%, color-mix(in srgb, var(--acc) 22%, #0a0c10), #0a0c10 70%)" }}
        data-plate-art={sim.id}
      >
        <PlateArt id={sim.id}>
          {sim.posterAlt && (
            // Plain <img>: the poster is already optimised WebP (made by `pnpm posters`) with explicit size, so
            // next/image would only add ~16 KB of client JavaScript to the home page for no benefit.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={`/plates/${sim.id}.webp`}
              alt={sim.posterAlt}
              width={POSTER.width}
              height={POSTER.height}
              sizes="(min-width: 1024px) 240px, 70vw"
              fetchPriority={index === 0 ? "high" : "auto"}
              loading={index === 0 ? "eager" : "lazy"}
              decoding="async"
              className="h-full w-full object-cover transition-transform duration-[800ms] ease-out group-hover:scale-[1.04]"
            />
          )}
        </PlateArt>
        <span className="absolute left-2.5 top-2.5 flex items-center gap-1.5 rounded-[2px] bg-bg/70 px-[7px] py-[5px] font-mono text-[10px] leading-none font-medium tracking-[0.1em] text-ink uppercase backdrop-blur-sm">
          <b className="live-dot h-1.5 w-1.5 rounded-full bg-live" />
          live
        </span>
      </div>
      <div className="p-3.5 pb-3">
        <p className="font-mono text-[10px] leading-none tracking-[0.12em] text-(--acc) uppercase">
          Plate {roman(number)} · {subject.label}
        </p>
        <h2 className="mt-2.5 min-h-[2.4em] text-[16.5px] leading-tight font-medium tracking-[-0.015em]">{sim.title}</h2>
        <p className="mt-1.5 min-h-[3.6em] text-[12.5px] leading-[1.45] text-mut">{sim.tagline ?? sim.description}</p>
        <div className="mt-3 flex items-center justify-between border-t border-hair pt-2.5">
          <Difficulty level={sim.difficulty} />
          <span className="font-mono text-[11px] text-mut transition-colors group-hover:text-ink">
            Open <em className="inline-block not-italic transition-transform duration-300 group-hover:translate-x-[3px]">→</em>
          </span>
        </div>
      </div>
    </Link>
  );
}
