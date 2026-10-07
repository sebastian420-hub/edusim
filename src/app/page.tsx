import { Footer } from "@/components/home/Footer";
import { Masthead } from "@/components/home/Masthead";
import { PlannedIndex } from "@/components/home/PlannedIndex";
import { Plate } from "@/components/home/Plate";
import { implementedSimulations, SIMULATIONS } from "@/lib/subjects";

export default function Home() {
  const live = implementedSimulations();
  const planned = SIMULATIONS.length - live.length;

  return (
    <div className="min-h-dvh bg-[radial-gradient(900px_420px_at_78%_-8%,rgb(94_234_212/0.07),transparent_70%),radial-gradient(700px_380px_at_8%_0,rgb(110_168_254/0.06),transparent_70%)]">
      <div className="mx-auto max-w-[1240px] px-5 sm:px-10">
        <Masthead live={live.length} planned={planned} />

        <main>
          <section className="grid gap-6 py-7 lg:grid-cols-[minmax(0,3.3fr)_minmax(0,8.7fr)] lg:gap-10 lg:pt-11 lg:pb-10">
            <div className="rise" style={{ "--i": 0 } as React.CSSProperties}>
              <p className="mb-5 font-mono text-label tracking-[0.14em] text-live uppercase">Interactive · GPU-computed</p>
              <h1 className="text-display font-medium text-balance">
                Science you can <span className="text-mut">reach into.</span>
              </h1>
              <p className="mt-5 max-w-[34ch] text-[15px] leading-normal text-mut">
                Live models of waves, orbits, chaos, neurons and cellular life, computed on your graphics card. Change a parameter. Watch the physics answer.
              </p>
              <dl className="mt-7 hidden border-t border-hair lg:block">
                {[
                  ["Engine", "WebGPU"],
                  ["Install", "none — runs in the browser"],
                  ["Works offline", "yes (static build)"],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-hair py-[9px] font-mono text-[11.5px] leading-none text-dim">
                    <dt>{k}</dt>
                    <dd className="text-ink">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* One row of plates that scrolls sideways with snap: a swipe on phones, four at a time on desktop. */}
            <ul
              aria-label="Simulations"
              className="-mx-5 flex scroll-pl-5 snap-x snap-mandatory gap-3 overflow-x-auto px-5 pb-1 [scrollbar-width:none] sm:-mx-10 sm:scroll-pl-10 sm:px-10 lg:mx-0 lg:scroll-pl-0 lg:gap-3.5 lg:px-0 lg:pb-3 lg:[scrollbar-color:rgb(148_163_184/0.35)_transparent] lg:[scrollbar-width:thin]"
            >
              {live.map((sim, i) => (
                <li key={sim.id} className="flex w-[min(70vw,300px)] shrink-0 snap-start *:flex-1 lg:w-[calc((100%-3*0.875rem)/4)]">
                  <Plate sim={sim} number={i + 1} index={i} />
                </li>
              ))}
            </ul>
          </section>

          <PlannedIndex />
        </main>

        <Footer />
      </div>
    </div>
  );
}
