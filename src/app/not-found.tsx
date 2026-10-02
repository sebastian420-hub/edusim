/* eslint-disable @next/next/no-html-link-for-pages -- see the comment on NotFound: next/link here costs ~9 KB on every page */
import { Logo } from "@/components/Logo";

// Plain <a> links on purpose: a full navigation is fine from a 404, and next/link would add its client code to this
// page's chunk, which Next loads on every route.
export default function NotFound() {
  return (
    <div className="mx-auto flex min-h-dvh w-full max-w-[1240px] flex-col px-5 sm:px-10">
      <header className="flex h-14 items-center border-b border-hair">
        <a href="/" className="flex items-center gap-2.5 text-[15px] font-semibold tracking-[-0.01em] outline-none focus-visible:ring-2 focus-visible:ring-live">
          <Logo />
          EduSim
        </a>
      </header>
      <main className="flex flex-1 flex-col justify-center pb-24">
        <p className="mb-5 font-mono text-label tracking-[0.14em] text-live uppercase">404 · Not found</p>
        <h1 className="text-display max-w-[16ch] font-medium text-balance">
          That simulation <span className="text-mut">doesn’t exist — yet.</span>
        </h1>
        <p className="mt-5 max-w-[38ch] text-[15px] leading-normal text-mut">
          The page you followed isn’t here. The simulations that are ready are on the front page; the rest are listed there as in preparation.
        </p>
        <a
          href="/"
          className="mt-8 inline-flex w-fit items-center gap-2 rounded-[3px] border border-hair-strong px-4 py-2.5 font-mono text-[12px] text-ink transition-colors outline-none hover:border-live focus-visible:ring-2 focus-visible:ring-live"
        >
          ← All simulations
        </a>
      </main>
    </div>
  );
}
