import { Logo } from "@/components/Logo";

export function Masthead({ live, planned }: { live: number; planned: number }) {
  return (
    <header className="flex h-14 items-center justify-between border-b border-hair">
      <div className="flex items-center gap-2.5 text-[15px] font-semibold tracking-[-0.01em]">
        <Logo />
        EduSim
      </div>
      <p className="font-mono text-label tracking-[0.04em] text-mut">
        <b className="font-medium text-live">{live}</b> live <span aria-hidden>&nbsp;·&nbsp;</span>
        <span className="sr-only">, </span>
        {planned} in preparation
      </p>
    </header>
  );
}
