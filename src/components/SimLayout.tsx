"use client";
import Link from "next/link";
import { useState } from "react";
import type { ReactNode } from "react";
import { SUBJECTS } from "@/lib/subjects";
import type { Difficulty, Subject } from "@/lib/subjects";
import type { GpuStatus } from "@/lib/gpu/runtime";
import { GpuStatusOverlay } from "./GpuStatusOverlay";

interface SimLayoutProps {
  title: string;
  subject: Subject;
  difficulty: Difficulty;
  status: GpuStatus;
  /** The <canvas> (and any overlays drawn on top of it). */
  children: ReactNode;
  controls: ReactNode;
  explanation: ReactNode;
  /** Start in the playing state (default true). */
  initiallyPlaying?: boolean;
  onPlayPause?: (playing: boolean) => void;
  onReset?: () => void;
  /** When provided a Step button is shown; stepping pauses the simulation. */
  onStep?: () => void;
}

const DIFFICULTY_STYLES: Record<Difficulty, string> = {
  easy: "bg-green-500/20 text-green-400 border-green-500/20",
  medium: "bg-amber-500/20 text-amber-400 border-amber-500/20",
  hard: "bg-red-500/20 text-red-400 border-red-500/20",
};

export function SimLayout({
  title,
  subject,
  difficulty,
  status,
  children,
  controls,
  explanation,
  initiallyPlaying = true,
  onPlayPause,
  onReset,
  onStep,
}: SimLayoutProps) {
  const [playing, setPlaying] = useState(initiallyPlaying);
  const subjectMeta = SUBJECTS[subject];

  const togglePlaying = () => {
    const next = !playing;
    setPlaying(next);
    onPlayPause?.(next);
  };

  const step = () => {
    if (playing) {
      setPlaying(false);
      onPlayPause?.(false);
    }
    onStep?.();
  };

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-slate-950 text-slate-100">
      <header className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-800 bg-slate-900 px-4 py-3 md:px-6">
        <Link
          href="/"
          className="rounded text-sm text-slate-400 outline-none transition-colors hover:text-slate-100 focus-visible:ring-2 focus-visible:ring-blue-500"
        >
          ← All simulations
        </Link>
        <h1 className="text-lg font-bold tracking-tight md:text-xl">{title}</h1>
        <span className="flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs font-semibold text-slate-300">
          <span aria-hidden>{subjectMeta.icon}</span> {subjectMeta.label}
        </span>
        <span
          className={`rounded border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${DIFFICULTY_STYLES[difficulty]}`}
        >
          {difficulty}
        </span>
      </header>

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <main className="relative min-h-[45vh] flex-1 bg-black md:min-h-0">
          {children}
          <GpuStatusOverlay status={status} />
        </main>

        <aside className="flex max-h-[55vh] w-full flex-col overflow-y-auto border-t border-slate-800 bg-slate-900 md:max-h-none md:w-80 md:border-l md:border-t-0 lg:w-96">
          <section className="border-b border-slate-800 p-5">
            <h2 className="mb-3 text-sm font-bold uppercase tracking-wider text-slate-400">Explanation</h2>
            <div className="space-y-3 text-sm leading-relaxed text-slate-300">{explanation}</div>
          </section>

          <section className="flex-1 bg-slate-900/50 p-5">
            <h2 className="mb-4 text-sm font-bold uppercase tracking-wider text-slate-400">Parameters</h2>
            <div className="space-y-4">{controls}</div>

            <div className="mt-6 flex gap-3">
              <button
                type="button"
                onClick={togglePlaying}
                className="flex-1 rounded-lg bg-blue-600 px-4 py-2.5 font-semibold text-white transition-colors hover:bg-blue-500"
              >
                {playing ? "Pause" : "Play"}
              </button>
              {onStep && (
                <button
                  type="button"
                  onClick={step}
                  className="rounded-lg bg-slate-700 px-4 py-2.5 font-semibold text-white transition-colors hover:bg-slate-600"
                >
                  Step
                </button>
              )}
              <button
                type="button"
                onClick={onReset}
                className="rounded-lg bg-slate-700 px-4 py-2.5 font-semibold text-white transition-colors hover:bg-slate-600"
              >
                Reset
              </button>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
