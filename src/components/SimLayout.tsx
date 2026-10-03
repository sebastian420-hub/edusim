"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import type { ReactNode, RefObject } from "react";
import { accentOf, SUBJECTS } from "@/lib/subjects";
import type { Difficulty, Subject } from "@/lib/subjects";
import { captureCanvas } from "@/lib/gpu/runtime";
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
  /** When provided, the sidebar gets an Explanation | Challenges tab bar. */
  challenges?: ReactNode;
  /** Enables "Save image" for this canvas. */
  canvasRef?: RefObject<HTMLCanvasElement | null>;
  /** Render scale chosen by the adaptive-quality governor (1 = full); a badge explains when it is lower. */
  quality?: number;
  /** Enables the "Defaults" button. */
  onResetDefaults?: () => void;
  /** Start in the playing state (default true). */
  initiallyPlaying?: boolean;
  /**
   * Makes the play/pause state the page's to own (it is then kept in sync with `onPlayPause`). Useful when
   * something other than the buttons can pause the simulation, such as a challenge that sets up an experiment.
   */
  playing?: boolean;
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

const slug = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

const headerButton =
  "rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-200 outline-none transition-colors hover:bg-slate-700 focus-visible:ring-2 focus-visible:ring-blue-500";

/** Keys typed into form controls and buttons must not trigger shortcuts. */
function isInteractive(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && !!target.closest("input, textarea, select, button, a, [contenteditable=true]");
}

export function SimLayout({
  title,
  subject,
  difficulty,
  status,
  children,
  controls,
  explanation,
  challenges,
  canvasRef,
  quality = 1,
  onResetDefaults,
  initiallyPlaying = true,
  playing: controlledPlaying,
  onPlayPause,
  onReset,
  onStep,
}: SimLayoutProps) {
  const [ownPlaying, setOwnPlaying] = useState(initiallyPlaying);
  const playing = controlledPlaying ?? ownPlaying;
  const [tab, setTab] = useState<"explanation" | "challenges">("explanation");
  const [notice, setNotice] = useState<string | null>(null);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const noticeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const subjectMeta = SUBJECTS[subject];

  const flash = (message: string) => {
    setNotice(message);
    clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(null), 2000);
  };

  const setPlayingState = (next: boolean) => {
    setOwnPlaying(next);
    onPlayPause?.(next);
  };

  const step = () => {
    if (playing) setPlayingState(false);
    onStep?.();
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      flash("Link copied");
    } catch {
      window.prompt("Copy this link:", window.location.href); // clipboard needs a secure context
    }
  };

  const saveImage = async () => {
    const canvas = canvasRef?.current;
    if (!canvas) return;
    try {
      const blob = await captureCanvas(canvas);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${slug(title)}.png`;
      a.click();
      URL.revokeObjectURL(url);
      flash("Image saved");
    } catch {
      flash("Could not save image");
    }
  };

  // Latest handlers for the global key listener (kept in a ref so the listener is attached once).
  const keyActions = useRef({ toggle: () => {}, reset: () => {}, step: () => {}, help: () => {} });
  useEffect(() => {
    keyActions.current = {
      toggle: () => setPlayingState(!playing),
      reset: () => onReset?.(),
      step: () => onStep && step(),
      help: () => setShowShortcuts((v) => !v),
    };
  });

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || isInteractive(e.target)) return;
      const actions = keyActions.current;
      if (e.key === " ") actions.toggle();
      else if (e.key === "r" || e.key === "R") actions.reset();
      else if (e.key === ".") actions.step();
      else if (e.key === "?") actions.help();
      else if (e.key === "Escape") setShowShortcuts(false);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => () => clearTimeout(noticeTimer.current), []);

  const tabClass = (active: boolean) =>
    `flex-1 border-b-2 px-3 py-2 text-xs font-bold uppercase tracking-wider transition-colors ${
      active ? "border-blue-500 text-slate-100" : "border-transparent text-slate-500 hover:text-slate-300"
    }`;

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
          <span aria-hidden className="h-1.5 w-1.5 rounded-full" style={{ background: accentOf(subject) }} />
          {subjectMeta.label}
        </span>
        <span
          className={`rounded border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${DIFFICULTY_STYLES[difficulty]}`}
        >
          {difficulty}
        </span>

        <div className="relative ml-auto flex items-center gap-2">
          {notice && (
            <span role="status" className="text-xs text-green-400">
              {notice}
            </span>
          )}
          <button type="button" onClick={copyLink} className={headerButton}>
            Copy link
          </button>
          {canvasRef && (
            <button type="button" onClick={saveImage} className={headerButton}>
              Save image
            </button>
          )}
          {onResetDefaults && (
            <button type="button" onClick={onResetDefaults} className={headerButton}>
              Defaults
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowShortcuts((v) => !v)}
            aria-expanded={showShortcuts}
            aria-label="Keyboard shortcuts"
            className={headerButton}
          >
            ?
          </button>
          {showShortcuts && (
            <div role="dialog" aria-label="Keyboard shortcuts" className="absolute right-0 top-full z-20 mt-2 w-56 rounded-lg border border-slate-700 bg-slate-900 p-3 text-xs shadow-xl">
              <h2 className="mb-2 font-bold uppercase tracking-wider text-slate-400">Shortcuts</h2>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-slate-300">
                <dt className="font-mono text-slate-100">Space</dt>
                <dd>Play / pause</dd>
                {onStep && (
                  <>
                    <dt className="font-mono text-slate-100">.</dt>
                    <dd>Step</dd>
                  </>
                )}
                <dt className="font-mono text-slate-100">R</dt>
                <dd>Reset</dd>
                <dt className="font-mono text-slate-100">?</dt>
                <dd>Show / hide this help</dd>
              </dl>
            </div>
          )}
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <main className="relative min-h-[45vh] flex-1 bg-black md:min-h-0">
          {children}
          {quality < 1 && status.state === "ready" && (
            <div
              data-testid="quality-badge"
              title="EduSim lowered the drawing resolution to keep the animation smooth on this device"
              className="pointer-events-none absolute bottom-3 left-3 rounded bg-black/60 px-2 py-1 text-[11px] text-amber-200"
            >
              Resolution {Math.round(quality * 100)}% — reduced to stay smooth
            </div>
          )}
          <GpuStatusOverlay status={status} />
        </main>

        <aside className="flex max-h-[55vh] w-full flex-col overflow-y-auto border-t border-slate-800 bg-slate-900 md:max-h-none md:w-80 md:border-l md:border-t-0 lg:w-96">
          <section className="border-b border-slate-800">
            {challenges ? (
              <div role="tablist" className="flex border-b border-slate-800">
                <button role="tab" type="button" aria-selected={tab === "explanation"} onClick={() => setTab("explanation")} className={tabClass(tab === "explanation")}>
                  Explanation
                </button>
                <button role="tab" type="button" aria-selected={tab === "challenges"} onClick={() => setTab("challenges")} className={tabClass(tab === "challenges")}>
                  Challenges
                </button>
              </div>
            ) : (
              <h2 className="px-5 pt-5 text-sm font-bold uppercase tracking-wider text-slate-400">Explanation</h2>
            )}
            <div className="space-y-3 p-5 text-sm leading-relaxed text-slate-300">
              {challenges && tab === "challenges" ? challenges : explanation}
            </div>
          </section>

          <section className="flex-1 bg-slate-900/50 p-5">
            <h2 className="mb-4 text-sm font-bold uppercase tracking-wider text-slate-400">Parameters</h2>
            <div className="space-y-4">{controls}</div>

            <div className="mt-6 flex gap-3">
              <button
                type="button"
                onClick={() => setPlayingState(!playing)}
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
