"use client";
import React, { ReactNode, useState } from "react";
import { Subject, Difficulty, SUBJECTS } from "@/lib/subjects";

interface SimLayoutProps {
  title: string;
  subject: Subject;
  difficulty: Difficulty;
  children: ReactNode; // SimCanvas goes here
  controls: ReactNode; // Parameter sliders go here
  explanation: ReactNode; // Explanation text goes here
  onPlayPause?: (playing: boolean) => void;
  onReset?: () => void;
}

export function SimLayout({
  title,
  subject,
  difficulty,
  children,
  controls,
  explanation,
  onPlayPause,
  onReset
}: SimLayoutProps) {
  const [mode, setMode] = useState<"guided" | "interactive" | "sandbox">("interactive");
  const [playing, setPlaying] = useState(true);

  const subjectMeta = SUBJECTS[subject];

  const handlePlayPause = () => {
    const next = !playing;
    setPlaying(next);
    onPlayPause?.(next);
  };

  const difficultyColors = {
    easy: "bg-green-500/20 text-green-400 border-green-500/20",
    medium: "bg-amber-500/20 text-amber-400 border-amber-500/20",
    hard: "bg-red-500/20 text-red-400 border-red-500/20"
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between px-6 py-4 bg-slate-900 border-b border-slate-800 shrink-0 gap-4">
        <div className="flex items-center gap-4 flex-wrap">
          <h1 className="text-xl font-bold tracking-tight">{title}</h1>
          <span className={`px-2.5 py-1 text-xs font-semibold rounded-full bg-slate-800 text-slate-300 border border-slate-700 flex items-center gap-1.5`}>
            <span>{subjectMeta.icon}</span> {subjectMeta.label}
          </span>
          <span className={`px-2 py-0.5 text-[10px] uppercase tracking-wider font-bold rounded border ${difficultyColors[difficulty]}`}>
            {difficulty}
          </span>
        </div>
        <div className="flex bg-slate-800 p-1 rounded-lg self-start md:self-auto">
          {(["guided", "interactive", "sandbox"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-1.5 text-sm font-medium rounded-md capitalize transition-colors ${mode === m ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
            >
              {m}
            </button>
          ))}
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex flex-col md:flex-row flex-1 overflow-hidden">
        {/* Left: Simulation Canvas */}
        <main className="flex-1 relative bg-black min-h-[50vh] md:min-h-0">
          {children}
        </main>

        {/* Right: Sidebar */}
        <aside className="w-full md:w-80 lg:w-96 bg-slate-900 border-l border-slate-800 flex flex-col overflow-hidden">
          {/* Explanation Panel */}
          <div className="p-6 border-b border-slate-800 flex-1 overflow-y-auto min-h-[200px]">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Explanation</h3>
            <div className="prose prose-invert prose-sm">
              {explanation}
            </div>
          </div>

          {/* Controls Panel */}
          <div className="p-6 bg-slate-900/50 flex-none shrink-0">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Parameters</h3>
            <div className="space-y-4">
              {controls}
            </div>

            {/* Transport Controls */}
            <div className="mt-8 flex gap-3">
              <button 
                onClick={handlePlayPause}
                className="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2.5 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {playing ? (
                  <><span className="w-3 h-3 bg-white block" style={{ clipPath: "polygon(0 0, 33% 0, 33% 100%, 0 100%, 0 0, 67% 0, 100% 0, 100% 100%, 67% 100%, 67% 0)" }}></span> Pause</>
                ) : (
                  <><span className="w-3 h-3 bg-white block" style={{ clipPath: "polygon(0 0, 0 100%, 100% 50%)" }}></span> Play</>
                )}
              </button>
              <button 
                onClick={onReset}
                className="bg-slate-700 hover:bg-slate-600 text-white font-semibold py-2.5 px-4 rounded-lg transition-colors flex items-center justify-center"
                aria-label="Reset"
                title="Reset"
              >
                Reset
              </button>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
