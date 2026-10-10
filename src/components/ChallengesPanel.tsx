"use client";
import { useEffect, useState } from "react";
import { loadCompletedChallenges, saveCompletedChallenges } from "@/lib/challenges";
import type { Challenge } from "@/lib/challenges";

interface Session {
  id: string;
  answer: number | null; // index of the chosen prediction, once locked in
  started: boolean;
  showHint: boolean;
  revealed: boolean;
  /** The goal was reached at some point this session: it stays reached even if a live readout drifts back. */
  achieved: boolean;
}

interface ChallengesPanelProps<P, R> {
  simId: string;
  challenges: Challenge<P, R>[];
  params: P;
  readouts: R;
  /** Applies a challenge's setup to the simulation. */
  onSetup: (patch: Partial<P>) => void;
}

const button = "rounded-lg px-3 py-2 text-sm font-semibold transition-colors";

export function ChallengesPanel<P, R>({ simId, challenges, params, readouts, onSetup }: ChallengesPanelProps<P, R>) {
  const [completed, setCompleted] = useState<string[]>(() => loadCompletedChallenges(simId));
  const [session, setSession] = useState<Session | null>(null);
  const [choice, setChoice] = useState<number | null>(null);

  const challenge = challenges.find((c) => c.id === session?.id);
  const liveGoal = !!challenge && !!session?.started && challenge.goal.check({ params, readouts });

  // Completion is derived from the live goal check, so it is recorded while rendering (React's
  // documented pattern for adjusting state from props) and only the persistence is an effect.
  if (liveGoal && challenge && !completed.includes(challenge.id)) {
    setCompleted([...completed, challenge.id]);
  }
  // Once reached, the goal stays reached for this attempt (a population that dips again, a sweep that restarts).
  if (liveGoal && session && !session.achieved) {
    setSession({ ...session, achieved: true });
  }
  const goalMet = liveGoal || !!session?.achieved;

  useEffect(() => {
    saveCompletedChallenges(simId, completed);
  }, [simId, completed]);

  const open = (c: Challenge<P, R>) => {
    setChoice(null);
    setSession({ id: c.id, answer: null, started: false, showHint: false, revealed: false, achieved: false });
  };

  const start = (c: Challenge<P, R>, answer: number | null) => {
    onSetup(c.setup);
    setSession({ id: c.id, answer, started: true, showHint: false, revealed: false, achieved: false });
  };

  if (!challenge || !session) {
    return (
      <div className="space-y-3">
        <p className="text-slate-400">
          Short experiments: predict what will happen, try it, then see why. {completed.length} of {challenges.length} done.
        </p>
        {challenges.map((c, i) => (
          <button
            key={c.id}
            type="button"
            onClick={() => open(c)}
            className="flex w-full items-start gap-3 rounded-lg border border-slate-700 bg-slate-800/60 p-3 text-left transition-colors hover:border-slate-500"
          >
            <span
              aria-hidden
              className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                completed.includes(c.id) ? "bg-green-500 text-slate-950" : "bg-slate-700 text-slate-300"
              }`}
            >
              {completed.includes(c.id) ? "✓" : i + 1}
            </span>
            <span>
              <span className="block font-semibold text-slate-100">{c.title}</span>
              <span className="block text-xs text-slate-400">{completed.includes(c.id) ? "Completed" : c.prompt}</span>
            </span>
          </button>
        ))}
      </div>
    );
  }

  const prediction = challenge.prediction;
  const needsPrediction = !!prediction && !session.started;
  const correct = prediction && session.answer !== null ? !!prediction.options[session.answer]?.correct : null;
  const showExplanation = goalMet || session.revealed;

  return (
    <div className="space-y-4">
      <button type="button" onClick={() => setSession(null)} className="text-xs text-slate-400 hover:text-slate-200">
        ← All challenges
      </button>
      <div>
        <h3 className="text-base font-bold text-slate-100">{challenge.title}</h3>
        <p className="mt-1">{challenge.prompt}</p>
      </div>

      {needsPrediction && prediction && (
        <fieldset className="space-y-2">
          <legend className="mb-1 font-semibold text-slate-100">Predict first: {prediction.question}</legend>
          {prediction.options.map((o, i) => (
            <label key={o.label} className="flex items-center gap-2 text-slate-300">
              <input type="radio" name={`predict-${challenge.id}`} checked={choice === i} onChange={() => setChoice(i)} />
              {o.label}
            </label>
          ))}
          <button
            type="button"
            disabled={choice === null}
            onClick={() => start(challenge, choice)}
            className={`${button} bg-blue-600 text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40`}
          >
            Lock in &amp; set up experiment
          </button>
        </fieldset>
      )}

      {!prediction && !session.started && (
        <button type="button" onClick={() => start(challenge, null)} className={`${button} bg-blue-600 text-white hover:bg-blue-500`}>
          Set up experiment
        </button>
      )}

      {session.started && (
        <div className="space-y-3">
          {prediction && session.answer !== null && (
            <p className="text-slate-400">
              You predicted: <em className="text-slate-200">{prediction.options[session.answer].label}</em>
            </p>
          )}
          <div
            className={`flex items-start gap-3 rounded-lg border p-3 ${
              goalMet ? "border-green-500/50 bg-green-500/10" : "border-slate-700 bg-slate-800/60"
            }`}
            role="status"
          >
            <span aria-hidden className={goalMet ? "text-green-400" : "text-slate-500"}>
              {goalMet ? "✓" : "○"}
            </span>
            <span>
              <span className="block text-xs uppercase tracking-wider text-slate-400">Goal</span>
              <span className="text-slate-100">{challenge.goal.description}</span>
            </span>
          </div>

          {!showExplanation && (
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={() => setSession({ ...session, showHint: !session.showHint })} className={`${button} bg-slate-700 text-white hover:bg-slate-600`}>
                {session.showHint ? "Hide hint" : "Hint"}
              </button>
              <button type="button" onClick={() => setSession({ ...session, revealed: true })} className={`${button} bg-slate-800 text-slate-300 hover:bg-slate-700`}>
                Show explanation
              </button>
            </div>
          )}
          {session.showHint && !showExplanation && <p className="text-slate-400">💡 {challenge.hint}</p>}

          {showExplanation && (
            <div className="space-y-2 rounded-lg border border-blue-500/30 bg-blue-500/10 p-3">
              {correct !== null && (
                <p className="font-semibold text-slate-100">
                  {correct ? "Your prediction was right." : "Not quite — look at what the simulation shows."}
                </p>
              )}
              <p>{challenge.explanation}</p>
              <button
                type="button"
                onClick={() => {
                  setSession(null);
                  setChoice(null);
                }}
                className={`${button} bg-slate-700 text-white hover:bg-slate-600`}
              >
                Back to challenges
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
