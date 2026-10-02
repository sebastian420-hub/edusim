/** A guided experiment: predict, set up, try, then explain. Defined as data next to each simulation. */
export interface Challenge<P, R> {
  id: string;
  title: string;
  /** What the student is asked to explore. */
  prompt: string;
  /** Parameters applied when the experiment starts. */
  setup: Partial<P>;
  /** Optional prediction made *before* trying; exactly one option should be correct. */
  prediction?: { question: string; options: { label: string; correct?: boolean }[] };
  goal: {
    /** Shown to the student, e.g. "Double the slit separation". */
    description: string;
    /** Evaluated live against the simulation's parameters and measured readouts. */
    check: (state: { params: P; readouts: R }) => boolean;
  };
  hint: string;
  /** Shown once the goal is reached (or on request): why it behaves this way. */
  explanation: string;
}

const storageKey = (simId: string) => `edusim:challenges:${simId}`;

export function loadCompletedChallenges(simId: string): string[] {
  try {
    const raw = window.localStorage.getItem(storageKey(simId));
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === "string") : [];
  } catch {
    return [];
  }
}

export function saveCompletedChallenges(simId: string, ids: string[]): void {
  try {
    window.localStorage.setItem(storageKey(simId), JSON.stringify(ids));
  } catch {
    // best-effort persistence
  }
}
