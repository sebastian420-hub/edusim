import type { Difficulty as Level } from "@/lib/subjects";

const FILLED: Record<Level, number> = { easy: 1, medium: 2, hard: 3 };

/** Three dots, filled up to the difficulty. */
export function Difficulty({ level }: { level: Level }) {
  return (
    <span role="img" aria-label={`Difficulty: ${level}`} className="inline-flex gap-[3px]">
      {[0, 1, 2].map((i) => (
        <i key={i} className={`h-[5px] w-[5px] rounded-full border border-dim ${i < FILLED[level] ? "bg-dim" : ""}`} />
      ))}
    </span>
  );
}
