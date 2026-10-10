"use client";
import { useId } from "react";

export interface ParameterSliderProps {
  label: string;
  value: number;
  onChange: (val: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  /** Digits shown after the decimal point; defaults to the precision implied by `step`. */
  decimals?: number;
  color?: "blue" | "green" | "purple" | "rose" | "red" | "amber" | "slate";
  className?: string;
  /** Logarithmic scale (for values spanning decades; min must be > 0). `step` is then ignored: 200 positions. */
  log?: boolean;
}

/** accent-* colours the native thumb; text-* is the thumb colour of the larger touch-screen thumb (globals.css). */
const ACCENTS: Record<NonNullable<ParameterSliderProps["color"]>, string> = {
  blue: "accent-blue-500 text-blue-500",
  green: "accent-green-500 text-green-500",
  purple: "accent-purple-500 text-purple-500",
  rose: "accent-rose-500 text-rose-500",
  red: "accent-red-500 text-red-500",
  amber: "accent-amber-500 text-amber-500",
  slate: "accent-slate-500 text-slate-500",
};

function decimalsOf(step: number): number {
  const s = String(step);
  return s.includes(".") ? s.split(".")[1].length : 0;
}

export function ParameterSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  unit = "",
  decimals,
  color = "blue",
  className = "",
  log = false,
}: ParameterSliderProps) {
  const id = useId();
  const digits = decimals ?? decimalsOf(step);
  // On a log scale the input runs over log10 of the value; what it reports back is rounded to `digits`.
  const toInput = (v: number) => (log ? Math.log10(v) : v);
  const fromInput = (x: number) => (log ? Number((10 ** x).toFixed(digits)) : x);

  return (
    <div className={`flex w-full flex-col gap-1 ${className}`}>
      <div className="flex items-center justify-between text-sm">
        <label className="font-medium text-slate-200" htmlFor={id}>
          {label}
        </label>
        <span className="font-mono text-xs text-slate-400">
          {value.toFixed(digits)}
          {unit}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={toInput(min)}
        max={toInput(max)}
        step={log ? (toInput(max) - toInput(min)) / 200 : step}
        value={toInput(value)}
        aria-valuetext={`${value.toFixed(digits)}${unit}`}
        onChange={(e) => onChange(fromInput(parseFloat(e.target.value)))}
        className={`slider h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-slate-700 ${ACCENTS[color]}`}
      />
      <div className="flex justify-between px-0.5 text-[10px] text-slate-400">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
