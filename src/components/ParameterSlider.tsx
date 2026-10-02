"use client";
import React from "react";

export interface ParameterSliderProps {
  label: string;
  value: number;
  onChange: (val: number) => void;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  color?: "blue" | "green" | "purple" | "rose" | "red" | "amber" | "slate";
  className?: string;
}

export function ParameterSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  unit = "",
  color = "blue",
  className = ""
}: ParameterSliderProps) {
  const colorClasses: Record<string, string> = {
    blue: "accent-blue-500",
    green: "accent-green-500",
    purple: "accent-purple-500",
    rose: "accent-rose-500",
    red: "accent-red-500",
    amber: "accent-amber-500",
    slate: "accent-slate-500",
  };

  const trackColor = colorClasses[color] || colorClasses.blue;

  return (
    <div className={`flex flex-col gap-1 w-full ${className}`}>
      <div className="flex justify-between items-center text-sm">
        <label className="font-medium text-slate-200" htmlFor={`slider-${label}`}>
          {label}
        </label>
        <span className="text-slate-400 font-mono text-xs">
          {value}{unit}
        </span>
      </div>
      <input
        id={`slider-${label}`}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={`w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer ${trackColor}`}
        aria-label={`${label} parameter`}
      />
      <div className="flex justify-between text-[10px] text-slate-500 px-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
