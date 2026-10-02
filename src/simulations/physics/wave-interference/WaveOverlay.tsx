"use client";
import { useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { intensity, peakAmplitude, profileY, SOURCE_X, viewToWorld, worldToView } from "./wave";
import type { DetectorReadouts, WaveParams } from "./wave";

export type MeasureTool = "none" | "probe" | "ruler";

interface WaveOverlayProps {
  params: WaveParams;
  readouts: DetectorReadouts;
  tool: MeasureTool;
  onDetectorX: (x: number) => void;
}

type Point = [number, number];

const CURVE_WIDTH = 110; // px of graph for the highest intensity
const labelStyle = { paintOrder: "stroke", stroke: "rgba(0,0,0,0.85)", strokeWidth: 3 } as const;

export function WaveOverlay({ params, readouts, tool, onDetectorX }: WaveOverlayProps) {
  const ref = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [probe, setProbe] = useState<Point | null>(null);
  const [ruler, setRuler] = useState<{ a: Point; b: Point } | null>(null);
  const dragging = useRef<"detector" | "ruler" | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setSize({ w: entry.contentRect.width, h: entry.contentRect.height }));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const { w, h } = size;
  const ready = w > 0 && h > 0;

  const worldAt = (e: PointerEvent): Point => {
    const rect = ref.current!.getBoundingClientRect();
    return viewToWorld(e.clientX - rect.left, e.clientY - rect.top, w, h);
  };

  const onDetectorDown = (e: PointerEvent) => {
    dragging.current = "detector";
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  };

  const onToolDown = (e: PointerEvent) => {
    const point = worldAt(e);
    if (tool === "probe") setProbe(point);
    if (tool === "ruler") {
      dragging.current = "ruler";
      setRuler({ a: point, b: point });
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
    }
  };

  const onMove = (e: PointerEvent) => {
    if (dragging.current === "detector") {
      const [x] = worldAt(e);
      const maxX = (w / h) * 2 - 0.3;
      onDetectorX(Math.round(Math.min(Math.max(x, SOURCE_X + 0.3, -1), Math.min(maxX, 3)) * 20) / 20);
    } else if (dragging.current === "ruler") {
      const b = worldAt(e);
      setRuler((r) => (r ? { ...r, b } : r));
    }
  };

  const onUp = (e: PointerEvent) => {
    dragging.current = null;
    if ((e.currentTarget as Element).hasPointerCapture(e.pointerId)) (e.currentTarget as Element).releasePointerCapture(e.pointerId);
  };

  const toView = ([x, y]: Point): Point => worldToView(x, y, w, h);

  let curve = "";
  let detX = 0;
  if (ready && params.detector) {
    detX = worldToView(params.detectorX, 0, w, h)[0];
    const peak = Math.max(...readouts.profile, 1e-9);
    const scale = Math.min(CURVE_WIDTH, Math.max(w - detX - 12, 0));
    curve = readouts.profile
      .map((value, i) => {
        const y = worldToView(0, profileY(i, readouts.profile.length), w, h)[1];
        return `${i === 0 ? "M" : "L"}${(detX + (value / peak) * scale).toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }

  const probeView = probe && ready ? toView(probe) : null;
  const rulerView = ruler && ready ? { a: toView(ruler.a), b: toView(ruler.b) } : null;
  const rulerLength = ruler ? Math.hypot(ruler.b[0] - ruler.a[0], ruler.b[1] - ruler.a[1]) : 0;
  const drawsSpacing = params.mode === "double-slit" || params.mode === "two-points";

  return (
    <>
      <svg
        ref={ref}
        className="absolute inset-0 h-full w-full"
        style={{ pointerEvents: "none", touchAction: "none" }}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      >
        {ready && tool !== "none" && (
          <rect width={w} height={h} fill="transparent" style={{ pointerEvents: "all", cursor: "crosshair" }} onPointerDown={onToolDown} />
        )}

        {ready && params.detector && (
          <g>
            <line x1={detX} x2={detX} y1={0} y2={h} stroke="#fbbf24" strokeWidth={2} strokeDasharray="6 4" />
            <path d={`${curve} L${detX},${h} L${detX},0 Z`} fill="rgba(251,191,36,0.18)" stroke="none" />
            <path d={curve} fill="none" stroke="#fbbf24" strokeWidth={1.75} />
            <text x={detX + 8} y={16} fill="#fde68a" fontSize={11} style={labelStyle}>
              Detector — relative intensity
            </text>
            <rect
              x={detX - 9}
              y={0}
              width={18}
              height={h}
              fill="transparent"
              style={{ pointerEvents: "all", cursor: "ew-resize" }}
              onPointerDown={onDetectorDown}
              aria-hidden="true" // pointer-only handle; the "Detector position" slider is the accessible control
              data-testid="detector-handle"
            />
          </g>
        )}

        {probeView && probe && (
          <g>
            <circle cx={probeView[0]} cy={probeView[1]} r={6} fill="none" stroke="#38bdf8" strokeWidth={2} />
            <line x1={probeView[0] - 10} x2={probeView[0] + 10} y1={probeView[1]} y2={probeView[1]} stroke="#38bdf8" />
            <line x1={probeView[0]} x2={probeView[0]} y1={probeView[1] - 10} y2={probeView[1] + 10} stroke="#38bdf8" />
            <text x={probeView[0] + 12} y={probeView[1] - 8} fill="#e0f2fe" fontSize={12} style={labelStyle}>
              {`I = ${intensity(params, probe[0], probe[1]).toFixed(3)}  |ψ| = ${peakAmplitude(params, probe[0], probe[1]).toFixed(3)}`}
            </text>
          </g>
        )}

        {rulerView && (
          <g>
            <line x1={rulerView.a[0]} y1={rulerView.a[1]} x2={rulerView.b[0]} y2={rulerView.b[1]} stroke="#f472b6" strokeWidth={2} />
            <circle cx={rulerView.a[0]} cy={rulerView.a[1]} r={4} fill="#f472b6" />
            <circle cx={rulerView.b[0]} cy={rulerView.b[1]} r={4} fill="#f472b6" />
            <text x={(rulerView.a[0] + rulerView.b[0]) / 2 + 8} y={(rulerView.a[1] + rulerView.b[1]) / 2 - 8} fill="#fbcfe8" fontSize={12} style={labelStyle}>
              {`${rulerLength.toFixed(2)} units`}
            </text>
          </g>
        )}
      </svg>

      <div className="pointer-events-none absolute left-3 top-3 max-w-[16rem] space-y-0.5 rounded bg-black/60 px-2.5 py-1.5 font-mono text-[11px] leading-snug text-slate-200">
        <div>
          λ = {(params.waveSpeed / params.frequency).toFixed(3)} · screen height = 4 units
        </div>
        {params.detector && (
          <div>
            Detector x = {params.detectorX.toFixed(2)} (L = {(params.detectorX - SOURCE_X).toFixed(2)})
          </div>
        )}
        {params.detector && drawsSpacing && (
          <div className="text-amber-200">
            Fringe spacing: {readouts.measuredSpacing !== undefined ? readouts.measuredSpacing.toFixed(2) : "—"} measured ·{" "}
            {readouts.theorySpacing !== undefined ? readouts.theorySpacing.toFixed(2) : "—"} λL/d
          </div>
        )}
      </div>
    </>
  );
}
