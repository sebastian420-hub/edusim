export interface ChartMarker {
  x: number;
  label?: string;
  color?: string;
}

interface MiniChartProps {
  title: string;
  xLabel: string;
  yLabel: string;
  x: number[];
  y: number[];
  /** Vertical reference lines (e.g. "you are here"). */
  markers?: ChartMarker[];
  /** A highlighted point on the curve. */
  point?: { x: number; y: number };
  yMax?: number;
  /** Text alternative for screen readers. */
  summary: string;
  className?: string;
}

const W = 280;
const H = 150;
const PAD = { left: 38, right: 10, top: 10, bottom: 30 };

/** Small dependency-free SVG line chart with axes, vertical markers and one highlighted point. */
export function MiniChart({ title, xLabel, yLabel, x, y, markers = [], point, yMax, summary, className = "" }: MiniChartProps) {
  const xMin = Math.min(...x);
  const xMax = Math.max(...x);
  const yTop = yMax ?? Math.max(1, ...y);
  const sx = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin || 1)) * (W - PAD.left - PAD.right);
  const sy = (v: number) => H - PAD.bottom - (v / yTop) * (H - PAD.top - PAD.bottom);
  const path = x.map((xv, i) => `${i === 0 ? "M" : "L"}${sx(xv).toFixed(1)},${sy(y[i]).toFixed(1)}`).join(" ");
  const xTicks = [xMin, (xMin + xMax) / 2, xMax];
  const yTicks = [0, yTop / 2, yTop];

  return (
    <figure className={className}>
      <figcaption className="mb-1 text-xs font-semibold text-slate-300">{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded bg-slate-950/60" role="img" aria-label={`${title}. ${summary}`}>
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line x1={PAD.left} x2={W - PAD.right} y1={sy(v)} y2={sy(v)} stroke="#334155" strokeWidth={0.5} />
            <text x={PAD.left - 4} y={sy(v) + 3} textAnchor="end" fontSize={9} fill="#94a3b8">
              {Math.round(v)}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <text key={`x${v}`} x={sx(v)} y={H - PAD.bottom + 11} textAnchor="middle" fontSize={9} fill="#94a3b8">
            {Math.round(v)}
          </text>
        ))}
        <text x={(PAD.left + W - PAD.right) / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#94a3b8">
          {xLabel}
        </text>
        <text x={10} y={(PAD.top + H - PAD.bottom) / 2} textAnchor="middle" fontSize={9} fill="#94a3b8" transform={`rotate(-90 10 ${(PAD.top + H - PAD.bottom) / 2})`}>
          {yLabel}
        </text>
        {markers.map((m) => (
          <g key={`${m.x}${m.label}`}>
            <line x1={sx(m.x)} x2={sx(m.x)} y1={PAD.top} y2={H - PAD.bottom} stroke={m.color ?? "#fbbf24"} strokeWidth={1} strokeDasharray="3 3" />
            {m.label && (
              <text x={sx(m.x) + 3} y={PAD.top + 8} fontSize={8.5} fill={m.color ?? "#fbbf24"}>
                {m.label}
              </text>
            )}
          </g>
        ))}
        <path d={path} fill="none" stroke="#34d399" strokeWidth={1.75} />
        {point && <circle cx={sx(point.x)} cy={sy(point.y)} r={3.5} fill="#f8fafc" stroke="#34d399" strokeWidth={1.5} />}
      </svg>
    </figure>
  );
}
