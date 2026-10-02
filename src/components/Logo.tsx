/** The EduSim mark: two interleaved sets of concentric rings — an interference pattern. */
export function Logo({ className = "h-[22px] w-[22px]" }: { className?: string }) {
  const radii = [3, 6, 9, 12];
  return (
    <svg viewBox="0 0 24 24" fill="none" strokeWidth="1" className={className} aria-hidden="true">
      <g stroke="#7db4ff" strokeOpacity="0.9">
        {radii.map((r) => (
          <circle key={r} cx="8" cy="12" r={r} />
        ))}
      </g>
      <g stroke="#5eead4" strokeOpacity="0.9">
        {radii.map((r) => (
          <circle key={r} cx="16" cy="12" r={r} />
        ))}
      </g>
    </svg>
  );
}
