import pkg from "../../../package.json";

export function Footer() {
  return (
    <footer className="flex flex-wrap justify-between gap-5 border-t border-hair pt-3.5 pb-5 font-mono text-label leading-snug tracking-normal text-dim">
      <p>Runs on WebGPU in current Chrome, Edge, Firefox and Safari.</p>
      <p>EduSim · v{pkg.version}</p>
    </footer>
  );
}
