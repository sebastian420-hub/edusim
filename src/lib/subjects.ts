export type Subject = "physics" | "chemistry" | "cs" | "biology";
export type Difficulty = "easy" | "medium" | "hard";

export interface SimMeta {
  id: string;
  title: string;
  subject: Subject;
  difficulty: Difficulty;
  description: string;
  /** One short line for the home page plate. */
  tagline?: string;
  /** Alt text for the home page poster (public/plates/<id>.webp, made by `pnpm posters`). */
  posterAlt?: string;
  path: string;
  implemented: boolean;
}

export const SUBJECTS: Record<Subject, { label: string }> = {
  physics: { label: "Physics" },
  chemistry: { label: "Chemistry" },
  cs: { label: "Computer Science" },
  biology: { label: "Biology" },
};

export const SIMULATIONS: SimMeta[] = [
  // Physics
  { id: "wave-interference", title: "Wave Interference & Diffraction", subject: "physics", difficulty: "easy", description: "Explore how waves combine, cancel, and create patterns through slits.", tagline: "Slits, fringes and the λL/d rule — measured live.", posterAlt: "Bright and dark interference fringes fanning out from a double slit", path: "/physics/wave-interference", implemented: true },
  { id: "n-body", title: "N-Body Orbital Mechanics", subject: "physics", difficulty: "medium", description: "Simulate gravitational systems with thousands of particles.", tagline: "Orbits, Kepler’s laws and colliding galaxies.", posterAlt: "Two spiral galaxies of glowing stars colliding", path: "/physics/n-body", implemented: true },
  { id: "fluid-dynamics", title: "Fluid Dynamics", subject: "physics", difficulty: "hard", description: "Interactive Navier-Stokes fluid simulation.", path: "/physics/fluid-dynamics", implemented: false },
  { id: "quantum-wave", title: "Quantum Wave Function", subject: "physics", difficulty: "medium", description: "Visualize quantum tunneling and wave-particle duality.", path: "/physics/quantum-wave", implemented: false },
  { id: "double-pendulum", title: "Double Pendulum Chaos", subject: "physics", difficulty: "easy", description: "Explore deterministic chaos and sensitivity to initial conditions.", tagline: "The butterfly effect, measured — and a million-pendulum fractal.", posterAlt: "A fractal map of flip times in rainbow colours around a dark central region", path: "/physics/double-pendulum", implemented: true },
  // Chemistry
  { id: "molecular-orbitals", title: "Molecular Orbital Viewer", subject: "chemistry", difficulty: "medium", description: "3D volumetric rendering of atomic and molecular orbitals.", path: "/chemistry/molecular-orbitals", implemented: false },
  { id: "reaction-dynamics", title: "Chemical Reaction Dynamics", subject: "chemistry", difficulty: "hard", description: "Particle collision simulation with Arrhenius kinetics.", path: "/chemistry/reaction-dynamics", implemented: false },
  { id: "periodic-table-3d", title: "3D Periodic Table", subject: "chemistry", difficulty: "easy", description: "Interactive 3D periodic table with electron configurations.", path: "/chemistry/periodic-table-3d", implemented: false },
  { id: "titration", title: "Acid/Base Titration", subject: "chemistry", difficulty: "medium", description: "Real-time titration with microscopic ion visualization.", path: "/chemistry/titration", implemented: false },
  { id: "crystal-lattice", title: "Crystal Lattice Structures", subject: "chemistry", difficulty: "easy", description: "3D Bravais lattices with Miller plane cutting.", path: "/chemistry/crystal-lattice", implemented: false },
  // CS
  { id: "cellular-automata", title: "Cellular Automata", subject: "cs", difficulty: "easy", description: "Game of Life and custom rules on massive GPU grids.", tagline: "Life, death and gliders on a grid of millions.", posterAlt: "A dense field of living cells in Conway’s Game of Life", path: "/cs/cellular-automata", implemented: true },
  { id: "sorting-visualizer", title: "Sorting Algorithm Visualizer", subject: "cs", difficulty: "medium", description: "Compare sorting algorithms from bubble sort to GPU bitonic sort.", path: "/cs/sorting-visualizer", implemented: false },
  { id: "graph-traversal", title: "Graph Traversal", subject: "cs", difficulty: "hard", description: "BFS, DFS, Dijkstra, A* with force-directed layouts.", path: "/cs/graph-traversal", implemented: false },
  { id: "neural-net-viz", title: "Neural Network Decision Boundary", subject: "cs", difficulty: "hard", description: "Train a neural network on GPU and watch the decision boundary evolve.", path: "/cs/neural-net-viz", implemented: false },
  { id: "pathfinding", title: "Pathfinding Visualization", subject: "cs", difficulty: "medium", description: "A*, Dijkstra with heuristic potential field visualization.", path: "/cs/pathfinding", implemented: false },
  { id: "bst", title: "Binary Search Tree", subject: "cs", difficulty: "easy", description: "BST, AVL, and Red-Black tree operations with animations.", path: "/cs/bst", implemented: false },
  // Biology
  { id: "hodgkin-huxley", title: "Hodgkin-Huxley Neuron", subject: "biology", difficulty: "medium", description: "Simulate ion channel dynamics and action potential generation.", tagline: "Ion channels firing an action potential, step by step.", posterAlt: "Voltage and gate traces of a neuron firing repeatedly", path: "/biology/hodgkin-huxley", implemented: true },
  { id: "axon-propagation", title: "Axon Action Potential Propagation", subject: "biology", difficulty: "medium", description: "Watch action potentials propagate along myelinated and unmyelinated axons.", path: "/biology/axon-propagation", implemented: false },
  { id: "synaptic-transmission", title: "Synaptic Transmission", subject: "biology", difficulty: "hard", description: "Neurotransmitter release, diffusion, and receptor binding.", path: "/biology/synaptic-transmission", implemented: false },
  { id: "neural-net-learning", title: "Neural Network Learning", subject: "biology", difficulty: "hard", description: "Biological perspective on neural network training and plasticity.", path: "/biology/neural-net-learning", implemented: false },
  { id: "membrane-potential", title: "Resting Membrane Potential", subject: "biology", difficulty: "easy", description: "Nernst and Goldman equations with ion diffusion visualization.", path: "/biology/membrane-potential", implemented: false },
  { id: "brain-map", title: "Brain Region Map", subject: "biology", difficulty: "medium", description: "3D cortical surface with connectome dynamics.", path: "/biology/brain-map", implemented: false },
];

export function implementedSimulations(): SimMeta[] {
  return SIMULATIONS.filter((s) => s.implemented);
}

export function simulationsFor(subject: Subject): SimMeta[] {
  return SIMULATIONS.filter((s) => s.subject === subject);
}

export function findSimulation(subject: string, id: string): SimMeta | undefined {
  return SIMULATIONS.find((s) => s.subject === subject && s.id === id);
}

export const SUBJECT_ORDER = Object.keys(SUBJECTS) as Subject[];

/** Subject accent colour as a CSS value (defined once, in globals.css). */
export const accentOf = (subject: Subject) => `var(--acc-${subject})`;

/** Simulations not built yet, grouped by subject in display order (empty subjects omitted). */
export function plannedBySubject(): { subject: Subject; sims: SimMeta[] }[] {
  return SUBJECT_ORDER.map((subject) => ({ subject, sims: SIMULATIONS.filter((s) => s.subject === subject && !s.implemented) })).filter((g) => g.sims.length > 0);
}
