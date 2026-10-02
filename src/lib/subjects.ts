export type Subject = "physics" | "chemistry" | "cs" | "biology";
export type Difficulty = "easy" | "medium" | "hard";

export interface SimMeta {
  id: string;
  title: string;
  subject: Subject;
  difficulty: Difficulty;
  description: string;
  icon: string;
  path: string;
  implemented: boolean;
}

export const SUBJECTS: Record<Subject, { label: string; icon: string }> = {
  physics: { label: "Physics", icon: "⚛️" },
  chemistry: { label: "Chemistry", icon: "🧪" },
  cs: { label: "Computer Science", icon: "💻" },
  biology: { label: "Biology", icon: "🧬" },
};

export const SIMULATIONS: SimMeta[] = [
  // Physics
  { id: "wave-interference", title: "Wave Interference & Diffraction", subject: "physics", difficulty: "easy", description: "Explore how waves combine, cancel, and create patterns through slits.", icon: "🌊", path: "/physics/wave-interference", implemented: true },
  { id: "n-body", title: "N-Body Orbital Mechanics", subject: "physics", difficulty: "medium", description: "Simulate gravitational systems with thousands of particles.", icon: "🪐", path: "/physics/n-body", implemented: false },
  { id: "fluid-dynamics", title: "Fluid Dynamics", subject: "physics", difficulty: "hard", description: "Interactive Navier-Stokes fluid simulation.", icon: "🌊", path: "/physics/fluid-dynamics", implemented: false },
  { id: "quantum-wave", title: "Quantum Wave Function", subject: "physics", difficulty: "medium", description: "Visualize quantum tunneling and wave-particle duality.", icon: "🔮", path: "/physics/quantum-wave", implemented: false },
  { id: "double-pendulum", title: "Double Pendulum Chaos", subject: "physics", difficulty: "easy", description: "Explore deterministic chaos and sensitivity to initial conditions.", icon: "🎯", path: "/physics/double-pendulum", implemented: false },
  // Chemistry
  { id: "molecular-orbitals", title: "Molecular Orbital Viewer", subject: "chemistry", difficulty: "medium", description: "3D volumetric rendering of atomic and molecular orbitals.", icon: "🔬", path: "/chemistry/molecular-orbitals", implemented: false },
  { id: "reaction-dynamics", title: "Chemical Reaction Dynamics", subject: "chemistry", difficulty: "hard", description: "Particle collision simulation with Arrhenius kinetics.", icon: "💥", path: "/chemistry/reaction-dynamics", implemented: false },
  { id: "periodic-table-3d", title: "3D Periodic Table", subject: "chemistry", difficulty: "easy", description: "Interactive 3D periodic table with electron configurations.", icon: "🧮", path: "/chemistry/periodic-table-3d", implemented: false },
  { id: "titration", title: "Acid/Base Titration", subject: "chemistry", difficulty: "medium", description: "Real-time titration with microscopic ion visualization.", icon: "🧫", path: "/chemistry/titration", implemented: false },
  { id: "crystal-lattice", title: "Crystal Lattice Structures", subject: "chemistry", difficulty: "easy", description: "3D Bravais lattices with Miller plane cutting.", icon: "💎", path: "/chemistry/crystal-lattice", implemented: false },
  // CS
  { id: "cellular-automata", title: "Cellular Automata", subject: "cs", difficulty: "easy", description: "Game of Life and custom rules on massive GPU grids.", icon: "🔥", path: "/cs/cellular-automata", implemented: true },
  { id: "sorting-visualizer", title: "Sorting Algorithm Visualizer", subject: "cs", difficulty: "medium", description: "Compare sorting algorithms from bubble sort to GPU bitonic sort.", icon: "📊", path: "/cs/sorting-visualizer", implemented: false },
  { id: "graph-traversal", title: "Graph Traversal", subject: "cs", difficulty: "hard", description: "BFS, DFS, Dijkstra, A* with force-directed layouts.", icon: "🕸️", path: "/cs/graph-traversal", implemented: false },
  { id: "neural-net-viz", title: "Neural Network Decision Boundary", subject: "cs", difficulty: "hard", description: "Train a neural network on GPU and watch the decision boundary evolve.", icon: "🧠", path: "/cs/neural-net-viz", implemented: false },
  { id: "pathfinding", title: "Pathfinding Visualization", subject: "cs", difficulty: "medium", description: "A*, Dijkstra with heuristic potential field visualization.", icon: "🗺️", path: "/cs/pathfinding", implemented: false },
  { id: "bst", title: "Binary Search Tree", subject: "cs", difficulty: "easy", description: "BST, AVL, and Red-Black tree operations with animations.", icon: "🌳", path: "/cs/bst", implemented: false },
  // Biology
  { id: "hodgkin-huxley", title: "Hodgkin-Huxley Neuron", subject: "biology", difficulty: "medium", description: "Simulate ion channel dynamics and action potential generation.", icon: "⚡", path: "/biology/hodgkin-huxley", implemented: true },
  { id: "axon-propagation", title: "Axon Action Potential Propagation", subject: "biology", difficulty: "medium", description: "Watch action potentials propagate along myelinated and unmyelinated axons.", icon: "🔌", path: "/biology/axon-propagation", implemented: false },
  { id: "synaptic-transmission", title: "Synaptic Transmission", subject: "biology", difficulty: "hard", description: "Neurotransmitter release, diffusion, and receptor binding.", icon: "🔬", path: "/biology/synaptic-transmission", implemented: false },
  { id: "neural-net-learning", title: "Neural Network Learning", subject: "biology", difficulty: "hard", description: "Biological perspective on neural network training and plasticity.", icon: "🤖", path: "/biology/neural-net-learning", implemented: false },
  { id: "membrane-potential", title: "Resting Membrane Potential", subject: "biology", difficulty: "easy", description: "Nernst and Goldman equations with ion diffusion visualization.", icon: "⚡", path: "/biology/membrane-potential", implemented: false },
  { id: "brain-map", title: "Brain Region Map", subject: "biology", difficulty: "medium", description: "3D cortical surface with connectome dynamics.", icon: "🧠", path: "/biology/brain-map", implemented: false },
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
