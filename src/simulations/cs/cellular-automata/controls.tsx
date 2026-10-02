"use client";

import { useEffect, useRef, useState } from "react";
import { CellularAutomataSim } from "./sim";
import { patterns } from "./patterns";

export default function CellularAutomataControls() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<CellularAutomataSim | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [gridSize, setGridSize] = useState(256);
  const [speed, setSpeed] = useState(10);
  const [ruleB, setRuleB] = useState("3");
  const [ruleS, setRuleS] = useState("23");
  const [theme, setTheme] = useState(0);
  const [drawingMode, setDrawingMode] = useState(false);

  // Pan and Zoom state
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    
    // Set actual canvas resolution based on container size
    const resizeObserver = new ResizeObserver(entries => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        canvas.width = width * window.devicePixelRatio;
        canvas.height = height * window.devicePixelRatio;
        if (simRef.current) {
          // Force a re-render
          simRef.current.setPanZoom(pan.x, pan.y, zoom);
        }
      }
    });
    
    resizeObserver.observe(canvas.parentElement!);

    const sim = new CellularAutomataSim(canvas, gridSize);
    sim.onGenerationChange = setGeneration;
    sim.initialize();
    simRef.current = sim;

    return () => {
      sim.destroy();
      resizeObserver.disconnect();
    };
  }, []); // Run once on mount

  useEffect(() => {
    if (simRef.current) simRef.current.setRules(ruleB, ruleS);
  }, [ruleB, ruleS]);

  useEffect(() => {
    if (simRef.current) {
      simRef.current.setSpeed(speed);
    }
  }, [speed]);

  useEffect(() => {
    if (simRef.current) simRef.current.setTheme(theme);
  }, [theme]);
  
  useEffect(() => {
    if (simRef.current) simRef.current.setPanZoom(pan.x, pan.y, zoom);
  }, [pan, zoom]);

  const handlePlayPause = () => {
    if (!simRef.current) return;
    if (isPlaying) {
      simRef.current.pause();
    } else {
      simRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleStep = () => {
    if (simRef.current) {
      simRef.current.pause();
      setIsPlaying(false);
      simRef.current.step();
    }
  };

  const handleClear = () => {
    if (simRef.current) simRef.current.clear();
  };

  const handleRandomize = () => {
    if (simRef.current) simRef.current.randomize();
  };

  const handleLoadPattern = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (simRef.current && e.target.value) {
      simRef.current.loadPattern(e.target.value);
    }
  };

  const handleGridSizeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const size = parseInt(e.target.value);
    setGridSize(size);
    if (simRef.current) {
      simRef.current.setGridSize(size);
    }
  };

  // Canvas Interactions
  const handlePointerDown = (e: React.PointerEvent) => {
    if (drawingMode) {
      if (simRef.current) {
        const rect = canvasRef.current!.getBoundingClientRect();
        simRef.current.toggleCell(e.clientX - rect.left, e.clientY - rect.top);
      }
    } else {
      isDragging.current = true;
      lastMouse.current = { x: e.clientX, y: e.clientY };
      canvasRef.current?.setPointerCapture(e.pointerId);
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (isDragging.current && !drawingMode) {
      const dx = e.clientX - lastMouse.current.x;
      const dy = e.clientY - lastMouse.current.y;
      
      const rect = canvasRef.current!.getBoundingClientRect();
      const nx = pan.x + (dx / rect.width);
      const ny = pan.y + (dy / rect.height);

      setPan({ x: nx, y: ny });
      lastMouse.current = { x: e.clientX, y: e.clientY };
    }
  };

  const handlePointerUp = (e: React.PointerEvent) => {
    isDragging.current = false;
    canvasRef.current?.releasePointerCapture(e.pointerId);
  };

  const handleWheel = (e: React.WheelEvent) => {
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(z => Math.max(0.1, Math.min(z * zoomFactor, 20.0)));
  };

  return (
    <div className="flex flex-col md:flex-row h-screen bg-neutral-950 text-neutral-200 font-sans">
      {/* Controls Sidebar */}
      <div className="w-full md:w-80 p-6 bg-neutral-900 border-r border-neutral-800 flex flex-col gap-6 overflow-y-auto">
        <div>
          <h1 className="text-2xl font-bold text-white mb-1">Cellular Automata</h1>
          <p className="text-sm text-neutral-400">WebGPU Compute Shader</p>
        </div>

        <div className="flex items-center justify-between bg-neutral-800/50 p-3 rounded-lg border border-neutral-700/50">
          <span className="text-sm text-neutral-400">Generation</span>
          <span className="font-mono text-xl text-green-400">{generation}</span>
        </div>

        {/* Playback Controls */}
        <div className="flex flex-col gap-3">
          <div className="flex gap-2">
            <button
              onClick={handlePlayPause}
              className="flex-1 py-2 px-4 bg-green-600 hover:bg-green-500 text-white rounded font-medium transition-colors"
            >
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              onClick={handleStep}
              className="py-2 px-4 bg-neutral-700 hover:bg-neutral-600 rounded font-medium transition-colors"
            >
              Step
            </button>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleClear}
              className="flex-1 py-2 px-4 bg-red-900/50 hover:bg-red-800/50 text-red-200 rounded font-medium transition-colors"
            >
              Clear
            </button>
            <button
              onClick={handleRandomize}
              className="flex-1 py-2 px-4 bg-neutral-700 hover:bg-neutral-600 rounded font-medium transition-colors"
            >
              Random
            </button>
          </div>
        </div>

        <div className="h-px bg-neutral-800 w-full" />

        {/* Simulation Params */}
        <div className="flex flex-col gap-4">
          <label className="flex flex-col gap-1">
            <span className="text-sm text-neutral-400">Grid Size</span>
            <select
              value={gridSize}
              onChange={handleGridSizeChange}
              className="bg-neutral-800 border border-neutral-700 rounded p-2 text-sm"
            >
              <option value="256">256 x 256</option>
              <option value="512">512 x 512</option>
              <option value="1024">1024 x 1024</option>
              <option value="2048">2048 x 2048</option>
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm text-neutral-400">Speed (GPS: {speed})</span>
            <input
              type="range"
              min="1"
              max="60"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="accent-green-500"
            />
          </label>
        </div>

        <div className="h-px bg-neutral-800 w-full" />

        {/* Rules */}
        <div className="flex flex-col gap-3">
          <span className="text-sm font-medium text-neutral-300">Rules (B/S)</span>
          <div className="flex gap-4">
            <label className="flex flex-col gap-1 flex-1">
              <span className="text-xs text-neutral-400">Birth</span>
              <div className="flex items-center">
                <span className="text-neutral-500 mr-2">B</span>
                <input
                  type="text"
                  value={ruleB}
                  onChange={(e) => setRuleB(e.target.value)}
                  className="bg-neutral-800 border border-neutral-700 rounded p-1.5 w-full text-sm font-mono"
                />
              </div>
            </label>
            <label className="flex flex-col gap-1 flex-1">
              <span className="text-xs text-neutral-400">Survive</span>
              <div className="flex items-center">
                <span className="text-neutral-500 mr-2">S</span>
                <input
                  type="text"
                  value={ruleS}
                  onChange={(e) => setRuleS(e.target.value)}
                  className="bg-neutral-800 border border-neutral-700 rounded p-1.5 w-full text-sm font-mono"
                />
              </div>
            </label>
          </div>
        </div>

        <div className="h-px bg-neutral-800 w-full" />

        {/* Library & Tools */}
        <div className="flex flex-col gap-4">
          <label className="flex flex-col gap-1">
            <span className="text-sm text-neutral-400">Pattern Library</span>
            <select
              onChange={handleLoadPattern}
              className="bg-neutral-800 border border-neutral-700 rounded p-2 text-sm"
              defaultValue="Glider Gun (Gosper)"
            >
              <option value="" disabled>Select pattern...</option>
              {patterns.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </select>
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-sm text-neutral-400">Color Theme</span>
            <select
              value={theme}
              onChange={(e) => setTheme(Number(e.target.value))}
              className="bg-neutral-800 border border-neutral-700 rounded p-2 text-sm"
            >
              <option value="0">Classic Green</option>
              <option value="1">Cyberpunk Neon</option>
              <option value="2">Minimal White</option>
            </select>
          </label>

          <button
            onClick={() => setDrawingMode(!drawingMode)}
            className={`py-2 px-4 rounded font-medium transition-colors border ${
              drawingMode 
                ? 'bg-blue-600/20 border-blue-500 text-blue-300' 
                : 'bg-neutral-800 border-neutral-700 text-neutral-300 hover:bg-neutral-700'
            }`}
          >
            {drawingMode ? "Stop Drawing" : "Draw Mode"}
          </button>
        </div>
      </div>

      {/* Canvas Area */}
      <div className="flex-1 relative overflow-hidden bg-black touch-none">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full cursor-crosshair"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
          onWheel={handleWheel}
        />
        
        <div className="absolute top-4 right-4 bg-neutral-900/80 backdrop-blur px-3 py-1.5 rounded text-xs text-neutral-400 border border-neutral-800 flex gap-4 pointer-events-none">
          <span>Zoom: {zoom.toFixed(2)}x</span>
          <span>{drawingMode ? "Click to place cells" : "Drag to pan, Scroll to zoom"}</span>
        </div>
      </div>
    </div>
  );
}
