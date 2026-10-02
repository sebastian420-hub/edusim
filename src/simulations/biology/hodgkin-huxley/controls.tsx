"use client";

import { useEffect, useRef, useState } from "react";
import { createSim, SimState } from "./sim";

export default function HHControls() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<Awaited<ReturnType<typeof createSim>> | null>(null);

  const [state, setState] = useState<SimState>({
    g_Na: 120.0,
    g_K: 36.0,
    g_L: 0.3,
    E_Na: 50.0,
    E_K: -77.0,
    E_L: -54.387,
    C_m: 1.0,
    I_inj: 10.0,
    temperature: 6.3,
    pulse_mode: 0,
  });

  useEffect(() => {
    if (!canvasRef.current) return;
    
    let sim: any = null;
    createSim(canvasRef.current).then((s) => {
      sim = s;
      simRef.current = sim;
    });
    
    return () => {
      if (sim) sim.destroy();
    };
  }, []);

  const updateState = (update: Partial<SimState>) => {
    const next = { ...state, ...update };
    setState(next);
    simRef.current?.setState(next);
  };

  const applyPreset = (preset: "normal" | "ttx" | "tea" | "anode") => {
    switch (preset) {
      case "normal":
        updateState({ g_Na: 120, g_K: 36, I_inj: 10, pulse_mode: 1 });
        break;
      case "ttx":
        updateState({ g_Na: 0, g_K: 36, I_inj: 20, pulse_mode: 1 });
        break;
      case "tea":
        updateState({ g_Na: 120, g_K: 0, I_inj: 10, pulse_mode: 1 });
        break;
      case "anode":
        updateState({ g_Na: 120, g_K: 36, I_inj: -20, pulse_mode: 1 });
        break;
    }
    simRef.current?.reset();
  };

  return (
    <div className="flex flex-col md:flex-row gap-6 p-4 w-full h-full bg-zinc-950 text-white">
      <div className="flex-1 flex flex-col min-h-[400px] border border-zinc-800 rounded-lg overflow-hidden relative">
        <canvas ref={canvasRef} className="w-full h-full" style={{ display: 'block', width: '100%', height: '100%' }} />
        <div className="absolute top-4 left-4 flex gap-4 text-xs font-mono">
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-[#33ff66] rounded-full"></div> Voltage (mV)</div>
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-[#ff3333] rounded-full"></div> m gate</div>
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-[#3366ff] rounded-full"></div> h gate</div>
          <div className="flex items-center gap-1"><div className="w-3 h-3 bg-[#ffcc33] rounded-full"></div> n gate</div>
        </div>
      </div>
      
      <div className="w-full md:w-80 flex flex-col gap-6 overflow-y-auto">
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Hodgkin-Huxley Neuron</h2>
          
          <div className="flex gap-2">
            <button onClick={() => simRef.current?.play()} className="bg-zinc-800 hover:bg-zinc-700 px-3 py-1 rounded">Play</button>
            <button onClick={() => simRef.current?.pause()} className="bg-zinc-800 hover:bg-zinc-700 px-3 py-1 rounded">Pause</button>
            <button onClick={() => simRef.current?.reset()} className="bg-zinc-800 hover:bg-zinc-700 px-3 py-1 rounded">Reset</button>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Injected Current (I_inj): {state.I_inj.toFixed(1)} µA/cm²</label>
            <input type="range" min="-20" max="50" step="0.5" value={state.I_inj} onChange={(e) => updateState({ I_inj: parseFloat(e.target.value) })} className="w-full" />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Temperature: {state.temperature.toFixed(1)} °C</label>
            <input type="range" min="0" max="40" step="0.1" value={state.temperature} onChange={(e) => updateState({ temperature: parseFloat(e.target.value) })} className="w-full" />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Na⁺ Conductance (TTX): {state.g_Na.toFixed(1)} mS/cm²</label>
            <input type="range" min="0" max="200" step="1" value={state.g_Na} onChange={(e) => updateState({ g_Na: parseFloat(e.target.value) })} className="w-full" />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">K⁺ Conductance (TEA): {state.g_K.toFixed(1)} mS/cm²</label>
            <input type="range" min="0" max="80" step="1" value={state.g_K} onChange={(e) => updateState({ g_K: parseFloat(e.target.value) })} className="w-full" />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Stimulus Mode</label>
            <div className="flex gap-4">
              <label><input type="radio" checked={state.pulse_mode === 0} onChange={() => updateState({ pulse_mode: 0 })} /> DC</label>
              <label><input type="radio" checked={state.pulse_mode === 1} onChange={() => updateState({ pulse_mode: 1 })} /> Pulse</label>
              <label><input type="radio" checked={state.pulse_mode === 2} onChange={() => updateState({ pulse_mode: 2 })} /> Twin</label>
            </div>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Presets</label>
            <div className="grid grid-cols-2 gap-2">
              <button onClick={() => applyPreset("normal")} className="bg-zinc-800 hover:bg-zinc-700 text-sm py-1 rounded">Normal AP</button>
              <button onClick={() => applyPreset("ttx")} className="bg-zinc-800 hover:bg-zinc-700 text-sm py-1 rounded">TTX Block</button>
              <button onClick={() => applyPreset("tea")} className="bg-zinc-800 hover:bg-zinc-700 text-sm py-1 rounded">TEA Block</button>
              <button onClick={() => applyPreset("anode")} className="bg-zinc-800 hover:bg-zinc-700 text-sm py-1 rounded">Anode Break</button>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}
