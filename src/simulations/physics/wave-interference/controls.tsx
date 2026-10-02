"use client";
import { useEffect, useRef, useState } from "react";
import { createWaveRenderer, DEFAULTS, WaveUniforms } from "./sim";

// Mock ParameterSlider since we don't have its definition
const ParameterSlider = ({ label, min, max, step, value, onChange }: any) => (
  <div className="flex flex-col gap-1 mb-2">
    <label className="text-sm font-medium flex justify-between">
      {label} <span>{value}</span>
    </label>
    <input 
      type="range" min={min} max={max} step={step} value={value} 
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full"
    />
  </div>
);

export default function WaveInterferenceControls() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<any>(null);
  const [params, setParams] = useState<WaveUniforms>(DEFAULTS);

  useEffect(() => {
    if (!canvasRef.current) return;
    
    let isSubscribed = true;
    createWaveRenderer(canvasRef.current).then(renderer => {
      if (!isSubscribed) {
        renderer.dispose();
        return;
      }
      rendererRef.current = renderer;
    });
    
    return () => {
      isSubscribed = false;
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  useEffect(() => {
    if (rendererRef.current) {
      rendererRef.current.setUniforms(params);
    }
  }, [params]);

  const updateParam = (key: keyof WaveUniforms, value: any) => {
    setParams(prev => ({ ...prev, [key]: value }));
  };

  const setPreset = (preset: string) => {
    switch (preset) {
      case 'double-slit':
        setParams(prev => ({ ...prev, sourceCount: 2, source1: [0, prev.slitSeparation/2], source2: [0, -prev.slitSeparation/2] }));
        break;
      case 'single-point':
        setParams(prev => ({ ...prev, sourceCount: 1, source1: [-1.5, 0] }));
        break;
      case 'two-points':
        setParams(prev => ({ ...prev, sourceCount: 2, source1: [-1.0, 0.5], source2: [-1.0, -0.5] }));
        break;
    }
  };

  return (
    <div className="flex flex-col md:flex-row gap-6 p-4">
      <div className="flex-1 min-h-[500px] bg-black rounded-lg overflow-hidden relative">
        <canvas ref={canvasRef} className="w-full h-full block" />
      </div>
      <div className="w-full md:w-80 bg-gray-800 text-white p-4 rounded-lg flex flex-col gap-4 overflow-y-auto max-h-[800px]">
        <h2 className="text-xl font-bold">Wave Interference</h2>
        
        <div className="flex gap-2 text-sm flex-wrap">
          <button className="px-2 py-1 bg-blue-900 hover:bg-blue-800 text-white rounded" onClick={() => setPreset('double-slit')}>Double Slit</button>
          <button className="px-2 py-1 bg-blue-900 hover:bg-blue-800 text-white rounded" onClick={() => setPreset('single-point')}>Single Point</button>
          <button className="px-2 py-1 bg-blue-900 hover:bg-blue-800 text-white rounded" onClick={() => setPreset('two-points')}>Two Points</button>
        </div>

        <div className="flex flex-col gap-2 mt-2">
          <div className="text-sm font-semibold">View Mode</div>
          <div className="flex gap-4">
            <label className="flex items-center gap-1 text-sm"><input type="radio" name="viewMode" checked={params.viewMode === 0} onChange={() => updateParam('viewMode', 0)} /> Amplitude</label>
            <label className="flex items-center gap-1 text-sm"><input type="radio" name="viewMode" checked={params.viewMode === 1} onChange={() => updateParam('viewMode', 1)} /> Intensity</label>
            <label className="flex items-center gap-1 text-sm"><input type="radio" name="viewMode" checked={params.viewMode === 2} onChange={() => updateParam('viewMode', 2)} /> 3D Water</label>
          </div>
        </div>

        <ParameterSlider label="Frequency" min={0.5} max={10.0} step={0.1} value={params.frequency} onChange={(v: number) => updateParam('frequency', v)} />
        <ParameterSlider label="Amplitude" min={0.1} max={2.0} step={0.1} value={params.amplitude} onChange={(v: number) => updateParam('amplitude', v)} />
        <ParameterSlider label="Wave Speed" min={0.1} max={3.0} step={0.1} value={params.waveSpeed} onChange={(v: number) => updateParam('waveSpeed', v)} />
        <ParameterSlider label="Damping" min={0.0} max={0.1} step={0.01} value={params.damping} onChange={(v: number) => updateParam('damping', v)} />
        
        {params.sourceCount > 1 && (
          <>
            <ParameterSlider label="Slit Width" min={0.05} max={0.5} step={0.05} value={params.slitWidth} onChange={(v: number) => updateParam('slitWidth', v)} />
            <ParameterSlider label="Slit Separation" min={0.1} max={1.5} step={0.1} value={params.slitSeparation} onChange={(v: number) => {
              updateParam('slitSeparation', v);
              // Update source positions for double slit if they are at x=0
              if (params.source1[0] === 0) {
                updateParam('source1', [0, v/2]);
                updateParam('source2', [0, -v/2]);
              }
            }} />
          </>
        )}
      </div>
    </div>
  );
}
