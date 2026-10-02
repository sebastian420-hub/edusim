"use client";
import { useEffect, useRef, useState } from "react";

export interface SimRenderer {
  ready: Promise<void>;
  setUniforms: (uniforms: Record<string, unknown>) => void;
  dispose: () => void;
}

interface SimCanvasProps {
  createRenderer: (canvas: HTMLCanvasElement) => SimRenderer;
  uniforms?: Record<string, unknown>;
  className?: string;
}

export function SimCanvas({ createRenderer, uniforms, className }: SimCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<SimRenderer | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const renderer = createRenderer(canvas);
    rendererRef.current = renderer;

    renderer.ready
      .then(() => setStatus("ready"))
      .catch((e) => {
        setStatus("error");
        setError(String(e));
      });

    return () => {
      renderer.dispose();
      rendererRef.current = null;
    };
  }, [createRenderer]);

  useEffect(() => {
    if (uniforms && rendererRef.current) {
      rendererRef.current.setUniforms(uniforms);
    }
  }, [uniforms]);

  return (
    <div className={`relative ${className || ""}`}>
      <canvas ref={canvasRef} className="w-full h-full block" />
      {status === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <div className="text-white text-sm animate-pulse">Initializing GPU...</div>
        </div>
      )}
      {status === "error" && (
        <div className="absolute inset-0 flex items-center justify-center bg-red-900/80">
          <div className="text-white text-sm text-center px-4">
            <p className="font-bold">GPU Error</p>
            <p className="text-xs mt-1 opacity-75">{error || "WebGPU not supported"}</p>
          </div>
        </div>
      )}
    </div>
  );
}
