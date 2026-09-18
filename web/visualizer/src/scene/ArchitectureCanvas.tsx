import { useEffect, useRef } from "react";
import type { StepSnapshot } from "../inference/snapshot.js";
import { type DiagramConfig, drawArchitecture, layoutRows } from "./diagram.js";

interface ArchitectureCanvasProps {
  config: DiagramConfig;
  step: StepSnapshot | null;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
}

const WIDTH = 640;

export function ArchitectureCanvas({ config, step, selectedLayer, onSelectLayer }: ArchitectureCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { totalHeight } = layoutRows(config);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = WIDTH * dpr;
    canvas.height = totalHeight * dpr;
    canvas.style.width = `${WIDTH}px`;
    canvas.style.height = `${totalHeight}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    drawArchitecture(ctx, { config, step, selectedLayer, width: WIDTH });
  }, [config, step, selectedLayer, totalHeight]);

  return (
    <div className="architecture">
      <div className="layer-tabs">
        {Array.from({ length: config.numLayers }, (_, i) => (
          <button
            key={i}
            type="button"
            className={i === selectedLayer ? "layer-tab active" : "layer-tab"}
            onClick={() => onSelectLayer(i)}
          >
            layer {i}
          </button>
        ))}
        <span className="layer-tabs-note">block repeats ×{config.numLayers} — pick one to inspect</span>
      </div>
      <canvas ref={canvasRef} />
    </div>
  );
}
