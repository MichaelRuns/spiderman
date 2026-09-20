import { useLayoutEffect, useEffect, useRef, useState } from "react";
import type { StepSnapshot } from "../inference/snapshot.js";
import { type DiagramConfig, drawArchitecture, layoutRows } from "./diagram.js";

interface ArchitectureCanvasProps {
  config: DiagramConfig;
  step: StepSnapshot | null;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
}

const MAX_WIDTH = 640;
const MIN_WIDTH = 260;

function clampWidth(available: number): number {
  return Math.round(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, available)));
}

export function ArchitectureCanvas({ config, step, selectedLayer, onSelectLayer }: ArchitectureCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(MAX_WIDTH);
  const { totalHeight } = layoutRows(config);

  // Cap at MAX_WIDTH on desktop (the diagram's designed size); on a narrow
  // viewport, shrink to fit rather than forcing horizontal scroll. Belt and
  // suspenders: a synchronous measurement before first paint (so there's no
  // flash of the wrong size), a ResizeObserver for the general case (the
  // container can change width for reasons that have nothing to do with the
  // window, e.g. layout changes elsewhere on the page), and a plain window
  // `resize` listener as an independent fallback path.
  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    setWidth(clampWidth(container.getBoundingClientRect().width));
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const measure = () => setWidth(clampWidth(container.getBoundingClientRect().width));

    const observer = new ResizeObserver(measure);
    observer.observe(container);
    window.addEventListener("resize", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = totalHeight * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${totalHeight}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    drawArchitecture(ctx, { config, step, selectedLayer, width });
  }, [config, step, selectedLayer, totalHeight, width]);

  return (
    <div className="architecture" ref={containerRef}>
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
