import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { memo } from "react";
import type { StepSnapshot } from "../inference/snapshot.js";
import { LayerPipeline } from "./LayerPipeline.js";

interface SceneProps {
  step: StepSnapshot | null;
  numLayers: number;
  numHeads: number;
  dK: number;
  contextLength: number;
}

const SURFACE = "#1a1a19";

// The rest of the app re-renders far more often than `step` actually changes
// (e.g. every generated token during the fast real compute pass, while
// `step` — the current *playback* position — only advances on the slower,
// user-controlled timer). Memoized so this expensive 3D subtree only
// recomputes when its own props genuinely change.
export const Scene = memo(function Scene({ step, numLayers, numHeads, dK, contextLength }: SceneProps) {
  return (
    <div className="scene-container">
      <Canvas camera={{ position: [0, 1, 11], fov: 45 }}>
        <color attach="background" args={[SURFACE]} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[4, 8, 6]} intensity={0.9} />
        <LayerPipeline
          step={step}
          numLayers={numLayers}
          numHeads={numHeads}
          dK={dK}
          contextLength={contextLength}
        />
        <OrbitControls enablePan minDistance={4} maxDistance={24} />
      </Canvas>
    </div>
  );
});
