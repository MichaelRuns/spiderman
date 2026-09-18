import { useEffect, useRef, useState } from "react";
import type { StepSnapshot } from "../inference/snapshot.js";

export interface Playback {
  currentStep: StepSnapshot | null;
  currentIndex: number;
  totalSteps: number;
  isPlaying: boolean;
  speedMs: number;
  setSpeedMs: (ms: number) => void;
}

/**
 * Steps through `steps` at `speedMs` per step, independent of how fast the
 * real generation that produced them ran — the "time slider" that slows
 * inference down to a human-readable pace without slowing (or being
 * measured as part of) the actual compute. `speedMs<=0` means "live": jump
 * straight to the newest step as soon as it arrives.
 */
export function usePlayback(steps: StepSnapshot[], initialSpeedMs = 150): Playback {
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [speedMs, setSpeedMs] = useState(initialSpeedMs);
  const timerRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (steps.length === 0) {
      setCurrentIndex(-1);
      return;
    }
    if (speedMs <= 0) {
      setCurrentIndex(steps.length - 1);
      return;
    }
    setCurrentIndex((prev) => (prev < 0 ? 0 : prev));
  }, [steps.length, speedMs]);

  useEffect(() => {
    if (speedMs <= 0 || currentIndex < 0 || currentIndex >= steps.length - 1) return;
    timerRef.current = window.setTimeout(() => setCurrentIndex((i) => i + 1), speedMs);
    return () => window.clearTimeout(timerRef.current);
  }, [currentIndex, steps.length, speedMs]);

  return {
    currentStep: currentIndex >= 0 ? (steps[currentIndex] ?? null) : null,
    currentIndex,
    totalSteps: steps.length,
    isPlaying: speedMs > 0 && currentIndex >= 0 && currentIndex < steps.length - 1,
    speedMs,
    setSpeedMs,
  };
}
