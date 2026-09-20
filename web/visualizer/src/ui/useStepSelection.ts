import { useEffect, useState } from "react";
import type { StepSnapshot } from "../inference/snapshot.js";

export interface StepSelection {
  currentStep: StepSnapshot | null;
  currentIndex: number;
  /** True while no token has been explicitly clicked — the view tracks the newest step as generation runs. */
  isFollowingLatest: boolean;
  select: (index: number) => void;
  followLatest: () => void;
}

/** Clicking a generated token pins the view to that step; "follow latest" (the default) tracks the newest step as it arrives. */
export function useStepSelection(steps: StepSnapshot[]): StepSelection {
  const [pinnedIndex, setPinnedIndex] = useState<number | null>(null);

  // A fresh generation run clears `steps` back to []; drop any stale pin from the previous run.
  useEffect(() => {
    if (steps.length === 0) setPinnedIndex(null);
  }, [steps.length === 0]);

  const currentIndex = pinnedIndex ?? steps.length - 1;
  const currentStep = currentIndex >= 0 ? (steps[currentIndex] ?? null) : null;

  return {
    currentStep,
    currentIndex,
    isFollowingLatest: pinnedIndex === null,
    select: (index: number) => setPinnedIndex(index),
    followLatest: () => setPinnedIndex(null),
  };
}
