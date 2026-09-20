import type { GenerationStep } from "../inference/generate.js";

export interface PerfStats {
  tokensGenerated: number;
  prefillMs: number | null;
  currentMs: number | null;
  minMs: number | null;
  maxMs: number | null;
  avgMs: number | null;
  tokensPerSec: number | null;
  totalDecodeMs: number;
  kvCacheBytes: number;
}

export const initialPerfStats: PerfStats = {
  tokensGenerated: 0,
  prefillMs: null,
  currentMs: null,
  minMs: null,
  maxMs: null,
  avgMs: null,
  tokensPerSec: null,
  totalDecodeMs: 0,
  kvCacheBytes: 0,
};

/** `dModel` and `numLayers` come from the loaded model's manifest, to size the KV cache footprint. */
export function updatePerfStats(prev: PerfStats, step: GenerationStep, dModel: number, numLayers: number): PerfStats {
  const kvCacheBytes = numLayers * 2 * step.cacheLength * dModel * 4; // K + V, float32

  if (step.isPrefill) {
    return { ...prev, prefillMs: step.latencyMs, kvCacheBytes };
  }

  const tokensGenerated = prev.tokensGenerated + 1;
  const totalDecodeMs = prev.totalDecodeMs + step.latencyMs;
  const minMs = prev.minMs === null ? step.latencyMs : Math.min(prev.minMs, step.latencyMs);
  const maxMs = prev.maxMs === null ? step.latencyMs : Math.max(prev.maxMs, step.latencyMs);
  const avgMs = totalDecodeMs / tokensGenerated;

  return {
    ...prev,
    tokensGenerated,
    currentMs: step.latencyMs,
    minMs,
    maxMs,
    avgMs,
    tokensPerSec: 1000 / avgMs,
    totalDecodeMs,
    kvCacheBytes,
  };
}
