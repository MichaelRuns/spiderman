import { describe, expect, it } from "vitest";
import type { GenerationStep } from "../inference/generate.js";
import type { StepSnapshot } from "../inference/snapshot.js";
import { initialPerfStats, updatePerfStats } from "./perfStats.js";

const emptySnapshot = {} as StepSnapshot; // perfStats never reads .snapshot

function step(overrides: Partial<GenerationStep>): GenerationStep {
  return {
    step: 0,
    tokenId: 0,
    tokenText: "x",
    latencyMs: 10,
    isPrefill: false,
    cacheLength: 1,
    contextLength: 256,
    done: false,
    snapshot: emptySnapshot,
    ...overrides,
  };
}

describe("updatePerfStats", () => {
  it("records prefill latency separately, without counting it as a generated token", () => {
    const stats = updatePerfStats(initialPerfStats, step({ isPrefill: true, latencyMs: 40, cacheLength: 5 }), 128, 4);
    expect(stats.prefillMs).toBe(40);
    expect(stats.tokensGenerated).toBe(0);
    expect(stats.avgMs).toBeNull();
  });

  it("tracks min/max/avg/tokens-per-sec across decode steps", () => {
    let stats = initialPerfStats;
    stats = updatePerfStats(stats, step({ latencyMs: 10, cacheLength: 2 }), 128, 4);
    stats = updatePerfStats(stats, step({ latencyMs: 30, cacheLength: 3 }), 128, 4);
    stats = updatePerfStats(stats, step({ latencyMs: 5, cacheLength: 4 }), 128, 4);

    expect(stats.tokensGenerated).toBe(3);
    expect(stats.minMs).toBe(5);
    expect(stats.maxMs).toBe(30);
    expect(stats.avgMs).toBeCloseTo(15, 5);
    expect(stats.tokensPerSec).toBeCloseTo(1000 / 15, 5);
    expect(stats.currentMs).toBe(5);
  });

  it("computes KV cache byte size as numLayers * 2 (K+V) * cacheLength * dModel * 4 bytes", () => {
    const stats = updatePerfStats(initialPerfStats, step({ cacheLength: 10 }), 128, 4);
    expect(stats.kvCacheBytes).toBe(4 * 2 * 10 * 128 * 4);
  });
});
