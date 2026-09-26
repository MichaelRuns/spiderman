import { describe, expect, it } from "vitest";
import { computeToyAttention, D_K, NUM_HEADS, SEQ, defaultTensor } from "./toyAttention.js";

describe("computeToyAttention", () => {
  const q = defaultTensor(1);
  const k = defaultTensor(2);
  const v = defaultTensor(3);

  it("produces the expected flat shapes", () => {
    const { scores, weights, output } = computeToyAttention(q, k, v, false);
    expect(scores).toHaveLength(NUM_HEADS * SEQ * SEQ);
    expect(weights).toHaveLength(NUM_HEADS * SEQ * SEQ);
    expect(output).toHaveLength(NUM_HEADS * SEQ * D_K);
  });

  it("every row of weights sums to 1 (softmax), with or without masking", () => {
    for (const causal of [false, true]) {
      const { weights } = computeToyAttention(q, k, v, causal);
      for (let h = 0; h < NUM_HEADS; h++) {
        for (let i = 0; i < SEQ; i++) {
          let rowSum = 0;
          for (let j = 0; j < SEQ; j++) rowSum += weights[h * SEQ * SEQ + i * SEQ + j]!;
          expect(rowSum).toBeCloseTo(1, 5);
        }
      }
    }
  });

  it("with causal masking, position 0 only attends to itself", () => {
    const { weights } = computeToyAttention(q, k, v, true);
    for (let h = 0; h < NUM_HEADS; h++) {
      expect(weights[h * SEQ * SEQ + 0 * SEQ + 0]).toBeCloseTo(1, 5);
      expect(weights[h * SEQ * SEQ + 0 * SEQ + 1]).toBeCloseTo(0, 5);
      expect(weights[h * SEQ * SEQ + 0 * SEQ + 2]).toBeCloseTo(0, 5);
    }
  });

  it("without masking, every position can attend to every other position (no forced zeros)", () => {
    const { weights } = computeToyAttention(q, k, v, false);
    // Not every weight is exactly zero anywhere in the unmasked case.
    expect(weights.some((w) => w === 0)).toBe(false);
  });
});
