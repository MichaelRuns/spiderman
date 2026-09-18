import { describe, expect, it } from "vitest";
import { topKPredictions } from "./predictions.js";

describe("topKPredictions", () => {
  it("ranks tokens by softmax probability, descending", () => {
    const logits = Float32Array.from([1, 5, 2, -3, 4]);
    const top3 = topKPredictions(logits, 3);
    expect(top3.map((p) => p.tokenId)).toEqual([1, 4, 2]);
    expect(top3[0]!.probability).toBeGreaterThan(top3[1]!.probability);
    expect(top3[1]!.probability).toBeGreaterThan(top3[2]!.probability);
  });

  it("probabilities sum to 1 across the full vocab (k = vocab size)", () => {
    const logits = Float32Array.from([1, 2, 3, 4]);
    const all = topKPredictions(logits, 4);
    const sum = all.reduce((s, p) => s + p.probability, 0);
    expect(sum).toBeCloseTo(1, 5);
  });

  it("clamps k to the vocab size", () => {
    const logits = Float32Array.from([1, 2]);
    expect(topKPredictions(logits, 10)).toHaveLength(2);
  });
});
