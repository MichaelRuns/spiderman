import { describe, expect, it } from "vitest";
import { sampleToken } from "../src/sampling.js";

describe("sampleToken", () => {
  it("is greedy (argmax) when temperature <= 0", () => {
    const logits = Float32Array.from([1, 5, 2, -3, 4]);
    expect(sampleToken(logits, { temperature: 0 })).toBe(1);
    expect(sampleToken(logits, { temperature: -1 })).toBe(1);
  });

  it("topK=1 always picks the single highest-probability token, at any temperature", () => {
    const logits = Float32Array.from([1, 5, 2, -3, 4]);
    for (let i = 0; i < 20; i++) {
      expect(sampleToken(logits, { temperature: 1.5, topK: 1 })).toBe(1);
    }
  });

  it("never samples a token given zero probability by topP filtering", () => {
    const logits = Float32Array.from([10, -10, -10, -10]); // token 0 totally dominates
    for (let i = 0; i < 50; i++) {
      const sampled = sampleToken(logits, { temperature: 1, topP: 0.5 });
      expect(sampled).toBe(0);
    }
  });

  it("respects probability mass over many draws (statistical, tolerant)", () => {
    // Two tokens, roughly 3:1 odds under softmax(temperature=1).
    const logits = Float32Array.from([Math.log(3), Math.log(1)]);
    const counts = [0, 0];
    const trials = 4000;
    for (let i = 0; i < trials; i++) {
      counts[sampleToken(logits, { temperature: 1 })]!++;
    }
    const ratio = counts[0]! / trials;
    expect(ratio).toBeGreaterThan(0.65);
    expect(ratio).toBeLessThan(0.85);
  });

  it("always returns a valid index into the logits array", () => {
    const logits = Float32Array.from([0.1, 0.2, 0.3, 0.4]);
    for (let i = 0; i < 100; i++) {
      const sampled = sampleToken(logits, { temperature: 0.8, topK: 3 });
      expect(sampled).toBeGreaterThanOrEqual(0);
      expect(sampled).toBeLessThan(logits.length);
    }
  });
});
