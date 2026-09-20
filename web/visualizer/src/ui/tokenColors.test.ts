import { describe, expect, it } from "vitest";
import { tokenChipColor } from "./tokenColors.js";

describe("tokenChipColor", () => {
  it("is deterministic — the same token id always gets the same color", () => {
    expect(tokenChipColor(42)).toBe(tokenChipColor(42));
  });

  it("gives different tokens different colors", () => {
    expect(tokenChipColor(0)).not.toBe(tokenChipColor(1));
    expect(tokenChipColor(1)).not.toBe(tokenChipColor(2));
  });

  it("stays well-distributed instead of clustering for consecutive ids", () => {
    const hues = Array.from({ length: 20 }, (_, id) => {
      const match = tokenChipColor(id).match(/hsla\(([\d.]+),/);
      return Number(match![1]);
    });
    // No two consecutive ids should land within a narrow hue band.
    for (let i = 1; i < hues.length; i++) {
      const diff = Math.abs(hues[i]! - hues[i - 1]!);
      expect(Math.min(diff, 360 - diff)).toBeGreaterThan(20);
    }
  });

  it("respects a custom alpha", () => {
    expect(tokenChipColor(7, 0.5)).toContain(", 0.5)");
  });
});
