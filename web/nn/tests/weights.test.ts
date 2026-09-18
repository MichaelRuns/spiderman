import { describe, expect, it } from "vitest";
import { stateDictFromBinary, type WeightsManifest } from "../src/weights.js";

describe("stateDictFromBinary", () => {
  it("slices Float32Array views out of a shared ArrayBuffer at the manifest's offsets", () => {
    // Two tensors: a [2,2] identity-ish matrix, then a [3] vector.
    const a = Float32Array.from([1, 2, 3, 4]);
    const b = Float32Array.from([5, 6, 7]);
    const buffer = new ArrayBuffer(a.byteLength + b.byteLength);
    new Float32Array(buffer, 0, a.length).set(a);
    new Float32Array(buffer, a.byteLength, b.length).set(b);

    const manifest: WeightsManifest = {
      config: { vocabSize: 1, contextLength: 1, dModel: 1, numLayers: 1, numHeads: 1, dFF: 1, ropeTheta: 1 },
      specialTokens: [],
      tensors: {
        "layer.a": { shape: [2, 2], byteOffset: 0, byteLength: a.byteLength },
        "layer.b": { shape: [3], byteOffset: a.byteLength, byteLength: b.byteLength },
      },
    };

    const stateDict = stateDictFromBinary(manifest, buffer);

    expect(stateDict["layer.a"]!.shape).toEqual([2, 2]);
    expect(Array.from(stateDict["layer.a"]!.data)).toEqual([1, 2, 3, 4]);
    expect(stateDict["layer.b"]!.shape).toEqual([3]);
    expect(Array.from(stateDict["layer.b"]!.data)).toEqual([5, 6, 7]);
  });
});
