import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { NDArray, causalMask, causalMaskWithCache } from "../src/ndarray.js";
import { KVCache } from "../src/kvcache.js";
import { TransformerLM } from "../src/modules/transformerLM.js";
import { stateDictFromBinary, type WeightsManifest } from "../src/weights.js";

describe("causalMaskWithCache", () => {
  it("matches causalMask when totalLen === newLen (empty cache / prefill)", () => {
    for (const n of [1, 2, 5]) {
      expect(causalMaskWithCache(n, n).data).toEqual(causalMask(n).data);
    }
  });

  it("makes every column visible for single-token decode against a non-empty cache", () => {
    const mask = causalMaskWithCache(1, 5); // 4 cached positions + this new one
    expect(Array.from(mask.data)).toEqual([1, 1, 1, 1, 1]);
  });

  it("only masks within the new block for multi-token prefill against a non-empty cache", () => {
    // 2 cached positions, 3 new ones: rows are absolute positions 2,3,4 over 5 total columns.
    const mask = causalMaskWithCache(3, 5);
    expect(mask.toNestedArray()).toEqual([
      [1, 1, 1, 0, 0], // pos 2 sees 0..2
      [1, 1, 1, 1, 0], // pos 3 sees 0..3
      [1, 1, 1, 1, 1], // pos 4 sees 0..4
    ]);
  });
});

describe("KVCache", () => {
  it("starts empty and grows as layers are appended to", () => {
    const cache = new KVCache(2);
    expect(cache.length).toBe(0);
    cache.layers[0]!.k = NDArray.zeros([2, 3, 4]); // [numHeads, cachedLen, dK]
    cache.layers[0]!.v = NDArray.zeros([2, 3, 4]);
    expect(cache.length).toBe(3);
  });

  it("reset() clears every layer back to empty", () => {
    const cache = new KVCache(2);
    cache.layers[0]!.k = NDArray.zeros([2, 3, 4]);
    cache.reset();
    expect(cache.length).toBe(0);
    expect(cache.layers[0]!.k).toBeNull();
  });
});

describe("KV-cached incremental generation matches full-sequence recompute", () => {
  const manifest: WeightsManifest = JSON.parse(readFileSync("../weights/manifest.json", "utf-8"));
  const bytes = readFileSync("../weights/weights.bin");
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

  function loadModel(): TransformerLM {
    const model = new TransformerLM({
      vocabSize: manifest.config.vocabSize,
      contextLength: manifest.config.contextLength,
      dModel: manifest.config.dModel,
      numLayers: manifest.config.numLayers,
      numHeads: manifest.config.numHeads,
      dFF: manifest.config.dFF,
      ropeTheta: manifest.config.ropeTheta,
    });
    model.loadWeights(stateDictFromBinary(manifest, buffer));
    return model;
  }

  it("produces bit-identical logits at every position, step-by-step vs all at once", () => {
    const tokenIds = [1, 42, 7, 100, 3, 17];

    const fullModel = loadModel();
    const fullLogits = fullModel.forward(NDArray.fromNested(tokenIds));

    const cachedModel = loadModel();
    const cache = new KVCache(manifest.config.numLayers);
    const stepLogits: NDArray[] = [];
    for (const id of tokenIds) {
      stepLogits.push(cachedModel.forward(NDArray.fromNested([id]), cache));
    }

    for (let pos = 0; pos < tokenIds.length; pos++) {
      const fullRow = fullLogits.data.slice(pos * manifest.config.vocabSize, (pos + 1) * manifest.config.vocabSize);
      const stepRow = stepLogits[pos]!.data;
      expect(Array.from(stepRow)).toEqual(Array.from(fullRow));
    }
  });

  it("also matches when the whole prompt is prefilled in one call, then decoded one token at a time", () => {
    const promptIds = [1, 42, 7];
    const decodeIds = [100, 3, 17];
    const allIds = [...promptIds, ...decodeIds];

    const fullModel = loadModel();
    const fullLogits = fullModel.forward(NDArray.fromNested(allIds));

    const cachedModel = loadModel();
    const cache = new KVCache(manifest.config.numLayers);
    cachedModel.forward(NDArray.fromNested(promptIds), cache); // prefill
    const stepLogits: NDArray[] = [];
    for (const id of decodeIds) {
      stepLogits.push(cachedModel.forward(NDArray.fromNested([id]), cache));
    }

    for (let i = 0; i < decodeIds.length; i++) {
      const pos = promptIds.length + i;
      const fullRow = fullLogits.data.slice(pos * manifest.config.vocabSize, (pos + 1) * manifest.config.vocabSize);
      expect(Array.from(stepLogits[i]!.data)).toEqual(Array.from(fullRow));
    }
  });

  it("throws once cached + new positions would exceed contextLength", () => {
    const model = loadModel();
    const cache = new KVCache(manifest.config.numLayers);
    const nearLimit = new Array(manifest.config.contextLength - 1).fill(1);
    model.forward(NDArray.fromNested(nearLimit), cache);
    expect(cache.length).toBe(manifest.config.contextLength - 1);
    // One more token fits exactly...
    model.forward(NDArray.fromNested([1]), cache);
    expect(cache.length).toBe(manifest.config.contextLength);
    // ...but one beyond that must throw, not silently corrupt via out-of-range RoPE lookups.
    expect(() => model.forward(NDArray.fromNested([1]), cache)).toThrow(/contextLength/);
  });
});
