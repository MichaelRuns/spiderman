import { describe, expect, it } from "vitest";
import { NDArray, sumAxis } from "../src/ndarray.js";
import { Embedding } from "../src/modules/embedding.js";
import { Linear } from "../src/modules/linear.js";
import { MultiHeadSelfAttention } from "../src/modules/attention.js";
import { RMSNorm } from "../src/modules/rmsnorm.js";
import { RoPE } from "../src/modules/rope.js";
import { SwiGLU } from "../src/modules/swiglu.js";
import { TransformerBlock } from "../src/modules/transformerBlock.js";
import { TransformerLM } from "../src/modules/transformerLM.js";
import type { StateDict } from "../src/weights.js";

function identityWeight(n: number): { shape: number[]; data: Float32Array } {
  const data = new Float32Array(n * n);
  for (let i = 0; i < n; i++) data[i * n + i] = 1;
  return { shape: [n, n], data };
}

describe("Linear", () => {
  it("applies x @ W^T and records input/output state", () => {
    const linear = new Linear(2, 3);
    linear.loadWeights({
      weight: { shape: [3, 2], data: Float32Array.from([1, 0, 0, 1, 1, 1]) },
    });
    const x = NDArray.fromNested([[1, 2]]);
    const y = linear.forward(x);
    expect(y.toNestedArray()).toEqual([[1, 2, 3]]);
    expect(linear.getState().output?.shape).toEqual([1, 3]);
  });
});

describe("Embedding", () => {
  it("looks up rows by token id", () => {
    const embedding = new Embedding(4, 2);
    embedding.loadWeights({
      weight: { shape: [4, 2], data: Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3]) },
    });
    const ids = NDArray.fromNested([2, 0]);
    expect(embedding.forward(ids).toNestedArray()).toEqual([
      [2, 2],
      [0, 0],
    ]);
  });
});

describe("RMSNorm", () => {
  it("normalizes a constant vector to the unit weight scale", () => {
    const norm = new RMSNorm(4);
    norm.loadWeights({ weight: { shape: [4], data: Float32Array.from([2, 2, 2, 2]) } });
    const x = NDArray.fromNested([[3, 3, 3, 3]]);
    const y = norm.forward(x);
    // RMS of a constant vector equals the constant, so normalized values are all 1 * weight(2).
    for (const v of y.data) expect(v).toBeCloseTo(2, 5);
  });
});

describe("SwiGLU", () => {
  it("produces the expected output shape", () => {
    const ffn = new SwiGLU(4, 8);
    ffn.loadWeights({
      "w1.weight": { shape: [8, 4], data: new Float32Array(32).fill(0.1) },
      "w2.weight": { shape: [4, 8], data: new Float32Array(32).fill(0.1) },
      "w3.weight": { shape: [8, 4], data: new Float32Array(32).fill(0.1) },
    });
    const x = NDArray.fromNested([[1, 2, 3, 4]]);
    expect(ffn.forward(x).shape).toEqual([1, 4]);
  });
});

describe("RoPE", () => {
  it("preserves vector norm (rotation is norm-preserving)", () => {
    const rope = new RoPE(4, 8);
    const x = NDArray.fromNested([[1, 2, 3, 4]]).reshape([1, 4]);
    const positions = NDArray.fromNested([2]);
    const rotated = rope.forward(x, positions);
    const originalNorm = Math.sqrt([1, 2, 3, 4].reduce((s, v) => s + v * v, 0));
    const rotatedNorm = Math.sqrt(Array.from(rotated.data).reduce((s, v) => s + v * v, 0));
    expect(rotatedNorm).toBeCloseTo(originalNorm, 5);
  });

  it("is the identity at position 0", () => {
    const rope = new RoPE(4, 8);
    const x = NDArray.fromNested([[1, 2, 3, 4]]);
    const positions = NDArray.fromNested([0]);
    const rotated = rope.forward(x, positions);
    expect(Array.from(rotated.data)).toEqual(Array.from(x.data));
  });
});

describe("MultiHeadSelfAttention", () => {
  it("respects the causal mask: position 0's output only depends on V[0]", () => {
    const attn = new MultiHeadSelfAttention(4, 2);
    attn.loadWeights({
      "WQ.weight": identityWeight(4),
      "WK.weight": identityWeight(4),
      "WV.weight": identityWeight(4),
      "output_proj.weight": identityWeight(4),
    });
    const x = NDArray.fromNested([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
    ]);
    const output = attn.forward(x);
    // With identity Q/K/V/output projections, position 0 attends only to itself (causal mask),
    // so its output should equal V[0] = x[0] exactly.
    expect(Array.from(output.data.slice(0, 4))).toEqual(Array.from(x.data.slice(0, 4)));
  });

  it("records per-head attention weights that sum to 1 over keys", () => {
    const attn = new MultiHeadSelfAttention(4, 2);
    attn.loadWeights({
      "WQ.weight": identityWeight(4),
      "WK.weight": identityWeight(4),
      "WV.weight": identityWeight(4),
      "output_proj.weight": identityWeight(4),
    });
    const x = NDArray.fromNested([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
    ]);
    attn.forward(x);
    const weights = attn.getState().attnWeights!;
    expect(weights.shape).toEqual([2, 3, 3]); // [numHeads, seq, seq]
    const weightsArr = new NDArray(weights.shape, weights.data);
    const rowSums = sumAxis(weightsArr, -1);
    for (const v of rowSums.data) expect(v).toBeCloseTo(1, 5);
  });
});

describe("TransformerBlock", () => {
  it("produces output with the same shape as its input", () => {
    const block = new TransformerBlock(4, 2, 8);
    block.loadWeights({
      "ln1.weight": { shape: [4], data: new Float32Array(4).fill(1) },
      "ln2.weight": { shape: [4], data: new Float32Array(4).fill(1) },
      "attn.WQ.weight": identityWeight(4),
      "attn.WK.weight": identityWeight(4),
      "attn.WV.weight": identityWeight(4),
      "attn.output_proj.weight": identityWeight(4),
      "ffn.w1.weight": { shape: [8, 4], data: new Float32Array(32).fill(0.01) },
      "ffn.w2.weight": { shape: [4, 8], data: new Float32Array(32).fill(0.01) },
      "ffn.w3.weight": { shape: [8, 4], data: new Float32Array(32).fill(0.01) },
    });
    const x = NDArray.fromNested([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);
    expect(block.forward(x).shape).toEqual(x.shape);
  });
});

describe("TransformerLM", () => {
  it("produces logits of shape [..., seq, vocabSize] and exposes per-layer state", () => {
    const model = new TransformerLM({
      vocabSize: 10,
      contextLength: 8,
      dModel: 4,
      numLayers: 2,
      numHeads: 2,
      dFF: 8,
    });

    const stateDict: StateDict = {
      "token_embeddings.weight": { shape: [10, 4], data: new Float32Array(40).fill(0.05) },
      "ln_final.weight": { shape: [4], data: new Float32Array(4).fill(1) },
      "lm_head.weight": { shape: [10, 4], data: new Float32Array(40).fill(0.05) },
    };
    for (let i = 0; i < 2; i++) {
      stateDict[`layers.${i}.ln1.weight`] = { shape: [4], data: new Float32Array(4).fill(1) };
      stateDict[`layers.${i}.ln2.weight`] = { shape: [4], data: new Float32Array(4).fill(1) };
      stateDict[`layers.${i}.attn.WQ.weight`] = identityWeight(4);
      stateDict[`layers.${i}.attn.WK.weight`] = identityWeight(4);
      stateDict[`layers.${i}.attn.WV.weight`] = identityWeight(4);
      stateDict[`layers.${i}.attn.output_proj.weight`] = identityWeight(4);
      stateDict[`layers.${i}.ffn.w1.weight`] = { shape: [8, 4], data: new Float32Array(32).fill(0.01) };
      stateDict[`layers.${i}.ffn.w2.weight`] = { shape: [4, 8], data: new Float32Array(32).fill(0.01) };
      stateDict[`layers.${i}.ffn.w3.weight`] = { shape: [8, 4], data: new Float32Array(32).fill(0.01) };
    }
    model.loadWeights(stateDict);

    const tokenIds = NDArray.fromNested([[1, 2, 3]]);
    const logits = model.forward(tokenIds);
    expect(logits.shape).toEqual([1, 3, 10]);

    // Visualization state should be reachable at every level of the tree.
    expect(model.getState().logits?.shape).toEqual([1, 3, 10]);
    expect(model.layers[0]!.attn.getState().attnWeights).toBeDefined();
  });
});
