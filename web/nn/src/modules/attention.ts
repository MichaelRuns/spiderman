import { NDArray, arangeArray, causalMaskWithCache, concatAxis, mergeHeads, splitHeads } from "../ndarray.js";
import { scaledDotProductAttention } from "../functions.js";
import { type StateDict } from "../weights.js";
import type { LayerKVCache } from "../kvcache.js";
import { Linear } from "./linear.js";
import { LayerModule } from "./module.js";
import type { RoPE } from "./rope.js";

export class MultiHeadSelfAttention extends LayerModule {
  numHeads: number;
  dK: number;
  WQ: Linear;
  WK: Linear;
  WV: Linear;
  output_proj: Linear;
  rope?: RoPE;

  constructor(dModel: number, numHeads: number, rope?: RoPE) {
    super();
    this.numHeads = numHeads;
    this.dK = dModel / numHeads;
    this.WQ = new Linear(dModel, dModel);
    this.WK = new Linear(dModel, dModel);
    this.WV = new Linear(dModel, dModel);
    this.output_proj = new Linear(dModel, dModel);
    this.rope = rope;
  }

  loadWeights(stateDict: StateDict, prefix = ""): void {
    this.WQ.loadWeights(stateDict, `${prefix}WQ.`);
    this.WK.loadWeights(stateDict, `${prefix}WK.`);
    this.WV.loadWeights(stateDict, `${prefix}WV.`);
    this.output_proj.loadWeights(stateDict, `${prefix}output_proj.`);
  }

  /**
   * `x`: [..., seq, dModel] — the new tokens only (the whole prompt on a
   * prefill call, or a single token on an incremental decode step).
   * `tokenPositions` defaults to 0..seq-1; pass explicit absolute positions
   * when decoding with a non-empty `cache`. When `cache` is given, this
   * layer's K/V for `x` are appended into it and attention runs over the
   * full cached history — with `cache` omitted, behavior is identical to
   * the plain full-sequence forward pass.
   */
  forward(x: NDArray, tokenPositions?: NDArray, cache?: LayerKVCache): NDArray {
    const newLen = x.shape[x.rank - 2]!;
    const positions = tokenPositions ?? arangeArray(newLen);

    let Q = splitHeads(this.WQ.forward(x), this.numHeads);
    let K = splitHeads(this.WK.forward(x), this.numHeads);
    let V = splitHeads(this.WV.forward(x), this.numHeads);

    if (this.rope) {
      Q = this.rope.forward(Q, positions);
      K = this.rope.forward(K, positions);
    }

    if (cache) {
      K = cache.k ? concatAxis(cache.k, K, K.rank - 2) : K;
      V = cache.v ? concatAxis(cache.v, V, V.rank - 2) : V;
      cache.k = K;
      cache.v = V;
    }
    const totalLen = K.shape[K.rank - 2]!;
    const mask = causalMaskWithCache(newLen, totalLen);

    const { output: attnOutput, weights } = scaledDotProductAttention(Q, K, V, mask);
    const merged = mergeHeads(attnOutput);
    const output = this.output_proj.forward(merged);

    // `attnWeights`: [..., numHeads, newLen, totalLen] — the classic per-head attention
    // heatmap. With a cache this is only the new rows, not the full seq x seq matrix.
    this.record({ input: x, attnWeights: weights, output });
    return output;
  }
}
