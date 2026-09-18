import { NDArray, arangeArray, causalMask, mergeHeads, splitHeads } from "../ndarray.js";
import { scaledDotProductAttention } from "../functions.js";
import { paramArray, type StateDict } from "../weights.js";
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

  /** `x`: [..., seq, dModel]. `tokenPositions` defaults to 0..seq-1. */
  forward(x: NDArray, tokenPositions?: NDArray): NDArray {
    const seqLen = x.shape[x.rank - 2]!;
    const positions = tokenPositions ?? arangeArray(seqLen);
    const mask = causalMask(seqLen);

    let Q = splitHeads(this.WQ.forward(x), this.numHeads);
    let K = splitHeads(this.WK.forward(x), this.numHeads);
    const V = splitHeads(this.WV.forward(x), this.numHeads);

    if (this.rope) {
      Q = this.rope.forward(Q, positions);
      K = this.rope.forward(K, positions);
    }

    const { output: attnOutput, weights } = scaledDotProductAttention(Q, K, V, mask);
    const merged = mergeHeads(attnOutput);
    const output = this.output_proj.forward(merged);

    // `attnWeights`: [..., numHeads, seq, seq] — the classic per-head attention heatmap.
    this.record({ input: x, attnWeights: weights, output });
    return output;
  }
}
