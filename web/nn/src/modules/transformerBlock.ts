import { NDArray, add } from "../ndarray.js";
import { type StateDict } from "../weights.js";
import type { LayerKVCache } from "../kvcache.js";
import { MultiHeadSelfAttention } from "./attention.js";
import { LayerModule } from "./module.js";
import { RMSNorm } from "./rmsnorm.js";
import type { RoPE } from "./rope.js";
import { SwiGLU } from "./swiglu.js";

export class TransformerBlock extends LayerModule {
  ln1: RMSNorm;
  ln2: RMSNorm;
  attn: MultiHeadSelfAttention;
  ffn: SwiGLU;

  constructor(dModel: number, numHeads: number, dFF: number, rope?: RoPE) {
    super();
    this.ln1 = new RMSNorm(dModel);
    this.ln2 = new RMSNorm(dModel);
    this.attn = new MultiHeadSelfAttention(dModel, numHeads, rope);
    this.ffn = new SwiGLU(dModel, dFF);
  }

  loadWeights(stateDict: StateDict, prefix = ""): void {
    this.ln1.loadWeights(stateDict, `${prefix}ln1.`);
    this.ln2.loadWeights(stateDict, `${prefix}ln2.`);
    this.attn.loadWeights(stateDict, `${prefix}attn.`);
    this.ffn.loadWeights(stateDict, `${prefix}ffn.`);
  }

  forward(x: NDArray, tokenPositions?: NDArray, layerCache?: LayerKVCache): NDArray {
    const afterAttn = add(x, this.attn.forward(this.ln1.forward(x), tokenPositions, layerCache));
    const output = add(afterAttn, this.ffn.forward(this.ln2.forward(afterAttn)));
    this.record({ input: x, output });
    return output;
  }
}
