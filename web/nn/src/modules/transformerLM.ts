import { NDArray, arangeArray } from "../ndarray.js";
import { type StateDict } from "../weights.js";
import type { KVCache } from "../kvcache.js";
import { Embedding } from "./embedding.js";
import { Linear } from "./linear.js";
import { LayerModule } from "./module.js";
import { RMSNorm } from "./rmsnorm.js";
import { RoPE } from "./rope.js";
import { TransformerBlock } from "./transformerBlock.js";

export interface TransformerLMConfig {
  vocabSize: number;
  contextLength: number;
  dModel: number;
  numLayers: number;
  numHeads: number;
  dFF: number;
  ropeTheta?: number;
}

export class TransformerLM extends LayerModule {
  dK: number;
  contextLength: number;
  rope: RoPE;
  token_embeddings: Embedding;
  layers: TransformerBlock[];
  ln_final: RMSNorm;
  lm_head: Linear;

  constructor(config: TransformerLMConfig) {
    super();
    const { vocabSize, contextLength, dModel, numLayers, numHeads, dFF, ropeTheta = 10000.0 } = config;
    this.dK = dModel / numHeads;
    this.contextLength = contextLength;
    this.rope = new RoPE(this.dK, contextLength, ropeTheta);
    this.token_embeddings = new Embedding(vocabSize, dModel);
    this.layers = Array.from({ length: numLayers }, () => new TransformerBlock(dModel, numHeads, dFF, this.rope));
    this.ln_final = new RMSNorm(dModel);
    this.lm_head = new Linear(dModel, vocabSize);
  }

  /** Keys match PyTorch's auto-derived `state_dict()` names for nn.py's TransformerLM. */
  loadWeights(stateDict: StateDict, prefix = ""): void {
    this.token_embeddings.loadWeights(stateDict, `${prefix}token_embeddings.`);
    this.layers.forEach((layer, i) => layer.loadWeights(stateDict, `${prefix}layers.${i}.`));
    this.ln_final.loadWeights(stateDict, `${prefix}ln_final.`);
    this.lm_head.loadWeights(stateDict, `${prefix}lm_head.`);
  }

  /**
   * `tokenIds`: [..., seq] integer token ids. Returns logits: [..., seq, vocabSize].
   *
   * With `cache` omitted, this is a plain full-sequence forward pass (positions
   * 0..seq-1). With `cache` given, `tokenIds` must be **unbatched** (shape
   * `[seq]`) — the new tokens to process this call (the whole prompt on the
   * first, "prefill" call against an empty cache, or a single token on each
   * later "decode" step) — and `cache` is grown in place, keeping a real
   * per-layer KV cache instead of recomputing over the whole history each step.
   */
  forward(tokenIds: NDArray, cache?: KVCache): NDArray {
    let x = this.token_embeddings.forward(tokenIds);
    const newLen = x.shape[x.rank - 2]!;
    const startPos = cache?.length ?? 0;
    if (startPos + newLen > this.contextLength) {
      throw new Error(
        `TransformerLM.forward: ${startPos + newLen} positions exceed contextLength=${this.contextLength}`,
      );
    }
    const positions = arangeArray(newLen, startPos);
    this.layers.forEach((layer, i) => {
      x = layer.forward(x, positions, cache?.layers[i]);
    });
    x = this.ln_final.forward(x);
    const logits = this.lm_head.forward(x);
    this.record({ logits });
    return logits;
  }
}
