import { NDArray, arangeArray } from "../ndarray.js";
import { type StateDict } from "../weights.js";
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
  rope: RoPE;
  token_embeddings: Embedding;
  layers: TransformerBlock[];
  ln_final: RMSNorm;
  lm_head: Linear;

  constructor(config: TransformerLMConfig) {
    super();
    const { vocabSize, contextLength, dModel, numLayers, numHeads, dFF, ropeTheta = 10000.0 } = config;
    this.dK = dModel / numHeads;
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

  /** `tokenIds`: [..., seq] integer token ids. Returns logits: [..., seq, vocabSize]. */
  forward(tokenIds: NDArray): NDArray {
    let x = this.token_embeddings.forward(tokenIds);
    const seqLen = x.shape[x.rank - 2]!;
    const positions = arangeArray(seqLen);
    for (const layer of this.layers) {
      x = layer.forward(x, positions);
    }
    x = this.ln_final.forward(x);
    const logits = this.lm_head.forward(x);
    this.record({ logits });
    return logits;
  }
}
