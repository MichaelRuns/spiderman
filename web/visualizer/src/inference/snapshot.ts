import type { KVCache, NDArray, TensorSnapshot, TransformerLM } from "@spiderman/nn";

export interface LayerSnapshot {
  blockOutput: TensorSnapshot;
  /** [numHeads, newLen, totalLen] this step — the full square matrix on the prefill step, one new row per decode step. */
  attnWeights: TensorSnapshot;
}

export interface StepSnapshot {
  step: number;
  tokenId: number;
  tokenText: string;
  isPrefill: boolean;
  embeddingOutput: TensorSnapshot;
  layers: LayerSnapshot[];
  lnFinalOutput: TensorSnapshot;
  logits: TensorSnapshot;
  /** Point-in-time KV cache tensors, one per layer — safe to hold onto (each forward() call produces fresh arrays, never mutates a previous step's). */
  kvCacheK: (NDArray | null)[];
  kvCacheV: (NDArray | null)[];
}

/** Uses `cloneState()` (not `getState()`) everywhere — see LayerModule's doc comment on why that matters when capturing across a decode loop. */
export function captureStep(
  model: TransformerLM,
  cache: KVCache,
  step: number,
  tokenId: number,
  tokenText: string,
  isPrefill: boolean,
): StepSnapshot {
  const embeddingOutput = model.token_embeddings.cloneState().output!;
  const layers = model.layers.map((layer) => ({
    blockOutput: layer.cloneState().output!,
    attnWeights: layer.attn.cloneState().attnWeights!,
  }));
  const lnFinalOutput = model.ln_final.cloneState().output!;
  const logits = model.cloneState().logits!;

  return {
    step,
    tokenId,
    tokenText,
    isPrefill,
    embeddingOutput,
    layers,
    lnFinalOutput,
    logits,
    kvCacheK: cache.layers.map((l) => l.k),
    kvCacheV: cache.layers.map((l) => l.v),
  };
}
