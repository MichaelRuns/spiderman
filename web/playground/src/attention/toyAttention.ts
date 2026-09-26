import { NDArray, causalMask, maskedFill, matmul, scale, scaledDotProductAttention, swapLastTwo } from "@spiderman/nn";

export const NUM_HEADS = 2;
export const SEQ = 3;
export const D_K = 4;

/** Deterministic, varied small values — not random each render, but not hand-typed either. */
export function defaultTensor(seed: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < NUM_HEADS * SEQ * D_K; i++) {
    out.push(Math.round(Math.sin(seed + i * 1.37) * 20) / 10);
  }
  return out;
}

export interface ToyAttentionResult {
  /** Q·Kᵀ / √d_k, flat [numHeads, seq, seq] — before any masking or softmax. */
  scores: number[];
  /** softmax(scores) (with the causal mask applied first, if requested) — the real scaledDotProductAttention's weights. */
  weights: number[];
  /** weights · V, flat [numHeads, seq, d_k] — the real scaledDotProductAttention's output. */
  output: number[];
}

/**
 * Runs the real @spiderman/nn attention math on tiny, fully-visible tensors.
 * `scores` is recomputed the same way scaledDotProductAttention does
 * internally (matmul + scale) purely so it can be shown as an intermediate
 * step; `weights`/`output` come straight from that real function, not a
 * second independent computation of the same thing.
 */
export function computeToyAttention(qFlat: number[], kFlat: number[], vFlat: number[], causal: boolean): ToyAttentionResult {
  const shape = [NUM_HEADS, SEQ, D_K];
  const Q = new NDArray(shape, Float32Array.from(qFlat));
  const K = new NDArray(shape, Float32Array.from(kFlat));
  const V = new NDArray(shape, Float32Array.from(vFlat));

  const scores = scale(matmul(Q, swapLastTwo(K)), 1 / Math.sqrt(D_K));
  const mask = causal ? causalMask(SEQ) : undefined;
  const maskedScores = mask ? maskedFill(scores, mask, -Infinity) : scores;

  const { weights, output } = scaledDotProductAttention(Q, K, V, mask);

  return {
    scores: Array.from((causal ? maskedScores : scores).data),
    weights: Array.from(weights.data),
    output: Array.from(output.data),
  };
}
