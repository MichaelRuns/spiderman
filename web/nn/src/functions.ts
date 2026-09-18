/**
 * Pure tensor -> tensor functions, mirroring the "functions" section of
 * spiderman_llm's nn.py (minus cross_entropy/training-only pieces — the JS
 * side is inference-only).
 */
import { NDArray, broadcastBinaryOp, mapUnary, matmul, maxAxis, maskedFill, sumAxis, swapLastTwo } from "./ndarray.js";

export function softmax(x: NDArray, axis = -1): NDArray {
  const ax = axis < 0 ? x.rank + axis : axis;
  const maxVals = maxAxis(x, ax, true);
  const shifted = broadcastBinaryOp(x, maxVals, (a, b) => a - b);
  const expX = mapUnary(shifted, Math.exp);
  const sumExp = sumAxis(expX, ax, true);
  return broadcastBinaryOp(expX, sumExp, (a, b) => a / b);
}

export function silu(x: NDArray): NDArray {
  return mapUnary(x, (v) => v / (1 + Math.exp(-v)));
}

export interface AttentionResult {
  output: NDArray;
  /** Post-softmax attention weights, kept around for visualization. */
  weights: NDArray;
}

export function scaledDotProductAttention(
  Q: NDArray,
  K: NDArray,
  V: NDArray,
  mask?: NDArray,
): AttentionResult {
  const dK = Q.shape[Q.rank - 1]!;
  const rawScores = matmul(Q, swapLastTwo(K));
  const scores = mapUnary(rawScores, (v) => v / Math.sqrt(dK));
  const maskedScores = mask ? maskedFill(scores, mask, -Infinity) : scores;
  const weights = softmax(maskedScores, -1);
  const output = matmul(weights, V);
  return { output, weights };
}
