import type { NDArray, TensorSnapshot } from "@spiderman/nn";

/** Last row of a [rows, cols] tensor (or [1, cols] — same thing), as a plain Float32Array. */
export function lastRow(snapshot: TensorSnapshot): Float32Array {
  const cols = snapshot.shape[snapshot.shape.length - 1]!;
  return snapshot.data.slice(snapshot.data.length - cols);
}

/**
 * Any [numHeads, newLen, X] per-head tensor -> [numHeads, X], keeping only
 * the most recent position per head (uniform across both the prefill step,
 * where newLen>1, and decode steps, where newLen===1). Used for attention
 * weights (X=totalLen) as well as the step-local q/k/v (X=dK).
 */
export function lastPositionByHead(snapshot: TensorSnapshot): Float32Array {
  const [numHeads, newLen, x] = snapshot.shape as [number, number, number];
  const out = new Float32Array(numHeads * x);
  for (let h = 0; h < numHeads; h++) {
    const headBase = h * newLen * x + (newLen - 1) * x;
    out.set(snapshot.data.subarray(headBase, headBase + x), h * x);
  }
  return out;
}

/**
 * KV cache tensor [numHeads, cachedLen, dK] -> row-major [cachedLen,
 * numHeads*dK] (position-major instead of head-major), so a growing cache
 * naturally maps onto a texture whose filled prefix grows by whole rows.
 */
export function kvCacheToPositionMajor(cache: NDArray | null): Float32Array | null {
  if (!cache) return null;
  const [numHeads, cachedLen, dK] = cache.shape as [number, number, number];
  const out = new Float32Array(cachedLen * numHeads * dK);
  for (let pos = 0; pos < cachedLen; pos++) {
    for (let h = 0; h < numHeads; h++) {
      const srcBase = h * cachedLen * dK + pos * dK;
      const dstBase = pos * numHeads * dK + h * dK;
      out.set(cache.data.subarray(srcBase, srcBase + dK), dstBase);
    }
  }
  return out;
}
