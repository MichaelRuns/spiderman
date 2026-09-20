import type { NDArray } from "./ndarray.js";

/** One transformer block's cached keys/values: `[numHeads, cachedLen, dK]`, or empty before the first step. */
export interface LayerKVCache {
  k: NDArray | null;
  v: NDArray | null;
}

/**
 * Per-layer key/value cache for incremental (one-token-at-a-time) decoding.
 * `TransformerLM.forward(tokenIds, cache)` appends into it and reads from it
 * automatically; construct one per generation run and pass it to every call.
 */
export class KVCache {
  readonly layers: LayerKVCache[];

  constructor(numLayers: number) {
    this.layers = Array.from({ length: numLayers }, () => ({ k: null, v: null }));
  }

  /** Number of positions cached so far (same across every layer). */
  get length(): number {
    const k = this.layers[0]?.k;
    if (!k) return 0;
    return k.shape[k.rank - 2] ?? 0;
  }

  reset(): void {
    for (const layer of this.layers) {
      layer.k = null;
      layer.v = null;
    }
  }
}
