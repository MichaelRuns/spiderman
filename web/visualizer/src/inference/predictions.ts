export interface Prediction {
  tokenId: number;
  probability: number;
}

/** Softmax over raw logits, then the top `k` (id, probability) pairs, descending. */
export function topKPredictions(logits: Float32Array, k: number): Prediction[] {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i]! > maxLogit) maxLogit = logits[i]!;
  }
  let sum = 0;
  const probs = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const p = Math.exp(logits[i]! - maxLogit);
    probs[i] = p;
    sum += p;
  }

  const ranked = Array.from(probs, (p, tokenId) => ({ tokenId, probability: p / sum }));
  ranked.sort((a, b) => b.probability - a.probability);
  return ranked.slice(0, k);
}
