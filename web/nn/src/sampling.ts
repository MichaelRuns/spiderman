/**
 * Turns a row of logits into a sampled next-token id. Operates on a plain
 * `Float32Array` (the caller slices the last position's row out of the
 * model's logits `NDArray` first) — no tensor machinery needed here.
 */
export interface SamplingOptions {
  /** Softmax temperature. <= 0 means greedy (argmax). Default 1.0. */
  temperature?: number;
  /** Keep only the top K highest-probability tokens before sampling. */
  topK?: number;
  /** Nucleus sampling: keep the smallest prefix of tokens whose cumulative probability reaches this. */
  topP?: number;
}

export function sampleToken(logits: Float32Array, options: SamplingOptions = {}): number {
  const { temperature = 1.0, topK, topP } = options;

  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) {
      if (logits[i]! > logits[best]!) best = i;
    }
    return best;
  }

  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i]! > maxLogit) maxLogit = logits[i]!;
  }
  let sum = 0;
  const probs = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const p = Math.exp((logits[i]! - maxLogit) / temperature);
    probs[i] = p;
    sum += p;
  }
  for (let i = 0; i < probs.length; i++) probs[i] = probs[i]! / sum;

  let candidates = Array.from(probs, (p, i) => ({ i, p }));
  candidates.sort((a, b) => b.p - a.p);

  if (topK !== undefined && topK > 0 && topK < candidates.length) {
    candidates = candidates.slice(0, topK);
  }

  if (topP !== undefined && topP > 0 && topP < 1) {
    let cumulative = 0;
    let cutoff = candidates.length;
    for (let i = 0; i < candidates.length; i++) {
      cumulative += candidates[i]!.p;
      if (cumulative >= topP) {
        cutoff = i + 1;
        break;
      }
    }
    candidates = candidates.slice(0, cutoff);
  }

  const renormSum = candidates.reduce((s, c) => s + c.p, 0);
  let r = Math.random() * renormSum;
  for (const candidate of candidates) {
    r -= candidate.p;
    if (r <= 0) return candidate.i;
  }
  return candidates[candidates.length - 1]!.i;
}
