import { KVCache, NDArray, sampleToken, type SamplingOptions, type TransformerLM, type Tokenizer } from "@spiderman/nn";
import { captureStep, type StepSnapshot } from "./snapshot.js";

export interface GenerationOptions extends SamplingOptions {
  maxNewTokens: number;
}

export interface GenerationStep {
  step: number;
  tokenId: number;
  tokenText: string;
  /** Wall time (ms) for the forward() call that produced this step's token — prefill for step 0, one decode call for every step after. */
  latencyMs: number;
  /** True only for step 0, whose latency covers the whole prompt, not a single token — keep it out of per-token min/avg stats. */
  isPrefill: boolean;
  cacheLength: number;
  contextLength: number;
  done: boolean;
  /** Per-module state captured right after this step's forward() call, for visualization. */
  snapshot: StepSnapshot;
}

/**
 * Runs real autoregressive generation: encode the prompt, prefill the KV
 * cache in one forward call, then decode one token at a time, sampling and
 * yielding after each step so a caller can render/update live without
 * waiting for the whole run to finish.
 */
export async function* generate(
  model: TransformerLM,
  tokenizer: Tokenizer,
  prompt: string,
  options: GenerationOptions,
  stopTokenIds: ReadonlySet<number> = new Set(),
): AsyncGenerator<GenerationStep> {
  const promptIds = tokenizer.encode(prompt);
  if (promptIds.length === 0) {
    throw new Error("Prompt must encode to at least one token.");
  }
  if (promptIds.length >= model.contextLength) {
    throw new Error(`Prompt (${promptIds.length} tokens) already fills or exceeds contextLength=${model.contextLength}.`);
  }

  const cache = new KVCache(model.layers.length);

  let stepStart = performance.now();
  let logits = model.forward(NDArray.fromNested(promptIds), cache);
  let latencyMs = performance.now() - stepStart;

  for (let step = 0; step < options.maxNewTokens; step++) {
    const vocabSize = logits.shape[logits.shape.length - 1]!;
    const lastRow = logits.data.slice(logits.data.length - vocabSize);
    const tokenId = sampleToken(lastRow, options);
    const tokenText = tokenizer.decode([tokenId]);
    const isPrefill = step === 0;
    const atContextLimit = cache.length >= model.contextLength;
    const stop = stopTokenIds.has(tokenId) || atContextLimit;

    yield {
      step,
      tokenId,
      tokenText,
      latencyMs,
      isPrefill,
      cacheLength: cache.length,
      contextLength: model.contextLength,
      done: stop || step === options.maxNewTokens - 1,
      snapshot: captureStep(model, cache, step, tokenId, tokenText, isPrefill),
    };

    if (stop) break;

    stepStart = performance.now();
    logits = model.forward(NDArray.fromNested([tokenId]), cache);
    latencyMs = performance.now() - stepStart;
  }
}
