import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { NDArray } from "../src/ndarray.js";
import { TransformerLM } from "../src/modules/transformerLM.js";
import { stateDictFromJSON } from "../src/weights.js";

/**
 * Cross-language parity: `tests/fixtures/transformer_lm.json` was generated
 * by running spiderman_llm's real (Python) TransformerLM with a fixed seed
 * — see the generation script referenced in the PR/commit that added this
 * fixture. If this test fails, the JS port has diverged from the Python
 * model it's supposed to mirror.
 */
const fixturePath = fileURLToPath(new URL("./fixtures/transformer_lm.json", import.meta.url));
const fixture = JSON.parse(readFileSync(fixturePath, "utf-8")) as {
  config: {
    vocab_size: number;
    context_length: number;
    d_model: number;
    num_layers: number;
    num_heads: number;
    d_ff: number;
    theta: number;
  };
  tokenIds: number[][];
  stateDict: Record<string, { shape: number[]; data: number[] }>;
  logits: { shape: number[]; data: number[] };
};

describe("TransformerLM parity with the Python nn.py implementation", () => {
  it("matches Python logits exactly, given the same weights and input", () => {
    const model = new TransformerLM({
      vocabSize: fixture.config.vocab_size,
      contextLength: fixture.config.context_length,
      dModel: fixture.config.d_model,
      numLayers: fixture.config.num_layers,
      numHeads: fixture.config.num_heads,
      dFF: fixture.config.d_ff,
      ropeTheta: fixture.config.theta,
    });
    model.loadWeights(stateDictFromJSON(fixture.stateDict));

    const tokenIds = NDArray.fromNested(fixture.tokenIds);
    const logits = model.forward(tokenIds);

    expect(logits.shape).toEqual(fixture.logits.shape);
    const expected = fixture.logits.data;
    for (let i = 0; i < expected.length; i++) {
      expect(logits.data[i]).toBeCloseTo(expected[i]!, 4);
    }
  });
});
