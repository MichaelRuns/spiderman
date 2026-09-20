import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { Tokenizer } from "../src/tokenizer.js";

/**
 * Cross-language parity: `tests/fixtures/tokenizer_parity.json` is generated
 * by the real Python `Tokenizer` (same trained vocab/merges as `web/weights/`)
 * — see `llm/src/spiderman_llm/scripts/gen_js_tokenizer_fixture.py`.
 */
const fixture: {
  specialTokens: string[];
  cases: Array<{ text: string; ids: number[]; decoded: string }>;
} = JSON.parse(readFileSync("./tests/fixtures/tokenizer_parity.json", "utf-8"));

function loadTokenizer(): Tokenizer {
  const rawVocab = JSON.parse(readFileSync("../weights/vocab.json", "utf-8"));
  const rawMerges = JSON.parse(readFileSync("../weights/merges.json", "utf-8"));
  return Tokenizer.fromJSON(rawVocab, rawMerges, fixture.specialTokens);
}

describe("Tokenizer parity with the Python implementation", () => {
  const tokenizer = loadTokenizer();

  for (const { text, ids, decoded } of fixture.cases) {
    const label = JSON.stringify(text.slice(0, 30));
    it(`encodes ${label} identically to Python`, () => {
      expect(tokenizer.encode(text)).toEqual(ids);
    });
    it(`decodes ${label}'s ids back to the same text`, () => {
      expect(tokenizer.decode(ids)).toEqual(decoded);
    });
  }
});

describe("Tokenizer basic behavior", () => {
  const tokenizer = loadTokenizer();

  it("round-trips arbitrary text through encode -> decode", () => {
    const text = "A little round-trip test, with punctuation & numbers 123.";
    expect(tokenizer.decode(tokenizer.encode(text))).toBe(text);
  });

  it("encodes the special token as a single id", () => {
    const ids = tokenizer.encode("<|endoftext|>");
    expect(ids).toHaveLength(1);
  });

  it("handles an empty string", () => {
    expect(tokenizer.encode("")).toEqual([]);
    expect(tokenizer.decode([])).toBe("");
  });
});
