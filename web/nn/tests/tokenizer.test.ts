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

describe("Tokenizer BPE merge tracing", () => {
  // Tiny hand-built tokenizer: vocab ids 0-4, merges learned in rank order.
  function tinyTokenizer(): Tokenizer {
    const vocab = new Map<number, string>([
      [0, "a"],
      [1, "b"],
      [2, "c"],
      [3, "ab"],
      [4, "abc"],
      [5, "bc"],
    ]);
    return new Tokenizer(vocab, [
      ["a", "b"], // rank 0
      ["ab", "c"], // rank 1
      ["b", "c"], // rank 2
    ]);
  }

  it("replays merges in rank order with the initial byte split first", () => {
    const steps = tinyTokenizer().traceMerges("abc");
    expect(steps[0]).toEqual({ parts: ["a", "b", "c"], merged: null, rank: null });
    expect(steps[1]).toEqual({ parts: ["ab", "c"], merged: ["a", "b"], rank: 0 });
    expect(steps[2]).toEqual({ parts: ["abc"], merged: ["ab", "c"], rank: 1 });
    expect(steps).toHaveLength(3);
  });

  it("prefers the lower-rank pair when several pairs could merge", () => {
    // ("b","c") is rank 2 but ("a","b") is rank 0, so "ab" merges first and
    // the trace then stops — "ab"+"c" already merged above, but here the
    // remaining parts are ["ab","c"] which merge at rank 1. To isolate the
    // preference, trace "bc": only ("b","c") applies.
    const steps = tinyTokenizer().traceMerges("bc");
    expect(steps[1]).toEqual({ parts: ["bc"], merged: ["b", "c"], rank: 2 });
  });

  it("stops when no learned pair applies", () => {
    const steps = tinyTokenizer().traceMerges("cb");
    expect(steps).toEqual([{ parts: ["c", "b"], merged: null, rank: null }]);
  });

  it("trace final parts match what encode produces", () => {
    const tokenizer = tinyTokenizer();
    const steps = tokenizer.traceMerges("abc");
    const ids = tokenizer.encode("abc");
    // "abc" is one pretoken; encode maps the trace's final parts to ids.
    expect(ids).toEqual([4]);
    expect(steps[steps.length - 1]!.parts).toEqual(["abc"]);
  });

  it("pretokenize splits on the GPT-2 pattern without encoding", () => {
    const tokenizer = tinyTokenizer();
    expect(tokenizer.pretokenize("hello world")).toEqual(["hello", " world"]);
    expect(tokenizer.pretokenize("don't")).toEqual(["don", "'t"]);
  });
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
