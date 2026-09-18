/**
 * Byte-level BPE tokenizer — encode/decode only, a JS port of
 * `llm/src/spiderman_llm/tokenizer.py`'s `Tokenizer` class so the visualizer
 * can turn a prompt into token ids and generated ids back into text.
 * Training (`train_bpe`) stays Python-only; this loads an already-trained
 * vocab/merges (as exported to `web/weights/vocab.json` + `merges.json`).
 *
 * A "byte-string" here is a plain JS `string` where every character's code
 * point is a raw byte value (0-255) — the JS stand-in for Python's `bytes`,
 * chosen so byte sequences can be used directly as Map keys.
 */

// Same pattern as llm/src/spiderman_llm/tokenizer.py's PRETOKENIZE_PATTERN (GPT-2's
// pretokenizer): keeps contractions, letter runs, digit runs, punctuation runs, and
// whitespace as separate chunks so merges never cross e.g. a word/space boundary.
const PRETOKENIZE_PATTERN = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

// Outside the [0,255] byte-value range, so it can never appear inside a byte-string —
// safe as an unambiguous separator when combining two byte-strings into one map key.
const PAIR_SEPARATOR = "￿";

function bytesToByteStr(bytes: ArrayLike<number>): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]!);
  return s;
}

function textToByteStr(text: string): string {
  return bytesToByteStr(new TextEncoder().encode(text));
}

function pairKey(a: string, b: string): string {
  return a + PAIR_SEPARATOR + b;
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export class Tokenizer {
  private readonly vocab: Map<number, string>;
  private readonly tokenToId: Map<string, number>;
  private readonly mergeRanks: Map<string, number>;
  private readonly specialTokens: string[];
  private readonly specialPattern: RegExp | null;

  constructor(vocab: Map<number, string>, merges: ReadonlyArray<[string, string]>, specialTokens: string[] = []) {
    this.vocab = new Map(vocab);
    this.mergeRanks = new Map(merges.map(([a, b], rank) => [pairKey(a, b), rank]));
    this.specialTokens = [...specialTokens];

    let nextId = 0;
    const existing = new Set(this.vocab.values());
    for (const id of this.vocab.keys()) {
      if (id + 1 > nextId) nextId = id + 1;
    }
    for (const token of this.specialTokens) {
      const tokenBytes = textToByteStr(token);
      if (!existing.has(tokenBytes)) {
        this.vocab.set(nextId, tokenBytes);
        existing.add(tokenBytes);
        nextId++;
      }
    }

    this.tokenToId = new Map();
    for (const [id, bytes] of this.vocab) this.tokenToId.set(bytes, id);

    this.specialPattern =
      this.specialTokens.length > 0
        ? new RegExp(
            `(${[...this.specialTokens]
              .sort((a, b) => b.length - a.length)
              .map(escapeRegExp)
              .join("|")})`,
          )
        : null;
  }

  /** Parse the JSON export format written by `Tokenizer.save` on the Python side. */
  static fromJSON(
    rawVocab: Record<string, number[]>,
    rawMerges: ReadonlyArray<[number[], number[]]>,
    specialTokens: string[] = [],
  ): Tokenizer {
    const vocab = new Map<number, string>();
    for (const [idStr, byteValues] of Object.entries(rawVocab)) {
      vocab.set(Number(idStr), bytesToByteStr(byteValues));
    }
    const merges: Array<[string, string]> = rawMerges.map(([a, b]) => [bytesToByteStr(a), bytesToByteStr(b)]);
    return new Tokenizer(vocab, merges, specialTokens);
  }

  private splitOnSpecials(text: string): Array<[string, boolean]> {
    if (!this.specialPattern) return [[text, false]];
    const result: Array<[string, boolean]> = [];
    for (const part of text.split(this.specialPattern)) {
      if (part.length === 0) continue;
      result.push([part, this.specialTokens.includes(part)]);
    }
    return result;
  }

  private encodePretoken(pretoken: string): number[] {
    let parts = Array.from(new TextEncoder().encode(pretoken), (b) => String.fromCharCode(b));
    while (parts.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.mergeRanks.get(pairKey(parts[i]!, parts[i + 1]!));
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      const merged = parts[bestIdx]! + parts[bestIdx + 1]!;
      parts = [...parts.slice(0, bestIdx), merged, ...parts.slice(bestIdx + 2)];
    }
    return parts.map((part) => {
      const id = this.tokenToId.get(part);
      if (id === undefined) throw new Error("Tokenizer: no vocab id for an encoded byte sequence");
      return id;
    });
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    for (const [chunk, isSpecial] of this.splitOnSpecials(text)) {
      if (isSpecial) {
        const id = this.tokenToId.get(textToByteStr(chunk));
        if (id === undefined) throw new Error(`Tokenizer: unknown special token "${chunk}"`);
        ids.push(id);
        continue;
      }
      for (const match of chunk.matchAll(PRETOKENIZE_PATTERN)) {
        ids.push(...this.encodePretoken(match[0]));
      }
    }
    return ids;
  }

  decode(ids: ReadonlyArray<number>): string {
    let totalLen = 0;
    const byteStrs = ids.map((id) => {
      const bytes = this.vocab.get(id);
      if (bytes === undefined) throw new Error(`Tokenizer: unknown token id ${id}`);
      totalLen += bytes.length;
      return bytes;
    });
    const bytes = new Uint8Array(totalLen);
    let offset = 0;
    for (const byteStr of byteStrs) {
      for (let i = 0; i < byteStr.length; i++) bytes[offset++] = byteStr.charCodeAt(i);
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  }
}
