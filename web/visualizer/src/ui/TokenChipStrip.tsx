import type { Tokenizer } from "@spiderman/nn";
import { useMemo } from "react";
import type { StepSnapshot } from "../inference/snapshot.js";
import { tokenChipColor } from "./tokenColors.js";

interface TokenChipStripProps {
  text: string;
  tokenizer: Tokenizer | null;
  /** The most recently generated tokens — assumed to be the trailing `steps.length` tokens of `text`'s current tokenization. */
  steps: StepSnapshot[];
  selectedIndex: number;
  isFollowingLatest: boolean;
  onSelect: (index: number) => void;
  onFollowLatest: () => void;
}

/**
 * Live tokenization of the current text (typed or generated, it's all the same field) —
 * every token shown as a chip. The trailing chips that came from a generation run are
 * clickable (they have a captured snapshot to inspect); anything the user typed is a
 * plain, read-only preview of how it'll tokenize.
 */
export function TokenChipStrip({
  text,
  tokenizer,
  steps,
  selectedIndex,
  isFollowingLatest,
  onSelect,
  onFollowLatest,
}: TokenChipStripProps) {
  const tokens = useMemo(() => {
    if (!tokenizer || text.length === 0) return [];
    return tokenizer.encode(text).map((id) => ({ id, text: tokenizer.decode([id]) }));
  }, [text, tokenizer]);

  if (tokens.length === 0) {
    return <p className="token-strip-empty">Tokens will appear here as you type, or as the model generates.</p>;
  }

  const generatedCount = Math.min(steps.length, tokens.length);
  const typedCount = tokens.length - generatedCount;

  return (
    <div className="token-strip">
      <div className="token-strip-header">
        <span className="token-strip-count">{tokens.length} tokens</span>
        {!isFollowingLatest && (
          <button type="button" className="follow-latest-button" onClick={onFollowLatest}>
            ↳ back to live
          </button>
        )}
      </div>
      <div className="token-chips">
        {tokens.map((token, i) => {
          const display = token.text === "" ? "·" : token.text;
          const backgroundColor = tokenChipColor(token.id);
          if (i < typedCount) {
            return (
              <span key={i} className="token-chip preview" style={{ backgroundColor }} title={`id ${token.id}`}>
                {display}
              </span>
            );
          }
          const stepIndex = i - typedCount;
          return (
            <button
              key={i}
              type="button"
              className={stepIndex === selectedIndex ? "token-chip selected" : "token-chip"}
              style={{ backgroundColor }}
              onClick={() => onSelect(stepIndex)}
              title={`id ${token.id} — step ${stepIndex}, click to inspect`}
            >
              {display}
            </button>
          );
        })}
      </div>
    </div>
  );
}
