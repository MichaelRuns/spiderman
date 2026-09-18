import type { Tokenizer } from "@spiderman/nn";
import type { StepSnapshot } from "../inference/snapshot.js";
import { TokenChipStrip } from "./TokenChipStrip.js";

interface UnifiedInputProps {
  text: string;
  onChange: (text: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
  tokenizer: Tokenizer;
  steps: StepSnapshot[];
  selectedIndex: number;
  isFollowingLatest: boolean;
  onSelect: (index: number) => void;
  onFollowLatest: () => void;
}

/**
 * One box for both typing and generated output — no separate "output" area.
 * Users type directly here; Generate continues from whatever text currently
 * sits in the box (typed, generated, or edited after the fact) and appends
 * new tokens straight into it.
 */
export function UnifiedInput({
  text,
  onChange,
  onGenerate,
  isGenerating,
  tokenizer,
  steps,
  selectedIndex,
  isFollowingLatest,
  onSelect,
  onFollowLatest,
}: UnifiedInputProps) {
  return (
    <div className="unified-input">
      <textarea
        value={text}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Once upon a time..."
        rows={5}
        disabled={isGenerating}
      />
      <TokenChipStrip
        text={text}
        tokenizer={tokenizer}
        steps={steps}
        selectedIndex={selectedIndex}
        isFollowingLatest={isFollowingLatest}
        onSelect={onSelect}
        onFollowLatest={onFollowLatest}
      />
      <button type="button" onClick={onGenerate} disabled={isGenerating || text.trim().length === 0}>
        {isGenerating ? "Generating…" : "Generate"}
      </button>
    </div>
  );
}
