import type { Tokenizer } from "@spiderman/nn";
import { topKPredictions } from "../inference/predictions.js";
import type { StepSnapshot } from "../inference/snapshot.js";
import { lastRow } from "../scene/tensorViews.js";

interface NextTokenRankingProps {
  step: StepSnapshot | null;
  tokenizer: Tokenizer;
  topK?: number;
}

/** What the model's softmax actually ranked as the next token, at the currently-selected step — with the token it sampled highlighted. */
export function NextTokenRanking({ step, tokenizer, topK = 8 }: NextTokenRankingProps) {
  if (!step) {
    return <p className="ranking-empty">Generate to see ranked next-token probabilities.</p>;
  }

  const predictions = topKPredictions(lastRow(step.logits), topK);
  const maxProb = predictions[0]?.probability ?? 1;

  return (
    <ul className="next-token-ranking">
      {predictions.map((p) => {
        const isSelected = p.tokenId === step.tokenId;
        const label = tokenizer.decode([p.tokenId]);
        return (
          <li key={p.tokenId} className={isSelected ? "ranking-row selected" : "ranking-row"}>
            <span className="ranking-token">{label === "" ? "·" : label}</span>
            <span className="ranking-bar-track">
              <span className="ranking-bar" style={{ width: `${(p.probability / maxProb) * 100}%` }} />
            </span>
            <span className="ranking-prob">{(p.probability * 100).toFixed(1)}%</span>
          </li>
        );
      })}
    </ul>
  );
}
