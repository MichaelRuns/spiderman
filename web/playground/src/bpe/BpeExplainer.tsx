import { useMemo, useState } from "react";
import { Tokenizer } from "@spiderman/nn";
import vocabJson from "../../../weights/vocab.json";
import mergesJson from "../../../weights/merges.json";
import manifestJson from "../../../weights/manifest.json";

/** Render a string for display: visible whitespace markers, `\xNN` otherwise. */
function renderBytes(s: string): string {
  let out = "";
  for (const ch of s) {
    const code = ch.codePointAt(0)!;
    if (code === 32) out += "␣";
    else if (code === 10) out += "⏎";
    else if (code === 9) out += "⇥";
    else if (code >= 33 && code <= 126) out += ch;
    else out += `\\x${code.toString(16).padStart(2, "0")}`;
  }
  return out;
}

const DEFAULT_TEXT = "Tokenizers turn text into numbers — unbelievable!";

export function BpeExplainer() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [chunkIdx, setChunkIdx] = useState(0);

  const tokenizer = useMemo(() => {
    const merges = mergesJson as unknown as Array<[number[], number[]]>;
    return Tokenizer.fromJSON(
      vocabJson as unknown as Record<string, number[]>,
      merges,
      ((manifestJson as unknown as { specialTokens?: string[] }).specialTokens ?? []),
    );
  }, []);

  const specialTokens = ((manifestJson as unknown as { specialTokens?: string[] }).specialTokens ?? []);
  const chunks = useMemo(() => tokenizer.pretokenize(text), [tokenizer, text]);
  const chunk = chunks[Math.min(chunkIdx, Math.max(0, chunks.length - 1))] ?? "";
  const isSpecial = specialTokens.includes(chunk);

  const steps = useMemo(
    () => (isSpecial || chunk === "" ? [] : tokenizer.traceMerges(chunk)),
    [tokenizer, chunk, isSpecial],
  );
  const finalParts = steps.length > 0 ? steps[steps.length - 1]!.parts : [];
  const ids = useMemo(() => (isSpecial || chunk === "" ? [] : tokenizer.encode(chunk)), [tokenizer, chunk, isSpecial]);

  // The first merges the BPE learner found — the most frequent byte pairs.
  const firstMerges = useMemo(() => {
    const merges = mergesJson as unknown as Array<[number[], number[]]>;
    const bytesToStr = (bytes: number[]) => String.fromCharCode(...bytes);
    return merges.slice(0, 12).map(([a, b], rank) => ({
      rank,
      left: renderBytes(bytesToStr(a)),
      right: renderBytes(bytesToStr(b)),
      merged: renderBytes(bytesToStr(a) + bytesToStr(b)),
    }));
  }, []);

  return (
    <div className="explainer">
      <p className="explainer-intro">
        BPE learns a vocabulary by repeatedly merging the most frequent adjacent byte pair —{" "}
        <code>traceMerges</code> replays that exact loop on one pretoken chunk using the real merge
        ranks, so every step below is what <code>encode</code> actually does. Merges never cross a
        pretoken boundary, which is why <code>␣un</code> and <code>un</code> can be different
        tokens.
      </p>

      <div className="controls">
        <label className="bpe-text-label">
          text
          <textarea
            className="bpe-text-input"
            rows={2}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setChunkIdx(0);
            }}
          />
        </label>
      </div>

      {chunks.length > 0 && (
        <div className="bpe-chunks">
          <span className="x-inputs-label">pretoken chunks:</span>
          {chunks.map((c, i) => (
            <button
              key={i}
              type="button"
              className={i === Math.min(chunkIdx, chunks.length - 1) ? "chip active" : "chip"}
              onClick={() => setChunkIdx(i)}
            >
              {renderBytes(c)}
            </button>
          ))}
        </div>
      )}

      {isSpecial ? (
        <p className="explainer-intro">
          <code>{chunk}</code> is a special token — it encodes as a single id with no BPE merges
          applied.
        </p>
      ) : (
        steps.length > 0 && (
          <>
            <div className="bpe-steps">
              {steps.map((step, i) => (
                <div key={i} className="bpe-step">
                  <span className="bpe-step-label">
                    {i === 0 ? "bytes" : `merge ${i}`}
                  </span>
                  <span className="bpe-parts">
                    {step.parts.map((part, j) => {
                      // The merged part sits at the first index where this
                      // step's parts differ from the previous step's.
                      let isNew = false;
                      if (i > 0) {
                        const prev = steps[i - 1]!.parts;
                        isNew = step.parts.findIndex((p, k) => p !== prev[k]) === j;
                      }
                      return (
                        <code key={j} className={isNew ? "token-chip merged-new" : "token-chip"}>
                          {renderBytes(part)}
                        </code>
                      );
                    })}
                  </span>
                  {step.merged !== null && step.rank !== null && (
                    <span className="bpe-merge-note">
                      merged <code>{renderBytes(step.merged[0])}</code> +{" "}
                      <code>{renderBytes(step.merged[1])}</code> (rank {step.rank})
                    </span>
                  )}
                </div>
              ))}
            </div>

            <div className="bpe-result">
              <span className="x-inputs-label">final tokens:</span>
              {finalParts.map((part, j) => (
                <code key={j} className="token-chip">
                  {renderBytes(part)}
                  <span className="token-id">#{ids[j]}</span>
                </code>
              ))}
              <span className="bpe-decode-note">decodes to {JSON.stringify(tokenizer.decode(ids))}</span>
            </div>
          </>
        )
      )}

      <h3 className="bpe-subhead">first merges the learner found</h3>
      <p className="explainer-intro">
        The merge list is ordered — rank 0 is the most frequent byte pair in the training corpus.
        These are the pairs this tokenizer learned first:
      </p>
      <table className="toy-grid bpe-merges-table">
        <thead>
          <tr>
            <th>rank</th>
            <th>left</th>
            <th>right</th>
            <th>merged</th>
          </tr>
        </thead>
        <tbody>
          {firstMerges.map((m) => (
            <tr key={m.rank}>
              <td>{m.rank}</td>
              <td>
                <code>{m.left}</code>
              </td>
              <td>
                <code>{m.right}</code>
              </td>
              <td>
                <code>{m.merged}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
