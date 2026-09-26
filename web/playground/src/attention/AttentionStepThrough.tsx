import { useMemo, useState } from "react";
import { divergingCss, sequentialBlueCss } from "../shared/colors.js";
import { D_K, NUM_HEADS, SEQ, computeToyAttention, defaultTensor } from "./toyAttention.js";

function idx3(h: number, i: number, j: number, cols: number): number {
  return h * SEQ * cols + i * cols + j;
}

function fmt(v: number): string {
  if (v === -Infinity) return "-∞";
  return v.toFixed(2);
}

interface GridProps {
  head: number;
  rows: number;
  cols: number;
  values: number[];
  editable?: boolean;
  onChange?: (row: number, col: number, value: number) => void;
  colorize?: (v: number) => string;
  rowLabel?: (row: number) => string;
  colLabel?: (col: number) => string;
}

function Grid({ head, rows, cols, values, editable, onChange, colorize, rowLabel, colLabel }: GridProps) {
  return (
    <table className="toy-grid">
      {colLabel && (
        <thead>
          <tr>
            <th />
            {Array.from({ length: cols }, (_, c) => (
              <th key={c}>{colLabel(c)}</th>
            ))}
          </tr>
        </thead>
      )}
      <tbody>
        {Array.from({ length: rows }, (_, r) => (
          <tr key={r}>
            {rowLabel && <th>{rowLabel(r)}</th>}
            {Array.from({ length: cols }, (_, c) => {
              const v = values[idx3(head, r, c, cols)]!;
              const bg = colorize ? colorize(v) : undefined;
              return (
                <td key={c} style={{ backgroundColor: bg }}>
                  {editable ? (
                    <input
                      type="number"
                      step={0.5}
                      value={v}
                      onChange={(e) => onChange?.(r, c, Number(e.target.value))}
                    />
                  ) : (
                    fmt(v)
                  )}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function AttentionStepThrough() {
  const [q, setQ] = useState<number[]>(() => defaultTensor(1));
  const [k, setK] = useState<number[]>(() => defaultTensor(2));
  const [v, setV] = useState<number[]>(() => defaultTensor(3));
  const [causal, setCausal] = useState(false);

  const result = useMemo(() => computeToyAttention(q, k, v, causal), [q, k, v, causal]);

  function makeSetter(setter: (fn: (prev: number[]) => number[]) => void) {
    return (head: number, row: number, col: number, cols: number, value: number) => {
      setter((prev) => {
        const next = [...prev];
        next[idx3(head, row, col, cols)] = value;
        return next;
      });
    };
  }
  const setQAt = makeSetter(setQ);
  const setKAt = makeSetter(setK);
  const setVAt = makeSetter(setV);

  const tokenLabel = (t: number) => `tok ${t}`;
  const dimLabel = (d: number) => `d${d}`;

  return (
    <div className="explainer">
      <p className="explainer-intro">
        Scaled dot-product attention, on {NUM_HEADS} heads × {SEQ} tokens × d_k={D_K} — small enough that
        every number involved fits on screen. Edit Q, K, or V below and watch scores, weights, and the
        output recompute using the real <code>@spiderman/nn</code> functions (
        <code>matmul</code>, <code>scale</code>, <code>softmax</code>, <code>scaledDotProductAttention</code>
        ) — the same code the full model runs.
      </p>

      <label className="causal-toggle">
        <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} />
        causal mask (token i can only see tokens ≤ i)
      </label>

      {Array.from({ length: NUM_HEADS }, (_, h) => (
        <div key={h} className="head-section">
          <h3>head {h}</h3>

          <div className="qkv-row">
            <div>
              <h4>Q</h4>
              <Grid
                head={h}
                rows={SEQ}
                cols={D_K}
                values={q}
                editable
                onChange={(r, c, val) => setQAt(h, r, c, D_K, val)}
                colorize={divergingCss}
                rowLabel={tokenLabel}
                colLabel={dimLabel}
              />
            </div>
            <div>
              <h4>K</h4>
              <Grid
                head={h}
                rows={SEQ}
                cols={D_K}
                values={k}
                editable
                onChange={(r, c, val) => setKAt(h, r, c, D_K, val)}
                colorize={divergingCss}
                rowLabel={tokenLabel}
                colLabel={dimLabel}
              />
            </div>
            <div>
              <h4>V</h4>
              <Grid
                head={h}
                rows={SEQ}
                cols={D_K}
                values={v}
                editable
                onChange={(r, c, val) => setVAt(h, r, c, D_K, val)}
                colorize={divergingCss}
                rowLabel={tokenLabel}
                colLabel={dimLabel}
              />
            </div>
          </div>

          <p className="step-label">scores = Q·Kᵀ / √d_k{causal ? " (masked positions → -∞)" : ""}</p>
          <Grid
            head={h}
            rows={SEQ}
            cols={SEQ}
            values={result.scores}
            colorize={(val) => (val === -Infinity ? "rgba(0,0,0,0.4)" : divergingCss(val / 3))}
            rowLabel={tokenLabel}
            colLabel={tokenLabel}
          />

          <p className="step-label">weights = softmax(scores) — each row sums to 1</p>
          <Grid
            head={h}
            rows={SEQ}
            cols={SEQ}
            values={result.weights}
            colorize={sequentialBlueCss}
            rowLabel={tokenLabel}
            colLabel={tokenLabel}
          />

          <p className="step-label">output = weights · V — each token's weighted sum over V</p>
          <Grid
            head={h}
            rows={SEQ}
            cols={D_K}
            values={result.output}
            colorize={(val) => divergingCss(val / 3)}
            rowLabel={tokenLabel}
            colLabel={dimLabel}
          />
        </div>
      ))}
    </div>
  );
}
