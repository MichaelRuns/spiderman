import type { StepSnapshot } from "../inference/snapshot.js";
import { diverging, rgbToCss, sequentialBlue, sequentialOrange } from "./colors.js";
import { attentionLastPositionByHead, kvCacheToPositionMajor, lastRow } from "./tensorViews.js";

const SURFACE = "#14161d";
const BORDER = "#2a2e3a";
const INK = "#e6e8ee";
const MUTED = "#898781";

// Box fills loosely echo the classic Transformer figure's palette (pink embed,
// orange attention, blue feed-forward, yellow norm, lavender linear, green
// softmax) while the boxes themselves reflect our actual pre-norm/RoPE/SwiGLU
// architecture rather than the paper's post-LN one.
const COLOR = {
  embed: "#4a3a3a",
  norm: "#4a4530",
  attn: "#4a3520",
  ffn: "#20304a",
  linear: "#35304a",
  softmax: "#203a2a",
};

function normalizeActivation(v: number): number {
  return v / 3;
}
function normalizeMagnitude(v: number): number {
  return Math.abs(v) / 3;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r = 6): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  label: string,
  fill: string,
  sublabel?: string,
): void {
  roundRect(ctx, x, y, w, h, 6);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = BORDER;
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = INK;
  ctx.font = "600 12px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(label, x + 8, y + 6);

  if (sublabel) {
    ctx.fillStyle = MUTED;
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText(sublabel, x + 8, y + 21);
  }
}

/** A horizontal strip of thin colored bars, one per value in `data`. */
function drawStrip(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  data: Float32Array | null,
  colormap: (v: number) => [number, number, number],
  normalize: (v: number) => number,
): void {
  if (!data || data.length === 0) return;
  const cellW = w / data.length;
  for (let i = 0; i < data.length; i++) {
    ctx.fillStyle = rgbToCss(colormap(normalize(data[i]!)));
    ctx.fillRect(x + i * cellW, y, Math.max(cellW, 0.6), h);
  }
}

/** A `rows` x `cols` grid of colored cells packed into [x,y,w,h], row-major `data`. */
function drawGrid(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  data: Float32Array | null,
  rows: number,
  cols: number,
  colormap: (v: number) => [number, number, number],
  normalize: (v: number) => number,
): void {
  const cellW = w / cols;
  const cellH = h / rows;
  const filled = data ? data.length : 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      const rgb = idx < filled ? colormap(normalize(data![idx]!)) : ([26, 26, 25] as [number, number, number]);
      ctx.fillStyle = rgbToCss(rgb);
      ctx.fillRect(x + c * cellW, y + r * cellH, Math.max(cellW, 0.6), Math.max(cellH, 0.6));
    }
  }
}

function drawArrow(ctx: CanvasRenderingContext2D, x: number, yFrom: number, yTo: number): void {
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, yFrom);
  ctx.lineTo(x, yTo);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 3, yTo - 5);
  ctx.lineTo(x, yTo);
  ctx.lineTo(x + 3, yTo - 5);
  ctx.stroke();
}

export interface DiagramConfig {
  numLayers: number;
  numHeads: number;
  dK: number;
  dModel: number;
  contextLength: number;
}

export interface DiagramRow {
  key: string;
  label: string;
  y: number;
  height: number;
}

/** Computes the vertical layout (used both to draw and to size the canvas/scroll container). */
export function layoutRows(config: DiagramConfig): { rows: DiagramRow[]; totalHeight: number } {
  const specs: Array<[string, string, number]> = [
    ["embed", "Input Embedding", 46],
    ["ln1", "RMSNorm", 34],
    ["attn", "Masked Multi-Head Attention (RoPE)", 130],
    ["add1", "Add (residual)", 20],
    ["ln2", "RMSNorm", 34],
    ["ffn", "Feed Forward (SwiGLU)", 46],
    ["add2", "Add (residual)", 20],
    ["lnf", "RMSNorm (final)", 34],
    ["linear", "Linear (lm_head)", 46],
    ["softmax", "Softmax → output probabilities", 28],
  ];
  const gap = 22;
  let y = 12;
  const rows: DiagramRow[] = [];
  for (const [key, label, height] of specs) {
    rows.push({ key, label, y, height });
    y += height + gap;
  }
  return { rows, totalHeight: y };
}

export interface DrawOptions {
  config: DiagramConfig;
  step: StepSnapshot | null;
  selectedLayer: number;
  width: number;
}

export function drawArchitecture(ctx: CanvasRenderingContext2D, options: DrawOptions): number {
  const { config, step, selectedLayer, width } = options;
  const { rows, totalHeight } = layoutRows(config);
  const pad = 16;
  const boxX = pad;
  const boxW = width - pad * 2;
  const midX = pad + boxW / 2;

  ctx.clearRect(0, 0, width, totalHeight);
  ctx.fillStyle = SURFACE;
  ctx.fillRect(0, 0, width, totalHeight);

  const byKey = Object.fromEntries(rows.map((r) => [r.key, r]));
  const layer = step?.layers[selectedLayer] ?? null;

  // Connecting arrows between every consecutive box, drawn first so boxes sit on top.
  for (let i = 0; i < rows.length - 1; i++) {
    drawArrow(ctx, midX, rows[i]!.y + rows[i]!.height, rows[i + 1]!.y);
  }

  const embed = byKey.embed!;
  drawBox(ctx, boxX, embed.y, boxW, 24, embed.label, COLOR.embed);
  drawStrip(ctx, boxX + 4, embed.y + 28, boxW - 8, 14, step ? lastRow(step.embeddingOutput) : null, diverging, normalizeActivation);

  const ln1 = byKey.ln1!;
  drawBox(ctx, boxX, ln1.y, boxW, ln1.height, ln1.label, COLOR.norm, `layer ${selectedLayer}`);

  const attn = byKey.attn!;
  drawBox(ctx, boxX, attn.y, boxW, 24, attn.label, COLOR.attn);
  const attnData = layer ? attentionLastPositionByHead(layer.attnWeights) : null;
  drawGrid(ctx, boxX + 4, attn.y + 28, boxW - 8, 40, attnData, config.numHeads, config.contextLength, sequentialBlue, (v) => v);
  ctx.fillStyle = MUTED;
  ctx.font = "10px system-ui, sans-serif";
  ctx.fillText("attn: head ↓ · key position →", boxX + 4, attn.y + 70);

  const kCache = step ? kvCacheToPositionMajor(step.kvCacheK[selectedLayer] ?? null) : null;
  const vCache = step ? kvCacheToPositionMajor(step.kvCacheV[selectedLayer] ?? null) : null;
  drawStrip(ctx, boxX + 4, attn.y + 86, boxW - 8, 10, kCache, sequentialBlue, normalizeMagnitude);
  drawStrip(ctx, boxX + 4, attn.y + 98, boxW - 8, 10, vCache, sequentialOrange, normalizeMagnitude);
  ctx.fillStyle = MUTED;
  ctx.fillText("K (blue) / V (orange) cache, by position", boxX + 4, attn.y + 112);

  const add1 = byKey.add1!;
  drawBox(ctx, midX - 30, add1.y, 60, add1.height, "+", COLOR.norm);

  const ln2 = byKey.ln2!;
  drawBox(ctx, boxX, ln2.y, boxW, ln2.height, ln2.label, COLOR.norm);

  const ffn = byKey.ffn!;
  drawBox(ctx, boxX, ffn.y, boxW, 24, ffn.label, COLOR.ffn);
  drawStrip(ctx, boxX + 4, ffn.y + 28, boxW - 8, 14, layer ? lastRow(layer.blockOutput) : null, diverging, normalizeActivation);

  const add2 = byKey.add2!;
  drawBox(ctx, midX - 30, add2.y, 60, add2.height, "+", COLOR.norm);

  const lnf = byKey.lnf!;
  drawBox(ctx, boxX, lnf.y, boxW, lnf.height, lnf.label, COLOR.norm);

  const linear = byKey.linear!;
  drawBox(ctx, boxX, linear.y, boxW, 24, linear.label, COLOR.linear);
  drawStrip(ctx, boxX + 4, linear.y + 28, boxW - 8, 14, step ? lastRow(step.logits) : null, diverging, normalizeActivation);

  const softmax = byKey.softmax!;
  drawBox(ctx, boxX, softmax.y, boxW, softmax.height, softmax.label, COLOR.softmax);

  return totalHeight;
}
