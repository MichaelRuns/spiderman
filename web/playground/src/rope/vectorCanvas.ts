const AQUA = "#1baf7a"; // before
const BLUE = "#3987e5"; // after
const MUTED = "#585a63";
const INK = "#e6e8ee";

/** Draws a small origin-centered 2D plot of `before` rotating to `after`, with an arc for the angle between them. */
export function drawVectorPair(
  canvas: HTMLCanvasElement,
  before: [number, number],
  after: [number, number],
  angleRad: number,
): void {
  const dpr = window.devicePixelRatio || 1;
  const size = canvas.clientWidth || 140;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, size, size);

  const cx = size / 2;
  const cy = size / 2;
  const maxMag = Math.max(1, Math.abs(before[0]), Math.abs(before[1]), Math.abs(after[0]), Math.abs(after[1]));
  const scale = (size / 2 - 16) / maxMag;

  // Axes.
  ctx.strokeStyle = MUTED;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(size, cy);
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, size);
  ctx.stroke();

  // Angle arc (radius independent of vector length, just to show rotation direction/size).
  if (Math.abs(angleRad) > 1e-6) {
    ctx.strokeStyle = MUTED;
    ctx.beginPath();
    const r = 18;
    // Canvas y grows downward, so flip the angle sign to match a standard math convention on screen.
    ctx.arc(cx, cy, r, -0, -angleRad, angleRad < 0);
    ctx.stroke();
  }

  const drawArrow = (v: [number, number], color: string): void => {
    const x = cx + v[0] * scale;
    const y = cy - v[1] * scale;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();

    const headLen = 6;
    const angle = Math.atan2(y - cy, x - cx);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - headLen * Math.cos(angle - Math.PI / 6), y - headLen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x - headLen * Math.cos(angle + Math.PI / 6), y - headLen * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  };

  drawArrow(before, AQUA);
  drawArrow(after, BLUE);

  ctx.font = "9px system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = AQUA;
  ctx.fillRect(4, size - 28, 8, 8);
  ctx.fillStyle = INK;
  ctx.fillText("before", 16, size - 24);
  ctx.fillStyle = BLUE;
  ctx.fillRect(4, size - 14, 8, 8);
  ctx.fillStyle = INK;
  ctx.fillText("after", 16, size - 10);
}
