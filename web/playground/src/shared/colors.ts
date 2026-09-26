/**
 * A small, self-contained color helper for this package (mirrors the
 * approach in web/visualizer/src/scene/colors.ts, not shared as a package
 * across the two apps since each only needs a couple of colormaps).
 */

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

const DIVERGING_NEGATIVE: [number, number, number] = [227, 73, 72]; // red
const DIVERGING_MIDPOINT: [number, number, number] = [56, 56, 53]; // dark-surface neutral gray
const DIVERGING_POSITIVE: [number, number, number] = [57, 135, 229]; // blue

/** Signed value normalized to roughly [-1,1] -> "rgb(...)" via the diverging blue<->red pair. */
export function divergingCss(value: number, alpha = 1): string {
  const clamped = Math.max(-1, Math.min(1, value));
  const [from, to] = clamped >= 0 ? [DIVERGING_MIDPOINT, DIVERGING_POSITIVE] : [DIVERGING_MIDPOINT, DIVERGING_NEGATIVE];
  const t = Math.abs(clamped);
  const rgb = [lerp(from[0], to[0], t), lerp(from[1], to[1], t), lerp(from[2], to[2], t)];
  return `rgba(${Math.round(rgb[0]!)}, ${Math.round(rgb[1]!)}, ${Math.round(rgb[2]!)}, ${alpha})`;
}

const BLUE_LIGHT: [number, number, number] = [20, 22, 29]; // near the app surface
const BLUE_DARK: [number, number, number] = [42, 120, 214];

/** Magnitude in [0,1] -> "rgb(...)" via a single blue ramp (attention weights: 0=nothing, 1=all the mass). */
export function sequentialBlueCss(magnitude: number, alpha = 1): string {
  const t = Math.max(0, Math.min(1, magnitude));
  const rgb = [lerp(BLUE_LIGHT[0], BLUE_DARK[0], t), lerp(BLUE_LIGHT[1], BLUE_DARK[1], t), lerp(BLUE_LIGHT[2], BLUE_DARK[2], t)];
  return `rgba(${Math.round(rgb[0]!)}, ${Math.round(rgb[1]!)}, ${Math.round(rgb[2]!)}, ${alpha})`;
}
