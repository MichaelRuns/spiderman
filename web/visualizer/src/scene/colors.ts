/**
 * Colormaps for the 3D visualization, built from the dataviz skill's
 * validated palette (references/palette.md): a single sequential hue for
 * true magnitude (attention weights, K in the KV cache), a second
 * sequential hue for a second simultaneous magnitude context (V in the KV
 * cache — "the second takes the next categorical slot's hue"), and the
 * diverging blue<->red pair with a gray midpoint for signed activations.
 */

interface RampStop {
  t: number; // 0..1
  rgb: [number, number, number]; // 0..255
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// Sequential blue, steps 100->700 from palette.md, remapped to t=0..1.
const BLUE_RAMP: RampStop[] = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
].map((hex, i, arr) => ({ t: i / (arr.length - 1), rgb: hexToRgb(hex) }));

// Sequential orange (categorical slot 2), a matching light->dark ramp for the "second context."
const ORANGE_RAMP: RampStop[] = ["#fbe3d5", "#f5b998", "#eb6834", "#c04f22", "#8f3a19"].map((hex, i, arr) => ({
  t: i / (arr.length - 1),
  rgb: hexToRgb(hex),
}));

const DIVERGING_NEGATIVE = hexToRgb("#e34948"); // red pole
const DIVERGING_MIDPOINT = hexToRgb("#383835"); // dark-surface neutral gray
const DIVERGING_POSITIVE = hexToRgb("#3987e5"); // blue pole

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpRgb(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

function sampleRamp(ramp: RampStop[], t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < ramp.length - 1; i++) {
    const a = ramp[i]!;
    const b = ramp[i + 1]!;
    if (clamped >= a.t && clamped <= b.t) {
      const localT = (clamped - a.t) / (b.t - a.t || 1);
      return lerpRgb(a.rgb, b.rgb, localT);
    }
  }
  return ramp[ramp.length - 1]!.rgb;
}

/** Magnitude in [0,1] -> RGB via the sequential blue ramp (attention weights, K). */
export function sequentialBlue(magnitude: number): [number, number, number] {
  return sampleRamp(BLUE_RAMP, magnitude);
}

/** Magnitude in [0,1] -> RGB via the sequential orange ramp (V, "the second context"). */
export function sequentialOrange(magnitude: number): [number, number, number] {
  return sampleRamp(ORANGE_RAMP, magnitude);
}

/** Signed value, normalized to roughly [-1,1] -> RGB via the diverging blue<->red pair. */
export function diverging(value: number): [number, number, number] {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped >= 0) return lerpRgb(DIVERGING_MIDPOINT, DIVERGING_POSITIVE, clamped);
  return lerpRgb(DIVERGING_MIDPOINT, DIVERGING_NEGATIVE, -clamped);
}

/** RGB (0..255 each) as a CSS color string. */
export function rgbToCss([r, g, b]: [number, number, number]): string {
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}
