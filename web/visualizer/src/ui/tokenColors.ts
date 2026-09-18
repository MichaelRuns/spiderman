// Golden-angle hue rotation: stepping by ~137.5° spreads any number of ids
// across the hue wheel without visually clustering, even for a 4,000-entry
// vocab — no need to pick or validate a fixed palette per id.
const GOLDEN_ANGLE = 137.508;

/** A deterministic background tint for a token id — the same token always gets the same color, so repeated tokens are visually recognizable wherever they appear. */
export function tokenChipColor(tokenId: number, alpha = 0.26): string {
  const hue = (tokenId * GOLDEN_ANGLE) % 360;
  return `hsla(${hue.toFixed(1)}, 55%, 55%, ${alpha})`;
}
