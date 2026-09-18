import { useEffect, useMemo } from "react";
import * as THREE from "three";

interface HeatmapPlaneProps {
  /** Texture grid size — width = columns, height = rows. */
  width: number;
  height: number;
  /** Row-major values for the currently-filled prefix; cells beyond this stay `emptyColor`. Null = nothing filled yet. */
  data: Float32Array | null;
  colormap: (v: number) => [number, number, number];
  /** Maps a raw data value to whatever range `colormap` expects (e.g. [0,1] or [-1,1]). */
  normalize?: (v: number) => number;
  planeWidth: number;
  planeHeight: number;
  emptyColor?: [number, number, number];
  position?: [number, number, number];
}

/**
 * A DataTexture-backed heatmap plane, preallocated once at [width,height]
 * and updated in place every step (mutate `.image.data`, flip
 * `needsUpdate`) rather than reallocated — avoids churning GPU memory on
 * every generated token. Used for activation strips, attention weights, and
 * the KV cache stack alike.
 */
export function HeatmapPlane({
  width,
  height,
  data,
  colormap,
  normalize = (v) => v,
  planeWidth,
  planeHeight,
  emptyColor = [26, 26, 25],
  position = [0, 0, 0],
}: HeatmapPlaneProps) {
  const pixels = useMemo(() => new Uint8Array(width * height * 4), [width, height]);
  const texture = useMemo(() => {
    const tex = new THREE.DataTexture(pixels, width, height, THREE.RGBAFormat);
    tex.needsUpdate = true;
    return tex;
  }, [pixels, width, height]);

  useEffect(() => {
    const filled = data ? data.length : 0;
    const total = width * height;
    for (let i = 0; i < total; i++) {
      const rgb = i < filled ? colormap(normalize(data![i]!)) : emptyColor;
      pixels[i * 4] = rgb[0];
      pixels[i * 4 + 1] = rgb[1];
      pixels[i * 4 + 2] = rgb[2];
      pixels[i * 4 + 3] = 255;
    }
    texture.needsUpdate = true;
  }, [data, width, height, pixels, texture, colormap, normalize, emptyColor]);

  return (
    <mesh position={position}>
      <planeGeometry args={[planeWidth, planeHeight]} />
      <meshBasicMaterial map={texture} toneMapped={false} />
    </mesh>
  );
}
