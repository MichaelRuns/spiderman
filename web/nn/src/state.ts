import { NDArray } from "./ndarray.js";

/** A plain, serializable snapshot of a tensor — safe to hand to a UI thread or postMessage. */
export interface TensorSnapshot {
  shape: number[];
  data: Float32Array;
}

export function snapshot(x: NDArray): TensorSnapshot {
  return { shape: [...x.shape], data: x.data.slice() };
}
