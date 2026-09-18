import { NDArray } from "./ndarray.js";

/**
 * The JS-side shape of a PyTorch `model.state_dict()`: parameter name ->
 * flat data + shape. Every module's `loadWeights` reads exactly the keys
 * PyTorch would auto-derive from the matching nn.py attribute names (e.g.
 * `MultiHeadSelfAttention.WQ` -> `"...WQ.weight"`), so a Python export step
 * can dump `state_dict()` more or less as-is and this loader consumes it
 * with no name translation.
 */
export interface StateDict {
  [key: string]: { shape: number[]; data: Float32Array };
}

export function paramArray(stateDict: StateDict, key: string): NDArray {
  const entry = stateDict[key];
  if (!entry) {
    throw new Error(`Missing parameter "${key}" in state dict (have: ${Object.keys(stateDict).join(", ")})`);
  }
  return new NDArray(entry.shape, entry.data);
}

/** Parse the JSON fixture/export format: { [key]: { shape: number[], data: number[] } }. */
export function stateDictFromJSON(raw: Record<string, { shape: number[]; data: number[] }>): StateDict {
  const stateDict: StateDict = {};
  for (const [key, { shape, data }] of Object.entries(raw)) {
    stateDict[key] = { shape, data: Float32Array.from(data) };
  }
  return stateDict;
}
