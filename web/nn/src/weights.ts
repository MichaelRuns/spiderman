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

/**
 * The manifest half of the binary export format written by
 * `llm/src/spiderman_llm/scripts/export_weights.py`: `weights.bin` (raw,
 * concatenated float32 bytes) plus this JSON index of where each tensor
 * lives in it. Deployed as `weights.bin` + `manifest.json` under
 * `web/weights/`.
 */
export interface WeightsManifest {
  config: {
    vocabSize: number;
    contextLength: number;
    dModel: number;
    numLayers: number;
    numHeads: number;
    dFF: number;
    ropeTheta: number;
  };
  specialTokens: string[];
  tensors: Record<string, { shape: number[]; byteOffset: number; byteLength: number }>;
}

/**
 * Build a StateDict from `weights.bin`'s ArrayBuffer by slicing a Float32Array
 * view per tensor at the manifest's recorded offsets — no copy beyond the
 * fetch itself (every offset is a multiple of 4 because every tensor's byte
 * length is, so alignment always holds).
 */
export function stateDictFromBinary(manifest: WeightsManifest, buffer: ArrayBuffer): StateDict {
  const stateDict: StateDict = {};
  for (const [key, { shape, byteOffset, byteLength }] of Object.entries(manifest.tensors)) {
    stateDict[key] = { shape, data: new Float32Array(buffer, byteOffset, byteLength / 4) };
  }
  return stateDict;
}
