import type { NDArray } from "../ndarray.js";
import { snapshot, type TensorSnapshot } from "../state.js";
import type { StateDict } from "../weights.js";

/**
 * Base class for every nn module. Beyond `forward`, each module records the
 * tensors worth visualizing from its last forward pass (inputs/outputs,
 * attention weights, etc.) so a visualizer can walk the tree after the fact
 * without threading extra plumbing through every call site.
 */
export abstract class LayerModule {
  /** Set to false to skip snapshotting (e.g. for a perf-sensitive bulk run). */
  recordState = true;

  private lastState: Record<string, TensorSnapshot> = {};

  protected record(entries: Record<string, NDArray>): void {
    if (!this.recordState) return;
    for (const [key, value] of Object.entries(entries)) {
      this.lastState[key] = snapshot(value);
    }
  }

  /** Snapshot of the tensors recorded during this module's last forward call. */
  getState(): Readonly<Record<string, TensorSnapshot>> {
    return this.lastState;
  }

  /** Load this module's parameters (and any submodules') from a state dict. */
  abstract loadWeights(stateDict: StateDict, prefix: string): void;
}
