import { NDArray, gather } from "../ndarray.js";
import { paramArray, type StateDict } from "../weights.js";
import { LayerModule } from "./module.js";

export class Embedding extends LayerModule {
  weight: NDArray; // [numEmbeddings, embeddingDim]

  constructor(numEmbeddings: number, embeddingDim: number) {
    super();
    this.weight = NDArray.zeros([numEmbeddings, embeddingDim]);
  }

  loadWeights(stateDict: StateDict, prefix = ""): void {
    this.weight = paramArray(stateDict, `${prefix}weight`);
  }

  /** `indices` holds integer token ids (any shape); output is `[...indices.shape, embeddingDim]`. */
  forward(indices: NDArray): NDArray {
    const output = gather(this.weight, indices);
    this.record({ output });
    return output;
  }
}
