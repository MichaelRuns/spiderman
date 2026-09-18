import { NDArray, matmul, swapLastTwo } from "../ndarray.js";
import { paramArray, type StateDict } from "../weights.js";
import { LayerModule } from "./module.js";

export class Linear extends LayerModule {
  weight: NDArray; // [outFeatures, inFeatures]
  private weightT: NDArray; // [inFeatures, outFeatures], cached for forward

  constructor(inFeatures: number, outFeatures: number) {
    super();
    this.weight = NDArray.zeros([outFeatures, inFeatures]);
    this.weightT = swapLastTwo(this.weight);
  }

  loadWeights(stateDict: StateDict, prefix = ""): void {
    this.weight = paramArray(stateDict, `${prefix}weight`);
    this.weightT = swapLastTwo(this.weight);
  }

  forward(x: NDArray): NDArray {
    const output = matmul(x, this.weightT);
    this.record({ input: x, output });
    return output;
  }
}
