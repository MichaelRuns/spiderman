import { NDArray, div, mapUnary, meanAxis, mul } from "../ndarray.js";
import { paramArray, type StateDict } from "../weights.js";
import { LayerModule } from "./module.js";

export class RMSNorm extends LayerModule {
  weight: NDArray; // [dModel]

  constructor(
    private readonly dModel: number,
    private readonly eps = 1e-5,
  ) {
    super();
    this.weight = NDArray.ones([dModel]);
  }

  loadWeights(stateDict: StateDict, prefix = ""): void {
    this.weight = paramArray(stateDict, `${prefix}weight`);
  }

  forward(x: NDArray): NDArray {
    const meanSq = meanAxis(mul(x, x), -1, true); // [..., 1]
    const rms = mapUnary(meanSq, (v) => Math.sqrt(v + this.eps));
    const normalized = div(x, rms);
    const output = mul(normalized, this.weight);
    this.record({ input: x, output });
    return output;
  }
}
