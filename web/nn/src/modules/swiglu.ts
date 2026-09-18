import { NDArray, mul } from "../ndarray.js";
import { silu } from "../functions.js";
import { type StateDict } from "../weights.js";
import { Linear } from "./linear.js";
import { LayerModule } from "./module.js";

export class SwiGLU extends LayerModule {
  w1: Linear; // dModel -> dFF
  w2: Linear; // dFF -> dModel
  w3: Linear; // dModel -> dFF

  constructor(dModel: number, dFF: number) {
    super();
    this.w1 = new Linear(dModel, dFF);
    this.w2 = new Linear(dFF, dModel);
    this.w3 = new Linear(dModel, dFF);
  }

  loadWeights(stateDict: StateDict, prefix = ""): void {
    this.w1.loadWeights(stateDict, `${prefix}w1.`);
    this.w2.loadWeights(stateDict, `${prefix}w2.`);
    this.w3.loadWeights(stateDict, `${prefix}w3.`);
  }

  forward(x: NDArray): NDArray {
    const gated = mul(silu(this.w1.forward(x)), this.w3.forward(x));
    const output = this.w2.forward(gated);
    this.record({ input: x, output });
    return output;
  }
}
