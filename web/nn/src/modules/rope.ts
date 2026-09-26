import { NDArray, add, gather, mul, selectLast, stackLast, sub } from "../ndarray.js";
import { type StateDict } from "../weights.js";
import { LayerModule } from "./module.js";

export class RoPE extends LayerModule {
  private readonly cosTable: NDArray; // [maxSeqLen, dK/2]
  private readonly sinTable: NDArray; // [maxSeqLen, dK/2]

  constructor(
    readonly dK: number,
    readonly maxSeqLen: number,
    readonly theta = 10000.0,
  ) {
    super();
    if (dK % 2 !== 0) throw new Error("dK must be even for RoPE.");
    const half = dK / 2;
    const cos = new Float32Array(maxSeqLen * half);
    const sin = new Float32Array(maxSeqLen * half);
    for (let m = 0; m < maxSeqLen; m++) {
      for (let i = 0; i < half; i++) {
        const freq = 1 / Math.pow(theta, (2 * i) / dK);
        const angle = m * freq;
        cos[m * half + i] = Math.cos(angle);
        sin[m * half + i] = Math.sin(angle);
      }
    }
    this.cosTable = new NDArray([maxSeqLen, half], cos);
    this.sinTable = new NDArray([maxSeqLen, half], sin);
  }

  // RoPE has no learned parameters — cos/sin tables are derived from `theta` at construction.
  loadWeights(): void {}

  /**
   * Per-pair rotation angle/cos/sin at a given position — the same numbers
   * `forward()` uses internally (read straight out of the precomputed
   * tables), exposed for callers that want to show the actual math rather
   * than just its result (e.g. a RoPE explainer). `thetaI[i]` is
   * `theta^(-2i/dK)`, so `angleRad[i] = position * thetaI[i]`.
   */
  anglesAt(position: number): { thetaI: number[]; angleRad: number[]; cos: number[]; sin: number[] } {
    const half = this.dK / 2;
    const thetaI: number[] = [];
    const angleRad: number[] = [];
    const cos: number[] = [];
    const sin: number[] = [];
    for (let i = 0; i < half; i++) {
      const freq = 1 / Math.pow(this.theta, (2 * i) / this.dK);
      thetaI.push(freq);
      angleRad.push(position * freq);
      cos.push(this.cosTable.data[position * half + i]!);
      sin.push(this.sinTable.data[position * half + i]!);
    }
    return { thetaI, angleRad, cos, sin };
  }

  forward(x: NDArray, positions: NDArray): NDArray {
    const cos = gather(this.cosTable, positions); // [seq, dK/2], broadcasts over leading dims
    const sin = gather(this.sinTable, positions);

    const half = this.dK / 2;
    const pairs = x.reshape([...x.shape.slice(0, -1), half, 2]);
    const xEven = selectLast(pairs, 0);
    const xOdd = selectLast(pairs, 1);

    const rotatedEven = sub(mul(xEven, cos), mul(xOdd, sin));
    const rotatedOdd = add(mul(xEven, sin), mul(xOdd, cos));

    const rotated = stackLast(rotatedEven, rotatedOdd).reshape(x.shape);
    this.record({ output: rotated });
    return rotated;
  }
}
