import { describe, expect, it } from "vitest";
import {
  NDArray,
  add,
  arangeArray,
  causalMask,
  gather,
  maskedFill,
  matmul,
  maxAxis,
  meanAxis,
  mergeHeads,
  mul,
  splitHeads,
  sumAxis,
  swapLastTwo,
} from "../src/ndarray.js";

describe("NDArray basics", () => {
  it("round-trips through fromNested/toNestedArray", () => {
    const x = NDArray.fromNested([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(x.shape).toEqual([2, 3]);
    expect(x.toNestedArray()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  it("reshape preserves data in row-major order", () => {
    const x = NDArray.fromNested([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
    expect(x.toNestedArray()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(x.reshape([3, -1]).toNestedArray()).toEqual([[1, 2], [3, 4], [5, 6]]);
  });

  it("transpose permutes axes", () => {
    const x = NDArray.fromNested([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(x.transpose([1, 0]).toNestedArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });
});

describe("broadcasting elementwise ops", () => {
  it("adds a row vector across a matrix", () => {
    const a = NDArray.fromNested([
      [1, 2],
      [3, 4],
    ]);
    const b = NDArray.fromNested([10, 20]);
    expect(add(a, b).toNestedArray()).toEqual([
      [11, 22],
      [13, 24],
    ]);
  });

  it("multiplies with a leading batch dim broadcast", () => {
    const a = NDArray.fromNested([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]); // [2,2,2]
    const b = NDArray.fromNested([
      [1, 0],
      [0, 1],
    ]); // [2,2]
    expect(mul(a, b).toNestedArray()).toEqual([
      [
        [1, 0],
        [0, 4],
      ],
      [
        [5, 0],
        [0, 8],
      ],
    ]);
  });
});

describe("reductions", () => {
  it("sums/means/maxes along an axis with and without keepdims", () => {
    const x = NDArray.fromNested([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(sumAxis(x, -1).toNestedArray()).toEqual([6, 15]);
    expect(sumAxis(x, -1, true).shape).toEqual([2, 1]);
    expect(meanAxis(x, 0).toNestedArray()).toEqual([2.5, 3.5, 4.5]);
    expect(maxAxis(x, -1).toNestedArray()).toEqual([3, 6]);
  });
});

describe("matmul", () => {
  it("multiplies plain 2D matrices", () => {
    const a = NDArray.fromNested([
      [1, 2],
      [3, 4],
    ]);
    const b = NDArray.fromNested([
      [5, 6],
      [7, 8],
    ]);
    expect(matmul(a, b).toNestedArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it("broadcasts a shared 2D weight across a batched 3D input", () => {
    const x = NDArray.fromNested([
      [
        [1, 0],
        [0, 1],
      ],
    ]); // [1,2,2]
    const w = NDArray.fromNested([
      [2, 0],
      [0, 3],
    ]); // [2,2]
    expect(matmul(x, w).toNestedArray()).toEqual([
      [
        [2, 0],
        [0, 3],
      ],
    ]);
  });

  it("swapLastTwo transposes only the trailing two axes", () => {
    const x = NDArray.fromNested([
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
    ]); // [1,2,3]
    expect(swapLastTwo(x).shape).toEqual([1, 3, 2]);
  });
});

describe("head split/merge round-trip", () => {
  it("splitHeads then mergeHeads is the identity", () => {
    const x = NDArray.fromNested(
      Array.from({ length: 2 }, (_, s) => Array.from({ length: 4 }, (_, d) => s * 4 + d)),
    ); // [seq=2, dModel=4]
    const split = splitHeads(x, 2); // [h=2, seq=2, dK=2]
    expect(split.shape).toEqual([2, 2, 2]);
    const merged = mergeHeads(split);
    expect(merged.shape).toEqual(x.shape);
    expect(merged.data).toEqual(x.data);
  });
});

describe("gather", () => {
  it("looks up rows by index, preserving index tensor shape", () => {
    const table = NDArray.fromNested([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const indices = NDArray.fromNested([
      [0, 2],
      [1, 1],
    ]);
    expect(gather(table, indices).toNestedArray()).toEqual([
      [
        [1, 2],
        [5, 6],
      ],
      [
        [3, 4],
        [3, 4],
      ],
    ]);
  });
});

describe("causal mask", () => {
  it("masks strictly-future positions", () => {
    const mask = causalMask(3);
    expect(mask.toNestedArray()).toEqual([
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
    ]);
  });

  it("maskedFill replaces masked-out entries", () => {
    const scores = NDArray.fromNested([
      [1, 1, 1],
      [1, 1, 1],
    ]);
    const mask = NDArray.fromNested([
      [1, 0, 1],
      [0, 1, 1],
    ]);
    expect(maskedFill(scores, mask, -Infinity).toNestedArray()).toEqual([
      [1, -Infinity, 1],
      [-Infinity, 1, 1],
    ]);
  });
});

describe("arangeArray", () => {
  it("produces 0..n-1", () => {
    expect(Array.from(arangeArray(4).data)).toEqual([0, 1, 2, 3]);
  });
});
