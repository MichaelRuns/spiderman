/**
 * A minimal, dependency-free, row-major, always-contiguous ndarray.
 *
 * This is the JS counterpart to the tensor ops nn.py leans on via
 * `torch.einsum`/broadcasting. It only supports what the transformer
 * modules actually need (no autograd — the JS side is inference-only,
 * weights are trained in Python and imported).
 */
export class NDArray {
  readonly shape: number[];
  readonly data: Float32Array;

  constructor(shape: ReadonlyArray<number>, data?: Float32Array) {
    this.shape = [...shape];
    const size = shapeSize(this.shape);
    if (data) {
      if (data.length !== size) {
        throw new Error(`NDArray: data length ${data.length} does not match shape [${shape}] (size ${size})`);
      }
      this.data = data;
    } else {
      this.data = new Float32Array(size);
    }
  }

  get rank(): number {
    return this.shape.length;
  }

  get size(): number {
    return this.data.length;
  }

  static zeros(shape: ReadonlyArray<number>): NDArray {
    return new NDArray(shape);
  }

  static filled(shape: ReadonlyArray<number>, value: number): NDArray {
    return new NDArray(shape, new Float32Array(shapeSize(shape)).fill(value));
  }

  static ones(shape: ReadonlyArray<number>): NDArray {
    return NDArray.filled(shape, 1);
  }

  /** Build an NDArray from nested JS arrays, e.g. [[1,2],[3,4]]. Mostly for tests. */
  static fromNested(nested: unknown): NDArray {
    const shape: number[] = [];
    let cursor: unknown = nested;
    while (Array.isArray(cursor)) {
      shape.push(cursor.length);
      cursor = cursor[0];
    }
    const flat: number[] = [];
    const walk = (node: unknown): void => {
      if (Array.isArray(node)) node.forEach(walk);
      else flat.push(node as number);
    };
    walk(nested);
    return new NDArray(shape, Float32Array.from(flat));
  }

  clone(): NDArray {
    return new NDArray(this.shape, this.data.slice());
  }

  toNestedArray(): unknown {
    const build = (offset: number, dims: number[]): unknown => {
      if (dims.length === 0) return this.data[offset];
      const head = dims[0]!;
      const rest = dims.slice(1);
      const stride = rest.reduce((a, b) => a * b, 1);
      return Array.from({ length: head }, (_, i) => build(offset + i * stride, rest));
    };
    return build(0, this.shape);
  }

  reshape(newShape: ReadonlyArray<number>): NDArray {
    const resolved = resolveShape(newShape, this.size);
    if (shapeSize(resolved) !== this.size) {
      throw new Error(`Cannot reshape array of size ${this.size} into shape [${newShape}]`);
    }
    return new NDArray(resolved, this.data);
  }

  /** General axis permutation; always materializes a new contiguous array. */
  transpose(axes: ReadonlyArray<number>): NDArray {
    if (axes.length !== this.rank) {
      throw new Error(`transpose: expected ${this.rank} axes, got ${axes.length}`);
    }
    const newShape = axes.map((a) => this.shape[a]!);
    const oldStrides = rowMajorStrides(this.shape);
    const newStrides = rowMajorStrides(newShape);
    const out = new Float32Array(this.size);
    for (let linear = 0; linear < this.size; linear++) {
      let rem = linear;
      let oldOffset = 0;
      for (let d = 0; d < this.rank; d++) {
        const coord = Math.floor(rem / newStrides[d]!);
        rem -= coord * newStrides[d]!;
        oldOffset += coord * oldStrides[axes[d]!]!;
      }
      out[linear] = this.data[oldOffset]!;
    }
    return new NDArray(newShape, out);
  }
}

function shapeSize(shape: ReadonlyArray<number>): number {
  return shape.reduce((a, b) => a * b, 1);
}

function rowMajorStrides(shape: ReadonlyArray<number>): number[] {
  const strides = new Array(shape.length).fill(0);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = acc;
    acc *= shape[i]!;
  }
  return strides;
}

function resolveShape(shape: ReadonlyArray<number>, size: number): number[] {
  const negIdx = shape.findIndex((d) => d === -1);
  if (negIdx === -1) return [...shape];
  const known = shape.reduce((acc, d, i) => (i === negIdx ? acc : acc * d), 1);
  const resolved = [...shape];
  resolved[negIdx] = size / known;
  return resolved;
}

function broadcastShapes(a: ReadonlyArray<number>, b: ReadonlyArray<number>): number[] {
  const rank = Math.max(a.length, b.length);
  const result = new Array<number>(rank);
  for (let i = 0; i < rank; i++) {
    const da = a[a.length - rank + i] ?? 1;
    const db = b[b.length - rank + i] ?? 1;
    if (da !== db && da !== 1 && db !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`);
    }
    result[i] = Math.max(da, db);
  }
  return result;
}

/** Strides (in elements of `shape`'s own unit) to read `shape` as if it were `outShape`. */
function broadcastStrides(shape: ReadonlyArray<number>, outShape: ReadonlyArray<number>): number[] {
  const rank = outShape.length;
  const padded = new Array(rank).fill(1);
  for (let i = 0; i < shape.length; i++) padded[rank - shape.length + i] = shape[i];
  const ownStrides = rowMajorStrides(padded);
  return padded.map((d, i) => (d === 1 ? 0 : ownStrides[i]!));
}

export function broadcastBinaryOp(a: NDArray, b: NDArray, fn: (x: number, y: number) => number): NDArray {
  const outShape = broadcastShapes(a.shape, b.shape);
  const outStrides = rowMajorStrides(outShape);
  const aStrides = broadcastStrides(a.shape, outShape);
  const bStrides = broadcastStrides(b.shape, outShape);
  const out = new Float32Array(shapeSize(outShape));
  for (let linear = 0; linear < out.length; linear++) {
    let rem = linear;
    let aOff = 0;
    let bOff = 0;
    for (let d = 0; d < outShape.length; d++) {
      const coord = Math.floor(rem / outStrides[d]!);
      rem -= coord * outStrides[d]!;
      aOff += coord * aStrides[d]!;
      bOff += coord * bStrides[d]!;
    }
    out[linear] = fn(a.data[aOff]!, b.data[bOff]!);
  }
  return new NDArray(outShape, out);
}

export function mapUnary(x: NDArray, fn: (v: number) => number): NDArray {
  const out = new Float32Array(x.size);
  for (let i = 0; i < x.size; i++) out[i] = fn(x.data[i]!);
  return new NDArray(x.shape, out);
}

export const add = (a: NDArray, b: NDArray): NDArray => broadcastBinaryOp(a, b, (x, y) => x + y);
export const sub = (a: NDArray, b: NDArray): NDArray => broadcastBinaryOp(a, b, (x, y) => x - y);
export const mul = (a: NDArray, b: NDArray): NDArray => broadcastBinaryOp(a, b, (x, y) => x * y);
export const div = (a: NDArray, b: NDArray): NDArray => broadcastBinaryOp(a, b, (x, y) => x / y);
export const scale = (x: NDArray, s: number): NDArray => mapUnary(x, (v) => v * s);

/** Reduce a single axis with a binary accumulator; mirrors torch's `dim=` + `keepdim=`. */
export function reduceAxis(
  x: NDArray,
  axis: number,
  fn: (acc: number, v: number) => number,
  init: number,
  keepdims = false,
): NDArray {
  const ax = axis < 0 ? x.rank + axis : axis;
  const outShapeKept = x.shape.map((d, i) => (i === ax ? 1 : d));
  const xStrides = rowMajorStrides(x.shape);
  const outStrides = rowMajorStrides(outShapeKept);
  const out = new Float32Array(shapeSize(outShapeKept)).fill(init);
  for (let linear = 0; linear < x.size; linear++) {
    let rem = linear;
    let outOffset = 0;
    for (let d = 0; d < x.rank; d++) {
      const coord = Math.floor(rem / xStrides[d]!);
      rem -= coord * xStrides[d]!;
      outOffset += (d === ax ? 0 : coord) * outStrides[d]!;
    }
    out[outOffset] = fn(out[outOffset]!, x.data[linear]!);
  }
  const kept = new NDArray(outShapeKept, out);
  return keepdims ? kept : kept.reshape(outShapeKept.filter((_, i) => i !== ax));
}

export const sumAxis = (x: NDArray, axis: number, keepdims = false): NDArray =>
  reduceAxis(x, axis, (acc, v) => acc + v, 0, keepdims);

export const maxAxis = (x: NDArray, axis: number, keepdims = false): NDArray =>
  reduceAxis(x, axis, (acc, v) => Math.max(acc, v), -Infinity, keepdims);

export function meanAxis(x: NDArray, axis: number, keepdims = false): NDArray {
  const ax = axis < 0 ? x.rank + axis : axis;
  const n = x.shape[ax]!;
  return scale(sumAxis(x, ax, keepdims), 1 / n);
}

/**
 * Batched matmul: the last two dims of each operand are treated as matrix
 * dims ([m, k] @ [k, n] -> [m, n]); any leading dims are broadcast against
 * each other. Covers Linear (x @ W^T), attention scores (Q @ K^T), and
 * attention output (weights @ V).
 */
export function matmul(a: NDArray, b: NDArray): NDArray {
  if (a.rank < 2 || b.rank < 2) throw new Error("matmul requires rank >= 2 tensors");
  const m = a.shape[a.rank - 2]!;
  const k1 = a.shape[a.rank - 1]!;
  const k2 = b.shape[b.rank - 2]!;
  const n = b.shape[b.rank - 1]!;
  if (k1 !== k2) throw new Error(`matmul: inner dims mismatch ${k1} vs ${k2}`);

  const aBatch = a.shape.slice(0, -2);
  const bBatch = b.shape.slice(0, -2);
  const batchShape = broadcastShapes(aBatch, bBatch);
  const batchSize = shapeSize(batchShape);
  const batchStridesOut = rowMajorStrides(batchShape);
  const aBatchStrides = broadcastStrides(aBatch, batchShape);
  const bBatchStrides = broadcastStrides(bBatch, batchShape);
  const aMatStride = m * k1;
  const bMatStride = k2 * n;

  const out = new Float32Array(batchSize * m * n);
  for (let batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    let rem = batchIdx;
    let aBatchOffset = 0;
    let bBatchOffset = 0;
    for (let d = 0; d < batchShape.length; d++) {
      const coord = Math.floor(rem / batchStridesOut[d]!);
      rem -= coord * batchStridesOut[d]!;
      aBatchOffset += coord * aBatchStrides[d]!;
      bBatchOffset += coord * bBatchStrides[d]!;
    }
    const aBase = aBatchOffset * aMatStride;
    const bBase = bBatchOffset * bMatStride;
    const outBase = batchIdx * m * n;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let p = 0; p < k1; p++) {
          sum += a.data[aBase + i * k1 + p]! * b.data[bBase + p * n + j]!;
        }
        out[outBase + i * n + j] = sum;
      }
    }
  }
  return new NDArray([...batchShape, m, n], out);
}

export function swapLastTwo(x: NDArray): NDArray {
  const axes = x.shape.map((_, i) => i);
  const r = axes.length;
  [axes[r - 2], axes[r - 1]] = [axes[r - 1]!, axes[r - 2]!];
  return x.transpose(axes);
}

/** "... seq (h d) -> ... h seq d" */
export function splitHeads(x: NDArray, numHeads: number): NDArray {
  const dModel = x.shape[x.rank - 1]!;
  const dK = dModel / numHeads;
  const reshaped = x.reshape([...x.shape.slice(0, -1), numHeads, dK]);
  const r = reshaped.rank;
  const axes = reshaped.shape.map((_, i) => i);
  [axes[r - 3], axes[r - 2]] = [axes[r - 2]!, axes[r - 3]!];
  return reshaped.transpose(axes);
}

/** "... h seq d -> ... seq (h d)" */
export function mergeHeads(x: NDArray): NDArray {
  const r = x.rank;
  const axes = x.shape.map((_, i) => i);
  [axes[r - 3], axes[r - 2]] = [axes[r - 2]!, axes[r - 3]!];
  const swapped = x.transpose(axes);
  const h = swapped.shape[swapped.rank - 2]!;
  const dK = swapped.shape[swapped.rank - 1]!;
  return swapped.reshape([...swapped.shape.slice(0, -2), h * dK]);
}

/** Gather rows: table [N, ...rest], indices any shape -> [...indices.shape, ...rest]. */
export function gather(table: NDArray, indices: NDArray): NDArray {
  const restShape = table.shape.slice(1);
  const restSize = shapeSize(restShape);
  const outShape = [...indices.shape, ...restShape];
  const out = new Float32Array(indices.size * restSize);
  for (let i = 0; i < indices.size; i++) {
    const row = Math.round(indices.data[i]!);
    out.set(table.data.subarray(row * restSize, (row + 1) * restSize), i * restSize);
  }
  return new NDArray(outShape, out);
}

/** x[..., index] where the last axis has size >= index+1; drops the last axis. */
export function selectLast(x: NDArray, index: number): NDArray {
  const outShape = x.shape.slice(0, -1);
  const k = x.shape[x.rank - 1]!;
  const out = new Float32Array(shapeSize(outShape));
  for (let i = 0; i < out.length; i++) out[i] = x.data[i * k + index]!;
  return new NDArray(outShape, out);
}

/** Interleave a, b (same shape) into a new trailing axis of size 2. */
export function stackLast(a: NDArray, b: NDArray): NDArray {
  const out = new Float32Array(a.size * 2);
  for (let i = 0; i < a.size; i++) {
    out[i * 2] = a.data[i]!;
    out[i * 2 + 1] = b.data[i]!;
  }
  return new NDArray([...a.shape, 2], out);
}

export function arangeArray(n: number): NDArray {
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = i;
  return new NDArray([n], data);
}

export function causalMask(seqLen: number): NDArray {
  const data = new Float32Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      data[i * seqLen + j] = j <= i ? 1 : 0;
    }
  }
  return new NDArray([seqLen, seqLen], data);
}

/** Where mask == 0, replace with `value` (broadcasts like the other binary ops). */
export function maskedFill(x: NDArray, mask: NDArray, value: number): NDArray {
  return broadcastBinaryOp(x, mask, (v, m) => (m === 0 ? value : v));
}
