/** Row selection for datasets (private): gather rows of a tensor by index, keeping its dtype. */

import { copy, fromData, toFlat, type Tensor } from 'aifn/tensor'

export type { Tensor }

/** Rows `index` of `t` along its first axis, as a new contiguous tensor of the same dtype. */
export function take(t: Tensor, index: Tensor | ArrayLike<number>): Tensor {
  const idx = 'shape' in index ? toFlat(index as Tensor) : Array.from(index as ArrayLike<number>)
  const src = copy(t)
  const row = t.shape.slice(1).reduce((a, b) => a * b, 1)
  const Ctor = src.data.constructor as new (n: number) => Float64Array | Float32Array | Int32Array
  const out = new Ctor(idx.length * row)
  idx.forEach((i, k) => out.set(src.data.subarray(i * row, (i + 1) * row), k * row))
  return fromData(out, [idx.length, ...t.shape.slice(1)])
}
