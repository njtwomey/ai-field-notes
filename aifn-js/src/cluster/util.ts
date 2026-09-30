/** Private helpers for `aifn/cluster`: dense views, distances and small tensors. */

import { copy, fromData, isContiguous, type Tensor } from 'aifn/tensor'

/** The elements of `t` in row-major order as a Float64Array (shared when already dense float64; do not mutate). */
export function values(t: Tensor): Float64Array {
  let size = 1
  for (const d of t.shape) size *= d
  if (t.dtype === 'float64' && t.offset === 0 && isContiguous(t) && t.data.length === size)
    return t.data as Float64Array
  return copy(t, 'float64').data as Float64Array
}

/** Rows, columns and values of a matrix [n, d], or throw naming the caller. */
export function matrix(x: Tensor, where: string): { n: number; d: number; v: Float64Array } {
  if (x.shape.length !== 2) throw new Error(`${where}: expected a matrix [n, d], got shape [${x.shape.join(', ')}]`)
  return { n: x.shape[0], d: x.shape[1], v: values(x) }
}

/** Squared Euclidean distance between row i of `a` and row j of `b` (both of width d). */
export function sq(a: Float64Array, i: number, b: Float64Array, j: number, d: number): number {
  let s = 0
  for (let c = 0; c < d; c++) {
    const t = a[i * d + c] - b[j * d + c]
    s += t * t
  }
  return s
}

/** All pairwise Euclidean distances between the rows of v [n, d], as a flat [n, n] array. */
export function pairwise(v: Float64Array, n: number, d: number): Float64Array {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) out[i * n + j] = out[j * n + i] = Math.sqrt(sq(v, i, v, j, d))
  }
  return out
}

/** Nearest row of `c` [k, d] to row i of `v`: its index and squared distance (ties to the lower index). */
export function nearest(v: Float64Array, i: number, c: Float64Array, k: number, d: number): [number, number] {
  let best = 0
  let bestD = Infinity
  for (let j = 0; j < k; j++) {
    const t = sq(v, i, c, j, d)
    if (t < bestD) {
      bestD = t
      best = j
    }
  }
  return [best, bestD]
}

export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])
export const ints = (v: ArrayLike<number>): Tensor => fromData(Int32Array.from(v), [v.length])
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])

/** Labels renumbered 0, 1, … in order of first appearance; negative labels (noise) are kept. */
export function canonical(labels: ArrayLike<number>): Int32Array {
  const map = new Map<number, number>()
  return Int32Array.from(labels, (l) => {
    if (l < 0) return l
    if (!map.has(l)) map.set(l, map.size)
    return map.get(l)!
  })
}
