/** Private helpers for `aifn/embed`: dense views, distances, neighbours and the classical-MDS core. */

import { eigh } from 'aifn/linalg'
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

/** A square matrix's size and values, or throw. */
export function square(x: Tensor, where: string): { n: number; v: Float64Array } {
  if (x.shape.length !== 2 || x.shape[0] !== x.shape[1]) throw new Error(`${where}: expected a square matrix [n, n]`)
  return { n: x.shape[0], v: values(x) }
}

/** Squared Euclidean distances between all rows of v [n, d], flat [n, n]. */
export function squaredDistances(v: Float64Array, n: number, d: number): Float64Array {
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0
      for (let c = 0; c < d; c++) s += (v[i * d + c] - v[j * d + c]) ** 2
      out[i * n + j] = out[j * n + i] = s
    }
  }
  return out
}

/** The k nearest other points of each point under a distance matrix D [n, n]: indices [n][k], nearest first. */
export function nearestNeighbours(D: Float64Array, n: number, k: number): number[][] {
  if (!(k >= 1 && k < n)) throw new Error(`nearest neighbours: k must lie in 1 … ${n - 1}`)
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => D[i * n + a] - D[i * n + b] || a - b)
      .slice(0, k),
  )
}

/**
 * Classical MDS on squared distances D² [n, n] (Torgerson, 1952; Gower, 1966): B = −½ J D² J with J = I − 11ᵀ/n,
 * and the coordinates Y = V_r Λ_r^½ from B's top r eigenpairs (negative eigenvalues give zero columns).
 */
export function classicalCore(D2: Float64Array, n: number, r: number) {
  const B = new Float64Array(n * n)
  const row = new Float64Array(n)
  let all = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) row[i] += D2[i * n + j] / n
    all += row[i] / n
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) B[i * n + j] = -0.5 * (D2[i * n + j] - row[i] - row[j] + all)
  const e = eigh(fromData(B, [n, n]))
  const lambda = e.values.data as Float64Array
  const V = e.vectors.data as Float64Array
  const Y = new Float64Array(n * r)
  for (let c = 0; c < r; c++) {
    const s = Math.sqrt(Math.max(lambda[c], 0))
    for (let i = 0; i < n; i++) Y[i * r + c] = V[i * n + c] * s
  }
  return { Y, eigenvalues: Float64Array.from(lambda) }
}

export const mat = (v: Float64Array, n: number, d: number): Tensor => fromData(v, [n, d])
export const vec = (v: ArrayLike<number>): Tensor => fromData(Float64Array.from(v), [v.length])
