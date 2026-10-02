/**
 * Shared by the embeddings of `aifn-applied/unsupervised/embedding`: squared distances between points (by
 * `aifn/numerics/linalg`) and their k-nearest-neighbour graphs.
 */

import { dense, fromData } from 'aifn/foundation/tensor'
import { squaredDistances as squaredDistanceMatrix } from 'aifn/numerics/linalg'
import { DomainError } from 'aifn/foundation/errors'

/** Squared Euclidean distances between all rows of v [n, d], flat [n, n] (`aifn/numerics/linalg`'s `squaredDistances`). */
export function squaredDistances(v: Float64Array, n: number, d: number): Float64Array {
  return dense.data(squaredDistanceMatrix(fromData(v, [n, d])))
}

/** The k nearest other points of each point under a distance matrix D [n, n]: indices [n][k], nearest first. */
export function nearestNeighbours(D: Float64Array, n: number, k: number): number[][] {
  if (!(k >= 1 && k < n)) throw new DomainError('nearest neighbours', `nearest neighbours: k must lie in 1 … ${n - 1}`)
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => D[i * n + a] - D[i * n + b] || a - b)
      .slice(0, k),
  )
}
