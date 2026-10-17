import { stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Graph } from 'aifn-compute/graph'
import { laplacian } from 'aifn-compute/graph/matrices'
import { kNearestNeighbourGraph } from 'aifn-compute/graph/structures'
import { eigh } from 'aifn-compute/numerics/linalg'

/**
 * The graph the spectral figures of this chapter share: 30 "sensors" on a jittered 6 × 5 grid, each joined to its
 * 3 nearest neighbours (symmetrised, unweighted). The fixed seed gives a connected graph with 54 edges and distinct
 * Laplacian eigenvalues.
 */
export const COLS = 6
export const ROWS = 5
export const N = COLS * ROWS

const jitter = toRows(uniform(stream('gnn-foundations/sensors/6'), -0.3, 0.3, { shape: [N, 2] })) as number[][]
export const POSITIONS: [number, number][] = jitter.map(([a, b], i) => [(i % COLS) + a, Math.floor(i / COLS) + b])
export const GRAPH: Graph = kNearestNeighbourGraph(POSITIONS, 3)
export const EDGES: [number, number][] = GRAPH.edges.map((e) => [e.from, e.to])
export const EDGE_SEGMENTS = EDGES.map(([i, j]) => ({ from: POSITIONS[i], to: POSITIONS[j] }))
export const XS = POSITIONS.map((p) => p[0])
export const YS = POSITIONS.map((p) => p[1])

/** The unnormalised Laplacian of the edge counts, L = D − A. */
export const LAPLACIAN: Tensor = laplacian(GRAPH, { weighted: false })

const decomposition = eigh(LAPLACIAN)
const descending = Array.from(toFlat(decomposition.values))
const columns = toRows(decomposition.vectors) as number[][]

/** Eigenvalues in ascending order: λ₀ = 0 ≤ λ₁ ≤ … ≤ λ_{N−1}. */
export const EIGENVALUES: number[] = descending.map((_, k) => Math.max(0, descending[N - 1 - k]))

/**
 * Eigenvectors in the same order, EIGENVECTORS[k][i] = u_k(i). Each is signed so that its value at node 0 is
 * non-negative, so neighbouring frequencies keep a consistent orientation.
 */
export const EIGENVECTORS: number[][] = EIGENVALUES.map((_, k) => {
  const col = columns.map((row) => row[N - 1 - k])
  return col[0] < 0 ? col.map((v) => -v) : col
})

/** The graph Fourier transform x̂_k = u_kᵀ x. */
export function graphFourier(x: readonly number[]): number[] {
  return EIGENVECTORS.map((u) => u.reduce((s, v, i) => s + v * x[i], 0))
}

/** The inverse transform x = Σ_k x̂_k u_k. */
export function inverseGraphFourier(xhat: readonly number[]): number[] {
  return Array.from({ length: N }, (_, i) => xhat.reduce((s, c, k) => s + c * EIGENVECTORS[k][i], 0))
}

/** The Dirichlet energy xᵀ L x = Σ over edges of (x_i − x_j)². */
export function dirichletEnergy(x: readonly number[]): number {
  return EDGES.reduce((s, [i, j]) => s + (x[i] - x[j]) ** 2, 0)
}

/** The number of edges whose endpoints have values of opposite sign (zero crossings of a graph signal). */
export function signChanges(x: readonly number[]): number {
  return EDGES.reduce((s, [i, j]) => s + (x[i] * x[j] < 0 ? 1 : 0), 0)
}
