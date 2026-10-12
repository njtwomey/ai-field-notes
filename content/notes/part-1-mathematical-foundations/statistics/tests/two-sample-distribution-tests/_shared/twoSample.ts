/**
 * The running problems of the two-sample notes and the distribution-free statistics they compare, composed from engine
 * primitives: kernel matrices (`gram`), distance matrices (`pairwiseDistances`), the engine's permutation test, and its
 * t, rank, Kolmogorov–Smirnov and binomial tests.
 *
 * The permutation tests permute point indices, not values: `permutationTest` shuffles one-dimensional samples, so it is
 * given the indices 0 … n + m − 1 and the statistic reads a precomputed matrix. One kernel or distance matrix per data
 * set then serves every permutation.
 *
 * A candidate for the engine: `mmdTest(x, y, { kernel, estimator: 'unbiased' | 'biased', resamples })` and
 * `energyTest(x, y, { resamples })` in `aifn-compute/probability/tests`, returning a `TestResult` whose `null` is the
 * permutation distribution, beside `mmdWitness(x, y, kernel, at)` and `classifierTwoSampleTest(x, y, classifier)`.
 */
import { child, normals, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { classProbabilities, dataset, hasPredictive, type Distribution } from 'aifn-compute/learning/estimators'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { pairwiseDistances } from 'aifn-compute/numerics/linalg'
import { normalLogPdf } from 'aifn-compute/numerics/special'
import { median, permutationTest } from 'aifn-compute/probability/stats'
import { binomialTest, ksTest, mannWhitneyU, welchTTest } from 'aifn-compute/probability/tests'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'

/** One point per row; one-dimensional samples are rows of length 1. */
export type Points = number[][]

/** How the second sample differs from N(0, 1) in one dimension. */
export type ShiftKind = 'mean' | 'scale' | 'shape'

export const SHIFT_LABELS: Record<ShiftKind, string> = {
  mean: 'mean shift: N(δ, 1)',
  scale: 'scale change: N(0, (1 + δ)²)',
  shape: 'shape change: two bumps, same mean and variance',
}

/**
 * n draws from the alternative of strength δ ≥ 0, given n standard normal draws z and n uniform draws u. The shape
 * change is the mixture ½N(−a, 1 − a²) + ½N(a, 1 − a²) with a = 0.95 min(δ, 1), which keeps the mean at 0 and the
 * variance at 1, so only the shape differs from N(0, 1).
 */
export function alternative(kind: ShiftKind, delta: number, z: readonly number[], u: readonly number[]): number[] {
  if (kind === 'mean') return z.map((v) => v + delta)
  if (kind === 'scale') return z.map((v) => (1 + delta) * v)
  const a = 0.95 * Math.min(delta, 1)
  const s = Math.sqrt(1 - a * a)
  return z.map((v, i) => (u[i] < 0.5 ? -a : a) + s * v)
}

/** A pair of one-dimensional samples: n from N(0, 1) and m from the alternative, from one seed. */
export function oneDimensionalPair(seed: number | Stream, n: number, m: number, kind: ShiftKind, delta: number) {
  const root = typeof seed === 'number' ? stream(seed) : seed
  const x = toFlat(normals(child(root, 'x'), [n]))
  const z = toFlat(normals(child(root, 'y'), [m]))
  const u = Array.from({ length: m }, (_, i) => uniform(child(root, 'u', i)))
  return { x, y: alternative(kind, delta, z, u) }
}

/** The density of the alternative of strength δ at x, for drawing it beside N(0, 1). */
export function alternativeDensity(kind: ShiftKind, delta: number, x: number): number {
  if (kind === 'mean') return Math.exp(normalLogPdf(x - delta))
  if (kind === 'scale') return Math.exp(normalLogPdf(x / (1 + delta))) / (1 + delta)
  const a = 0.95 * Math.min(delta, 1)
  const s = Math.sqrt(1 - a * a)
  return (0.5 * (Math.exp(normalLogPdf((x + a) / s)) + Math.exp(normalLogPdf((x - a) / s)))) / s
}

/**
 * The two-dimensional problem: n points from P = N(0, I) and m from Q, a unit-variance Gaussian with mean μ and
 * correlation ρ. With μ = 0 and ρ ≠ 0, Q differs from P only in its dependence: both marginals are N(0, 1).
 */
export function twoDimensionalPair(seed: number, n: number, m: number, mu: readonly [number, number], rho: number) {
  const root = stream(seed)
  const zx = toRows(normals(child(root, 'x'), [n, 2])) as number[][]
  const zy = toRows(normals(child(root, 'y'), [m, 2])) as number[][]
  const c = Math.sqrt(1 - rho * rho)
  return { x: zx, y: zy.map(([a, b]) => [mu[0] + a, mu[1] + rho * a + c * b]) }
}

/** Rows of length 1 from a one-dimensional sample. */
export const asRows = (v: readonly number[]): Points => v.map((x) => [x])

/** The Gaussian kernel matrix exp(−‖a − b‖²/2σ²) of the pooled sample, row-major [N × N]. */
export function kernelMatrix(pooled: Points, sigma: number): Float64Array {
  const t = tensor(pooled)
  return Float64Array.from(toFlat(gram(rbf({ lengthscale: sigma }), t, t)))
}

/** The Euclidean distance matrix of the pooled sample, row-major [N × N]. */
export function distanceMatrix(pooled: Points): Float64Array {
  return Float64Array.from(toFlat(pairwiseDistances(pooled) as Tensor))
}

/** The median distance between distinct pooled points: the usual default kernel width ("median heuristic"). */
export function medianHeuristic(pooled: Points): number {
  const D = distanceMatrix(pooled)
  const N = pooled.length
  const off: number[] = []
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) off.push(D[i * N + j])
  return median(off)
}

/** Sum of M[i, j] over i in a and j in b, skipping i = j when `distinct`. */
function blockSum(M: Float64Array, N: number, a: ArrayLike<number>, b: ArrayLike<number>, distinct: boolean) {
  let s = 0
  for (let p = 0; p < a.length; p++) {
    const row = a[p] * N
    for (let q = 0; q < b.length; q++) if (!distinct || a[p] !== b[q]) s += M[row + b[q]]
  }
  return s
}

/**
 * MMD² between the groups of pooled indices ix and iy, from the kernel matrix K. `unbiased` is the U-statistic,
 * which leaves out the k(x_i, x_i) terms and can be negative; otherwise the biased V-statistic, the squared distance
 * between the two empirical mean embeddings.
 */
export function mmd2(K: Float64Array, N: number, ix: ArrayLike<number>, iy: ArrayLike<number>, unbiased = true) {
  const n = ix.length
  const m = iy.length
  const xx = blockSum(K, N, ix, ix, unbiased) / (unbiased ? n * (n - 1) : n * n)
  const yy = blockSum(K, N, iy, iy, unbiased) / (unbiased ? m * (m - 1) : m * m)
  const xy = blockSum(K, N, ix, iy, false) / (n * m)
  return xx + yy - 2 * xy
}

/**
 * The energy distance 2 E‖X − Y‖ − E‖X − X′‖ − E‖Y − Y′‖ between the groups ix and iy, from the distance matrix D,
 * as the V-statistic of Székely and Rizzo.
 */
export function energy(D: Float64Array, N: number, ix: ArrayLike<number>, iy: ArrayLike<number>) {
  const n = ix.length
  const m = iy.length
  return (
    (2 * blockSum(D, N, ix, iy, false)) / (n * m) -
    blockSum(D, N, ix, ix, false) / (n * n) -
    blockSum(D, N, iy, iy, false) / (m * m)
  )
}

const range = (from: number, to: number) => Array.from({ length: to - from }, (_, i) => from + i)

/**
 * A permutation test of a statistic of a pooled matrix (kernel or distance): the first n pooled points are one
 * sample, the rest the other. Large values are evidence against the null. Returns the engine's observed value, null
 * values and p-value.
 */
export function matrixPermutationTest(
  s: Stream,
  M: Float64Array,
  n: number,
  stat: (M: Float64Array, N: number, ix: ArrayLike<number>, iy: ArrayLike<number>) => number,
  resamples: number,
) {
  const N = Math.round(Math.sqrt(M.length))
  return permutationTest(s, range(0, n), range(n, N), (ix, iy) => stat(M, N, ix, iy), {
    resamples,
    alternative: 'greater',
  })
}

/** The MMD witness function μ_x(t) − μ_y(t) = mean k(x_i, t) − mean k(y_j, t) at the query points. */
export function witness(x: Points, y: Points, at: Points, sigma: number): number[] {
  const k = rbf({ lengthscale: sigma })
  const q = tensor(at)
  const mean = (s: Points) => {
    const G = toRows(gram(k, q, tensor(s))) as number[][]
    return G.map((row) => row.reduce((a, v) => a + v, 0) / s.length)
  }
  const mx = mean(x)
  const my = mean(y)
  return mx.map((v, i) => v - my[i])
}

/**
 * The classifier two-sample test: train k-nearest neighbours to tell the samples apart on half of each, count correct
 * labels on the other half, and test the count against Binomial(n_test, ½) with the engine's exact binomial test. Also
 * returns the fitted classifier's probability that a point came from the second sample, at the query points.
 */
export function classifierTest(x: Points, y: Points, k: number, at?: Points) {
  const half = (s: Points) => ({ train: s.filter((_, i) => i % 2 === 0), test: s.filter((_, i) => i % 2 === 1) })
  const hx = half(x)
  const hy = half(y)
  const d = x[0].length
  const rows = (s: Points) => fromData(Float64Array.from(s.flat()), [s.length, d])
  const labels = (a: number, b: number) => fromData(Float64Array.from([...Array(a).fill(0), ...Array(b).fill(1)]))
  const model = kNearestNeighbours({ k }).fit(
    dataset(rows([...hx.train, ...hy.train]), labels(hx.train.length, hy.train.length)),
  )
  if (!hasPredictive(model)) throw new Error('k-nearest neighbours should give class probabilities')
  const probability = (s: Points) => {
    const dist = (model as unknown as { predictive(x: Tensor): Distribution }).predictive(rows(s))
    return (toRows(classProbabilities(dist)) as number[][]).map((p) => p[1])
  }
  const px = probability(hx.test)
  const py = probability(hy.test)
  // Ties (probability exactly ½) count as half right, so a classifier that cannot tell scores ½ on average.
  const score = (p: number, label: 0 | 1) => (p === 0.5 ? 0.5 : (p > 0.5 ? 1 : 0) === label ? 1 : 0)
  const correct = px.reduce((a, p) => a + score(p, 0), 0) + py.reduce((a, p) => a + score(p, 1), 0)
  const nTest = px.length + py.length
  const test = binomialTest(Math.round(correct), nTest, { p: 0.5, alternative: 'greater' })
  return { accuracy: correct / nTest, nTest, pValue: test.pValue, test, field: at ? probability(at) : [] }
}

/** The one-dimensional tests the comparison runs, by name. */
export const ONE_D_TESTS = ['Welch t', 'Mann–Whitney', 'Kolmogorov–Smirnov', 'energy', 'MMD', 'classifier'] as const
export type OneDTest = (typeof ONE_D_TESTS)[number]

/** The p-value of every test in ONE_D_TESTS on one pair of one-dimensional samples. */
export function allPValues(s: Stream, x: readonly number[], y: readonly number[], resamples: number) {
  const pooled = asRows([...x, ...y])
  const n = x.length
  const sigma = medianHeuristic(pooled)
  return {
    'Welch t': welchTTest(x, y).pValue,
    'Mann–Whitney': mannWhitneyU(x, y).pValue,
    'Kolmogorov–Smirnov': ksTest(x, y).pValue,
    energy: matrixPermutationTest(child(s, 'energy'), distanceMatrix(pooled), n, energy, resamples).pValue,
    MMD: matrixPermutationTest(child(s, 'mmd'), kernelMatrix(pooled, sigma), n, mmd2, resamples).pValue,
    classifier: classifierTest(asRows(x), asRows(y), 9).pValue,
  } satisfies Record<OneDTest, number>
}
