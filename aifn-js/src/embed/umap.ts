/**
 * UMAP, simplified but faithful to McInnes, Healy and Melville (2018, "UMAP: Uniform Manifold Approximation and
 * Projection for dimension reduction", arXiv:1802.03426) and umap-learn's defaults: exact k-nearest neighbours, a
 * fuzzy simplicial set (local connectivity ρᵢ, bandwidths σᵢ with Σⱼ exp(−(dᵢⱼ − ρᵢ)/σᵢ) = log₂ k, fuzzy union
 * w + wᵀ − w∘wᵀ), the output curve 1/(1 + a d^(2b)) fitted to `minDist` and `spread`, and a stochastic layout: each
 * epoch samples edges by weight (edge e every 1/wₑ·max w epochs), pulls their ends together and pushes each end away
 * from `negativeSamples` random points, with a learning rate falling linearly to 0.
 *
 * Simplifications: exact neighbours (no nearest-neighbour descent), and a Laplacian-eigenmap or random start.
 */

import type { Dataset, Estimator, FitOptions, Trained } from 'aifn/estimators'
import { eigh } from 'aifn/linalg'
import { stream as makeStream, type Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import { trace, type Algorithm } from 'aifn/trace'
import { mat, matrix, squaredDistances, values, vec } from './util'

/** The fuzzy graph of the data: per-point ρ and σ, and the symmetric membership strengths. */
export interface FuzzyGraph {
  /** Distance to the nearest neighbour (local connectivity 1) [n]. */
  rho: Tensor
  /** Bandwidth found by bisection [n]. */
  sigma: Tensor
  /** The k nearest neighbours of each point, itself first [n, k] (int32). */
  neighbours: Tensor
  /** Edges (i < j) with their membership strength after the fuzzy union. */
  edges: { from: Int32Array; to: Int32Array; weight: Float64Array }
}

/**
 * The fuzzy simplicial set of the rows of x with `neighbours` k (default 15, itself included, as umap-learn):
 * wᵢⱼ = exp(−max(0, dᵢⱼ − ρᵢ)/σᵢ), then w + wᵀ − w∘wᵀ.
 */
export function fuzzyGraph(x: Tensor, neighbours = 15): FuzzyGraph {
  const { n, d, v } = matrix(x, 'fuzzyGraph')
  const k = Math.min(neighbours, n)
  const D = Float64Array.from(squaredDistances(v, n, d), Math.sqrt)
  const nb = new Int32Array(n * k)
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j).sort((a, b) =>
      a === i ? -1 : b === i ? 1 : D[i * n + a] - D[i * n + b] || a - b,
    )
    for (let r = 0; r < k; r++) nb[i * k + r] = order[r]
  }
  const target = Math.log2(k)
  const rho = new Float64Array(n)
  const sigma = new Float64Array(n)
  const W = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    let first = 0
    for (let r = 1; r < k; r++) {
      const t = D[i * n + nb[i * k + r]]
      if (t > 0) {
        first = t
        break
      }
    }
    rho[i] = first
    let lo = 0
    let hi = Infinity
    let s = 1
    for (let it = 0; it < 64; it++) {
      let psum = 0
      for (let r = 1; r < k; r++) psum += Math.exp(-Math.max(0, D[i * n + nb[i * k + r]] - rho[i]) / s)
      if (Math.abs(psum - target) < 1e-5) break
      if (psum > target) {
        hi = s
        s = (lo + hi) / 2
      } else {
        lo = s
        s = hi === Infinity ? s * 2 : (lo + hi) / 2
      }
    }
    // umap-learn floors σ at 10⁻³ of the mean neighbour distance.
    let mean = 0
    for (let r = 1; r < k; r++) mean += D[i * n + nb[i * k + r]] / (k - 1)
    sigma[i] = Math.max(s, 1e-3 * mean)
    for (let r = 1; r < k; r++) {
      const j = nb[i * k + r]
      W.set(i * n + j, Math.exp(-Math.max(0, D[i * n + j] - rho[i]) / sigma[i]))
    }
  }
  const from: number[] = []
  const to: number[] = []
  const weight: number[] = []
  const seen = new Set<number>()
  for (const key of [...W.keys()].sort((a, b) => a - b)) {
    const i = Math.floor(key / n)
    const j = key % n
    const a = Math.min(i, j)
    const b = Math.max(i, j)
    if (seen.has(a * n + b)) continue
    seen.add(a * n + b)
    const wij = W.get(i * n + j) ?? 0
    const wji = W.get(j * n + i) ?? 0
    from.push(a)
    to.push(b)
    weight.push(wij + wji - wij * wji)
  }
  return {
    rho: vec(rho),
    sigma: vec(sigma),
    neighbours: fromData(nb, [n, k]),
    edges: { from: Int32Array.from(from), to: Int32Array.from(to), weight: Float64Array.from(weight) },
  }
}

/**
 * The output curve's parameters: a and b minimising the squared error of 1/(1 + a x^(2b)) against 1 for x < minDist
 * and exp(−(x − minDist)/spread) beyond, on 300 points of [0, 3·spread] (umap-learn's `find_ab_params`), by
 * Gauss–Newton with step halving.
 */
export function curveParameters(minDist = 0.1, spread = 1): { a: number; b: number } {
  const xs = Array.from({ length: 300 }, (_, i) => (3 * spread * i) / 299)
  const ys = xs.map((x) => (x < minDist ? 1 : Math.exp(-(x - minDist) / spread)))
  const loss = (a: number, b: number) => xs.reduce((s, x, i) => s + (1 / (1 + a * x ** (2 * b)) - ys[i]) ** 2, 0)
  let a = 1
  let b = 1
  for (let it = 0; it < 200; it++) {
    let JtJ00 = 0
    let JtJ01 = 0
    let JtJ11 = 0
    let g0 = 0
    let g1 = 0
    xs.forEach((x, i) => {
      if (x === 0) return
      const p = x ** (2 * b)
      const f = 1 / (1 + a * p)
      const r = f - ys[i]
      const da = -p * f * f
      const db = -a * p * 2 * Math.log(x) * f * f
      JtJ00 += da * da
      JtJ01 += da * db
      JtJ11 += db * db
      g0 += da * r
      g1 += db * r
    })
    const det = JtJ00 * JtJ11 - JtJ01 * JtJ01
    if (!(Math.abs(det) > 0)) break
    const s0 = (JtJ11 * g0 - JtJ01 * g1) / det
    const s1 = (JtJ00 * g1 - JtJ01 * g0) / det
    const before = loss(a, b)
    let t = 1
    while (t > 1e-8 && !(a - t * s0 > 0 && b - t * s1 > 0 && loss(a - t * s0, b - t * s1) <= before)) t /= 2
    if (t <= 1e-8) break
    a -= t * s0
    b -= t * s1
    if (Math.abs(t * s0) < 1e-12 && Math.abs(t * s1) < 1e-12) break
  }
  return { a, b }
}

/** One epoch of UMAP's layout. */
export interface UmapState {
  embedding: Tensor
  epoch: number
  /** The learning rate used in the epoch that produced this state. */
  alpha: number
  /** Edge samples taken in that epoch. */
  samples: number
  stream: Stream
}

/**
 * UMAP's stochastic layout as a traceable algorithm (one step = one epoch; `epochs` default 200). Epoch e uses the
 * substream `stream.child(e)`. `init` takes an embedding, or a spectral start (default: the Laplacian eigenmap of the
 * fuzzy graph, scaled to [0, 10]) or a uniform random one in [−10, 10]².
 */
export function umapSteps(
  graph: FuzzyGraph,
  params: {
    dims?: number
    epochs?: number
    minDist?: number
    spread?: number
    negativeSamples?: number
    learningRate?: number
  } = {},
): Algorithm<{ embedding?: Tensor; start?: 'spectral' | 'random' }, UmapState> {
  const { dims = 2, epochs = 200, minDist = 0.1, spread = 1, negativeSamples = 5, learningRate = 1 } = params
  const { a, b } = curveParameters(minDist, spread)
  const n = graph.rho.shape[0]
  const { from, to, weight } = graph.edges
  let wmax = 0
  for (const w of weight) wmax = Math.max(wmax, w)
  // Edges below wmax/epochs are never sampled (umap-learn drops them).
  const period = Float64Array.from(weight, (w) => (w >= wmax / epochs ? wmax / w : Infinity))
  const clip = (g: number) => Math.max(-4, Math.min(4, g))
  const spectralStart = (): Float64Array => {
    const W = new Float64Array(n * n)
    for (let e = 0; e < from.length; e++) W[from[e] * n + to[e]] = W[to[e] * n + from[e]] = weight[e]
    const deg = new Float64Array(n)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += W[i * n + j]
    const M = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) M[i * n + j] = deg[i] && deg[j] ? W[i * n + j] / Math.sqrt(deg[i] * deg[j]) : 0
    const V = eigh(fromData(M, [n, n])).vectors.data as Float64Array
    const Y = new Float64Array(n * dims)
    for (let c = 0; c < dims; c++) {
      let lo = Infinity
      let hi = -Infinity
      for (let i = 0; i < n; i++) {
        const u = V[i * n + c + 1]
        lo = Math.min(lo, u)
        hi = Math.max(hi, u)
      }
      for (let i = 0; i < n; i++) Y[i * dims + c] = hi > lo ? (10 * (V[i * n + c + 1] - lo)) / (hi - lo) : 0
    }
    return Y
  }
  return {
    name: 'umap-layout',
    init: ({ embedding, start = 'spectral' }, s) => {
      const st = s ?? makeStream('umap')
      let Y: Float64Array
      if (embedding) Y = Float64Array.from(values(embedding))
      else if (start === 'random' || n <= dims + 1) {
        const r = st.child('init')
        Y = Float64Array.from({ length: n * dims }, () => 20 * r.uniform() - 10)
      } else Y = spectralStart()
      return { embedding: mat(Y, n, dims), epoch: 0, alpha: learningRate, samples: 0, stream: st }
    },
    step: (state) => {
      const Y = Float64Array.from(values(state.embedding))
      const e = state.epoch + 1
      const alpha = learningRate * (1 - state.epoch / epochs)
      const r = state.stream.child(e)
      let samples = 0
      for (let k = 0; k < from.length; k++) {
        // Edge k is sampled in the epochs e where ⌊e / period⌋ increases.
        if (!Number.isFinite(period[k]) || Math.floor(e / period[k]) === Math.floor((e - 1) / period[k])) continue
        samples++
        const i = from[k]
        const j = to[k]
        let d2 = 0
        for (let c = 0; c < dims; c++) d2 += (Y[i * dims + c] - Y[j * dims + c]) ** 2
        const attract = d2 > 0 ? (-2 * a * b * d2 ** (b - 1)) / (1 + a * d2 ** b) : 0
        for (let c = 0; c < dims; c++) {
          const g = clip(attract * (Y[i * dims + c] - Y[j * dims + c]))
          Y[i * dims + c] += alpha * g
          Y[j * dims + c] -= alpha * g
        }
        for (let s = 0; s < negativeSamples; s++) {
          const m = r.int(n)
          if (m === i) continue
          let q2 = 0
          for (let c = 0; c < dims; c++) q2 += (Y[i * dims + c] - Y[m * dims + c]) ** 2
          const repel = (2 * b) / ((0.001 + q2) * (1 + a * q2 ** b))
          for (let c = 0; c < dims; c++)
            Y[i * dims + c] += alpha * (q2 > 0 ? clip(repel * (Y[i * dims + c] - Y[m * dims + c])) : 4)
        }
      }
      return { embedding: mat(Y, n, dims), epoch: e, alpha, samples, stream: state.stream }
    },
    done: (state) => state.epoch >= epochs,
  }
}

/** A fitted UMAP embedding. */
export interface UmapModel extends Trained<UmapState> {
  readonly kind: 'umap'
  readonly embedding: Tensor
  readonly graph: FuzzyGraph
  readonly a: number
  readonly b: number
}

/** UMAP of the rows of x (see `fuzzyGraph`, `umapSteps`). */
export function umap(
  params: {
    neighbours?: number
    dims?: number
    epochs?: number
    minDist?: number
    spread?: number
    negativeSamples?: number
    start?: 'spectral' | 'random'
  } = {},
): Estimator<Dataset<Tensor>, UmapModel> {
  const { neighbours = 15, epochs = 200, start = 'spectral', minDist = 0.1, spread = 1, ...rest } = params
  return {
    name: 'umap',
    params: { neighbours, epochs, start, minDist, spread, ...rest },
    fit({ x }, options: FitOptions = {}) {
      const graph = fuzzyGraph(x, neighbours)
      const training = trace(umapSteps(graph, { epochs, minDist, spread, ...rest }), { start }, epochs, {
        stream: options.stream,
        every: options.trace?.every ?? 5,
      })
      const final = training.steps[training.steps.length - 1]
      const { a, b } = curveParameters(minDist, spread)
      return { kind: 'umap', embedding: final.embedding, graph, a, b, training }
    },
  }
}
