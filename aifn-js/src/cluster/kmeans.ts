/**
 * k-means and relatives:
 *
 * - `kmeansSteps`, `kmeans`: Lloyd's algorithm (Lloyd, 1957/1982) with k-means++ seeding (Arthur and Vassilvitskii,
 *   2007) and restarts, as scikit-learn's `KMeans(algorithm='lloyd')`.
 * - `kmeansPlusPlus`: the seeding with every pick's D² probabilities exposed.
 * - `miniBatchKMeansSteps`, `miniBatchKMeans`: Sculley's (2010) mini-batch k-means with per-centre learning rates.
 * - `kMedoidsSteps`, `kMedoids`: PAM, BUILD then SWAP (Kaufman and Rousseeuw, 1990, "Finding Groups in Data", ch. 2).
 */

import type { Decides, Estimator, FitOptions, Fitted, Scores, Trained, Transforms } from 'aifn/estimators'
import type { Dataset } from 'aifn/estimators'
import { stream as makeStream, type Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/trace'
import { ints, mat, matrix, nearest, pairwise, sq, values, vec } from './util'

// ── k-means++ ────────────────────────────────────────────────────────────────────────────────────────────────────

/** The result of k-means++ seeding. */
export interface KMeansPlusPlus {
  /** The chosen rows, in the order picked. */
  indices: Tensor
  /** Their coordinates [k, d]. */
  centroids: Tensor
  /** Pick j's sampling probabilities over the rows [k, n]: uniform for the first, D(x)² / Σ D² after. */
  probabilities: Tensor
}

/**
 * k-means++ seeding (Arthur and Vassilvitskii, 2007): the first centre is a uniform row; each next centre is row x with
 * probability D(x)² / Σ D², D the distance to the nearest centre so far. With `trials` > 1 each pick draws that many
 * candidates and keeps the one that lowers the potential most (scikit-learn's greedy variant uses 2 + ⌊log k⌋).
 * Pick j draws from `s.child('pick', j)`.
 */
export function kmeansPlusPlus(s: Stream, x: Tensor, k: number, params: { trials?: number } = {}): KMeansPlusPlus {
  const { n, d, v } = matrix(x, 'kmeansPlusPlus')
  if (!(k >= 1 && k <= n)) throw new Error(`kmeansPlusPlus: k must lie in 1 … ${n}`)
  const trials = params.trials ?? 1
  const indices: number[] = []
  const probs = new Float64Array(k * n)
  const D2 = new Float64Array(n).fill(Infinity)
  const draw = (sub: Stream, weights: Float64Array, total: number): number => {
    let u = sub.uniform() * total
    for (let i = 0; i < n; i++) {
      u -= weights[i]
      if (u < 0) return i
    }
    for (let i = n - 1; i >= 0; i--) if (weights[i] > 0) return i
    return n - 1
  }
  for (let j = 0; j < k; j++) {
    const sub = s.child('pick', j)
    let pick: number
    if (j === 0) {
      probs.fill(1 / n, 0, n)
      pick = sub.int(n)
    } else {
      let total = 0
      for (let i = 0; i < n; i++) total += D2[i]
      for (let i = 0; i < n; i++) probs[j * n + i] = total > 0 ? D2[i] / total : 1 / n
      if (total === 0) pick = sub.int(n)
      else {
        pick = draw(sub, D2, total)
        let bestPotential = Infinity
        for (let t = 0; t < trials; t++) {
          const c = t === 0 ? pick : draw(sub, D2, total)
          let potential = 0
          for (let i = 0; i < n; i++) potential += Math.min(D2[i], sq(v, i, v, c, d))
          if (potential < bestPotential) {
            bestPotential = potential
            pick = c
          }
        }
      }
    }
    indices.push(pick)
    for (let i = 0; i < n; i++) D2[i] = Math.min(D2[i], sq(v, i, v, pick, d))
  }
  const centroids = new Float64Array(k * d)
  indices.forEach((i, j) => centroids.set(v.subarray(i * d, (i + 1) * d), j * d))
  return { indices: ints(indices), centroids: mat(centroids, k, d), probabilities: mat(probs, k, n) }
}

// ── Lloyd ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** One state of Lloyd's algorithm: centroids and the assignment to them. */
export interface KMeansState {
  /** Centroids [k, d]. */
  centroids: Tensor
  /** The nearest centroid of each row [n] (int32). */
  labels: Tensor
  /** Σᵢ ‖xᵢ − c_labelᵢ‖² at these centroids. */
  inertia: number
  /** Rows per cluster [k]. */
  sizes: Tensor
  /** Total squared movement of the centroids in the last update (0 at the start). */
  shift: number
  /** Clusters that were empty in the last update; their centroids stay where they were. */
  empty: number[]
  iteration: number
  /** The assignment did not change (strict convergence) or the shift fell to `tol`. */
  converged: boolean
}

/** How to start Lloyd's algorithm: given centroids [k, d], or a seeding drawn from the stream. */
export interface KMeansInit {
  centroids?: Tensor
  seeding?: 'k-means++' | 'random'
}

function assign(v: Float64Array, n: number, c: Float64Array, k: number, d: number) {
  const labels = new Int32Array(n)
  const sizes = new Float64Array(k)
  let inertia = 0
  for (let i = 0; i < n; i++) {
    const [j, dist] = nearest(v, i, c, k, d)
    labels[i] = j
    sizes[j]++
    inertia += dist
  }
  return { labels, sizes, inertia }
}

/**
 * Lloyd's algorithm as a traceable algorithm on the rows of x [n, d]: each step moves every centroid to the mean of
 * its rows and reassigns every row to its nearest centroid (ties to the lower index). Done when no label changes or
 * the centroids' total squared shift is at most `tol` (default 0). An empty cluster keeps its centroid and is reported
 * in `empty`. `init` takes centroids, or seeds them from the stream (k-means++ by default).
 */
export function kmeansSteps(x: Tensor, params: { k: number; tol?: number }): Algorithm<KMeansInit, KMeansState> {
  const { n, d, v } = matrix(x, 'kmeansSteps')
  const { k, tol = 0 } = params
  return {
    name: 'lloyd',
    init: ({ centroids, seeding = 'k-means++' }, s) => {
      let c: Float64Array
      if (centroids) {
        if (centroids.shape[0] !== k || centroids.shape[1] !== d)
          throw new Error(`kmeansSteps: centroids must be [${k}, ${d}]`)
        c = Float64Array.from(values(centroids))
      } else {
        const st = s ?? makeStream('kmeans')
        if (seeding === 'random') {
          const picks = new Set<number>()
          const sub = st.child('random-seeding')
          while (picks.size < k) picks.add(sub.int(n))
          c = new Float64Array(k * d)
          ;[...picks].forEach((i, j) => c.set(v.subarray(i * d, (i + 1) * d), j * d))
        } else c = Float64Array.from(values(kmeansPlusPlus(st, x, k).centroids))
      }
      const a = assign(v, n, c, k, d)
      return {
        centroids: mat(c, k, d),
        labels: fromData(a.labels, [n]),
        inertia: a.inertia,
        sizes: vec(a.sizes),
        shift: 0,
        empty: [],
        iteration: 0,
        converged: false,
      }
    },
    step: (state) => {
      const old = values(state.centroids)
      const labels = state.labels.data as Int32Array
      const sums = new Float64Array(k * d)
      const counts = new Float64Array(k)
      for (let i = 0; i < n; i++) {
        counts[labels[i]]++
        for (let j = 0; j < d; j++) sums[labels[i] * d + j] += v[i * d + j]
      }
      const c = new Float64Array(k * d)
      const empty: number[] = []
      let shift = 0
      for (let j = 0; j < k; j++) {
        if (counts[j] === 0) empty.push(j)
        for (let t = 0; t < d; t++) {
          c[j * d + t] = counts[j] ? sums[j * d + t] / counts[j] : old[j * d + t]
          shift += (c[j * d + t] - old[j * d + t]) ** 2
        }
      }
      const a = assign(v, n, c, k, d)
      let same = true
      for (let i = 0; i < n; i++) if (a.labels[i] !== labels[i]) same = false
      return {
        centroids: mat(c, k, d),
        labels: fromData(a.labels, [n]),
        inertia: a.inertia,
        sizes: vec(a.sizes),
        shift,
        empty,
        iteration: state.iteration + 1,
        converged: same || shift <= tol,
      }
    },
    done: (state) => state.converged,
  }
}

/** A fitted k-means (or mini-batch k-means) model. */
export interface KMeansModel
  extends Fitted<Tensor, Tensor>, Decides<Tensor, Tensor>, Scores<Tensor>, Transforms<Tensor, Tensor> {
  readonly kind: 'kmeans' | 'mini-batch-kmeans'
  readonly centroids: Tensor
  /** Labels of the training rows. */
  readonly labels: Tensor
  readonly inertia: number
  readonly iterations: number
  readonly converged: boolean
  /** The inertia of every restart (k-means). */
  readonly restarts: Tensor
}

function centroidModel(centroids: Tensor, d: number) {
  const c = values(centroids)
  const k = centroids.shape[0]
  const sqd = (q: Tensor) => {
    const { n: m, v, d: dq } = matrix(q, 'kmeans')
    if (dq !== d) throw new Error(`kmeans: fitted on ${d} features, given ${dq}`)
    const out = new Float64Array(m * k)
    for (let i = 0; i < m; i++) for (let j = 0; j < k; j++) out[i * k + j] = sq(v, i, c, j, d)
    return { out, m }
  }
  return {
    /** Negative squared distances to the centroids [m, k]. */
    forward: (q: Tensor) => {
      const { out, m } = sqd(q)
      return mat(
        out.map((t) => -t),
        m,
        k,
      )
    },
    score: (q: Tensor) => {
      const { out, m } = sqd(q)
      return mat(
        out.map((t) => -t),
        m,
        k,
      )
    },
    /** Euclidean distances to the centroids [m, k]. */
    transform: (q: Tensor) => {
      const { out, m } = sqd(q)
      return mat(out.map(Math.sqrt), m, k)
    },
    decide: (q: Tensor) => {
      const { out, m } = sqd(q)
      const labels = new Int32Array(m)
      for (let i = 0; i < m; i++) for (let j = 1; j < k; j++) if (out[i * k + j] < out[i * k + labels[i]]) labels[i] = j
      return fromData(labels, [m])
    },
  }
}

/**
 * k-means by Lloyd's algorithm: `restarts` runs (default 10) from k-means++ seedings, each from
 * `stream.child('restart', r)`, keeping the lowest inertia; or one run from given `centroids`. `decide` gives the
 * nearest centroid, `transform` the distances to the centroids, `score` their negated squares. The best run is kept in
 * `training`.
 */
export function kmeans(params: {
  k: number
  restarts?: number
  centroids?: Tensor
  seeding?: 'k-means++' | 'random'
  maxIterations?: number
  tol?: number
}): Estimator<Dataset<Tensor>, KMeansModel & Trained<KMeansState>> {
  const { k, restarts = 10, centroids, seeding = 'k-means++', maxIterations = 300, tol = 0 } = params
  return {
    name: 'kmeans',
    params: { k, restarts, seeding, maxIterations, tol },
    fit({ x }, options: FitOptions = {}) {
      const { d } = matrix(x, 'kmeans')
      const alg = kmeansSteps(x, { k, tol })
      const s = options.stream ?? makeStream('kmeans')
      const runs = centroids ? 1 : restarts
      let best: Trace<KMeansState> | null = null
      const inertias: number[] = []
      for (let r = 0; r < runs; r++) {
        const t = trace(alg, { centroids, seeding }, maxIterations, {
          stream: s.child('restart', r),
          every: options.trace?.every ?? 1,
          checkpointEvery: options.trace?.checkpointEvery,
          record: {
            inertia: (st) => st.inertia,
            ...(options.trace?.record as Record<string, (st: KMeansState, i: number) => number> | undefined),
          },
        })
        const final = t.steps[t.steps.length - 1]
        inertias.push(final.inertia)
        if (!best || final.inertia < best.steps[best.steps.length - 1].inertia) best = t
      }
      const final = best!.steps[best!.steps.length - 1]
      return {
        kind: 'kmeans',
        centroids: final.centroids,
        labels: final.labels,
        inertia: final.inertia,
        iterations: final.iteration,
        converged: final.converged,
        restarts: vec(inertias),
        training: best!,
        ...centroidModel(final.centroids, d),
      }
    },
  }
}

// ── Mini-batch k-means ───────────────────────────────────────────────────────────────────────────────────────────

/** One state of mini-batch k-means. */
export interface MiniBatchKMeansState {
  centroids: Tensor
  /** How many rows each centroid has absorbed so far [k] (its learning rate is 1/count). */
  counts: Tensor
  /** The rows of the latest batch (int32). */
  batch: Tensor
  /** The inertia of the whole data at these centroids. */
  inertia: number
  iteration: number
  stream: Stream
}

/**
 * Mini-batch k-means (Sculley, 2010, Algorithm 1): each step draws `batchSize` rows uniformly with replacement (from
 * `stream.child(t)`), assigns them to their nearest centroids, and moves each centroid towards each of its rows by the
 * per-centre rate 1/count. `init` takes centroids or seeds them by k-means++ from the stream.
 */
export function miniBatchKMeansSteps(
  x: Tensor,
  params: { k: number; batchSize?: number },
): Algorithm<KMeansInit, MiniBatchKMeansState> {
  const { n, d, v } = matrix(x, 'miniBatchKMeansSteps')
  const { k, batchSize = Math.min(n, 64) } = params
  return {
    name: 'mini-batch-kmeans',
    init: ({ centroids }, s) => {
      const st = s ?? makeStream('mini-batch-kmeans')
      const c = centroids
        ? Float64Array.from(values(centroids))
        : Float64Array.from(values(kmeansPlusPlus(st.child('seeding'), x, k).centroids))
      return {
        centroids: mat(c, k, d),
        counts: vec(new Float64Array(k)),
        batch: ints([]),
        inertia: assign(v, n, c, k, d).inertia,
        iteration: 0,
        stream: st,
      }
    },
    step: (state) => {
      const c = Float64Array.from(values(state.centroids))
      const counts = Float64Array.from(values(state.counts))
      const sub = state.stream.child(state.iteration + 1)
      const batch = Int32Array.from({ length: batchSize }, () => sub.int(n))
      const nearestOf = Int32Array.from(batch, (i) => nearest(v, i, c, k, d)[0])
      batch.forEach((i, b) => {
        const j = nearestOf[b]
        counts[j]++
        const eta = 1 / counts[j]
        for (let t = 0; t < d; t++) c[j * d + t] = (1 - eta) * c[j * d + t] + eta * v[i * d + t]
      })
      return {
        centroids: mat(c, k, d),
        counts: vec(counts),
        batch: fromData(batch, [batchSize]),
        inertia: assign(v, n, c, k, d).inertia,
        iteration: state.iteration + 1,
        stream: state.stream,
      }
    },
  }
}

/** Mini-batch k-means for `steps` batches (default 100). */
export function miniBatchKMeans(params: {
  k: number
  batchSize?: number
  steps?: number
  centroids?: Tensor
}): Estimator<Dataset<Tensor>, KMeansModel & Trained<MiniBatchKMeansState>> {
  const { k, batchSize, steps = 100, centroids } = params
  return {
    name: 'mini-batch-kmeans',
    params: { k, batchSize, steps },
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'miniBatchKMeans')
      const training = trace(miniBatchKMeansSteps(x, { k, batchSize }), { centroids }, steps, {
        stream: options.stream,
        every: options.trace?.every ?? 1,
        record: { inertia: (s) => s.inertia },
      })
      const final = training.steps[training.steps.length - 1]
      const a = assign(v, n, values(final.centroids), k, d)
      return {
        kind: 'mini-batch-kmeans',
        centroids: final.centroids,
        labels: fromData(a.labels, [n]),
        inertia: a.inertia,
        iterations: final.iteration,
        converged: false,
        restarts: vec([a.inertia]),
        training,
        ...centroidModel(final.centroids, d),
      }
    },
  }
}

// ── k-medoids (PAM) ──────────────────────────────────────────────────────────────────────────────────────────────

/** One PAM state. */
export interface KMedoidsState {
  /** The medoids (row indices, int32 [k]). */
  medoids: Tensor
  /** The nearest medoid's position in `medoids` for each row [n]. */
  labels: Tensor
  /** Σᵢ distance to the nearest medoid. */
  cost: number
  /** The swap made in this step, [medoid out, row in], or null. */
  swap: [number, number] | null
  iteration: number
  converged: boolean
}

function medoidCost(D: Float64Array, n: number, medoids: readonly number[]) {
  const labels = new Int32Array(n)
  let cost = 0
  for (let i = 0; i < n; i++) {
    let best = Infinity
    medoids.forEach((m, j) => {
      if (D[i * n + m] < best) {
        best = D[i * n + m]
        labels[i] = j
      }
    })
    cost += best
  }
  return { labels, cost }
}

/**
 * PAM as a traceable algorithm on a distance matrix [n, n]: `init` runs BUILD (greedily add the medoid that lowers
 * the cost most) unless medoids are given; each step makes the best cost-lowering swap of a medoid with a non-medoid,
 * and it is done when no swap lowers the cost.
 */
export function kMedoidsSteps(
  distances: Tensor,
  params: { k: number },
): Algorithm<{ medoids?: readonly number[] }, KMedoidsState> {
  const [n, n2] = distances.shape
  if (n !== n2) throw new Error('kMedoidsSteps: distances must be [n, n]')
  const D = values(distances)
  const { k } = params
  const make = (
    medoids: number[],
    iteration: number,
    swap: [number, number] | null,
    converged: boolean,
  ): KMedoidsState => {
    const { labels, cost } = medoidCost(D, n, medoids)
    return { medoids: ints(medoids), labels: fromData(labels, [n]), cost, swap, iteration, converged }
  }
  return {
    name: 'pam',
    init: ({ medoids }) => {
      if (medoids) return make([...medoids], 0, null, false)
      const chosen: number[] = []
      for (let j = 0; j < k; j++) {
        let best = -1
        let bestCost = Infinity
        for (let c = 0; c < n; c++) {
          if (chosen.includes(c)) continue
          const cost = medoidCost(D, n, [...chosen, c]).cost
          if (cost < bestCost) {
            bestCost = cost
            best = c
          }
        }
        chosen.push(best)
      }
      return make(chosen, 0, null, false)
    },
    step: (state) => {
      const medoids = Array.from(state.medoids.data as Int32Array)
      let best: [number, number] | null = null
      let bestCost = state.cost
      for (let j = 0; j < k; j++) {
        for (let c = 0; c < n; c++) {
          if (medoids.includes(c)) continue
          const trial = medoids.slice()
          trial[j] = c
          const cost = medoidCost(D, n, trial).cost
          if (cost < bestCost - 1e-12 * Math.max(1, Math.abs(bestCost))) {
            bestCost = cost
            best = [j, c]
          }
        }
      }
      if (!best) return { ...state, iteration: state.iteration + 1, swap: null, converged: true }
      const out = medoids[best[0]]
      medoids[best[0]] = best[1]
      return make(medoids, state.iteration + 1, [out, best[1]], false)
    },
    done: (state) => state.converged,
  }
}

/** A fitted k-medoids model. */
export interface KMedoidsModel extends Decides<Tensor, Tensor>, Trained<KMedoidsState> {
  readonly kind: 'k-medoids'
  /** Medoid row indices [k] and their coordinates [k, d]. */
  readonly medoids: Tensor
  readonly centres: Tensor
  readonly labels: Tensor
  readonly cost: number
}

/** k-medoids by PAM on Euclidean distances between the rows of x. `decide` assigns new rows to the nearest medoid. */
export function kMedoids(params: { k: number; maxIterations?: number }): Estimator<Dataset<Tensor>, KMedoidsModel> {
  const { k, maxIterations = 100 } = params
  return {
    name: 'k-medoids',
    params,
    fit({ x }, options: FitOptions = {}) {
      const { n, d, v } = matrix(x, 'kMedoids')
      const training = trace(kMedoidsSteps(fromData(pairwise(v, n, d), [n, n]), { k }), {}, maxIterations, {
        every: options.trace?.every ?? 1,
        record: { cost: (s) => s.cost },
      })
      const final = training.steps[training.steps.length - 1]
      const m = final.medoids.data as Int32Array
      const centres = new Float64Array(k * d)
      m.forEach((i, j) => centres.set(v.subarray(i * d, (i + 1) * d), j * d))
      return {
        kind: 'k-medoids',
        medoids: final.medoids,
        centres: mat(centres, k, d),
        labels: final.labels,
        cost: final.cost,
        training,
        decide: centroidModel(mat(centres, k, d), d).decide,
      }
    },
  }
}
