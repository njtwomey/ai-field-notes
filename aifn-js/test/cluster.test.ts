import { describe, expect, it } from 'vitest'
import {
  agglomerative,
  agglomerativeSteps,
  canonicalLabels,
  cutTree,
  dbscan,
  dendrogram,
  gaussianMixture,
  gaussianMixtureSteps,
  kMedoids,
  kmeans,
  kmeansPlusPlus,
  kmeansSteps,
  linkage,
  meanShift,
  mergeTree,
  miniBatchKMeans,
  optics,
  spectralClustering,
  CORE,
} from 'aifn/cluster'
import { leaves } from 'aifn/graph'
import { stream } from 'aifn/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn/tensor'
import { extend, run, seek, trace } from 'aifn/trace'
import { fixture } from './fixtures'

type Step = { weights: number[]; means: number[][]; covariances: number[][][]; lower_bound: number }
type Fx = {
  x: number[][]
  init: number[][]
  kmeans: { centroids: number[][]; labels: number[]; inertia: number }
  gmm: Record<'full' | 'diagonal' | 'spherical', { weights: number[]; covariances: number[][][]; steps: Step[] }>
  linkage: Record<'single' | 'complete' | 'average' | 'ward', number[][]>
  leaves: number[]
  dbscan: { labels: number[]; core: number[] }
  optics: { ordering: number[]; reachability: number[]; core: number[]; predecessor: number[] }
  xs: number[][]
  spectral: number[]
  meanshift: { centres: number[][]; labels: number[] }
}
const fx = fixture<Fx>('cluster')
const X = tensor(fx.x)

function close(a: Tensor | number[], b: unknown, digits = 8) {
  const got = Array.isArray(a) ? a : toFlat(a)
  const want = (b as number[]).flat(3) as number[]
  expect(got.length).toBe(want.length)
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], digits))
}

const partition = (labels: Tensor | number[]) =>
  Array.from(canonicalLabels(Array.isArray(labels) ? labels : toFlat(labels)))

describe('k-means', () => {
  it('Lloyd from fixed centroids matches scikit-learn', () => {
    const m = kmeans({ k: 3, centroids: tensor(fx.init) }).fit({ x: X })
    close(m.centroids, fx.kmeans.centroids)
    expect(toFlat(m.labels)).toEqual(fx.kmeans.labels)
    expect(m.inertia).toBeCloseTo(fx.kmeans.inertia, 8)
    expect(m.converged).toBe(true)
  })
  it('inertia never rises and the trace protocol holds', () => {
    const alg = kmeansSteps(X, { k: 3 })
    const t = trace(alg, {}, 50, { stream: stream(2), record: { inertia: (s) => s.inertia } })
    const inertia = toFlat(t.series.inertia)
    for (let i = 1; i < inertia.length; i++) expect(inertia[i]).toBeLessThanOrEqual(inertia[i - 1] + 1e-12)
    const opts = {
      centroids: tensor([
        [0, 0],
        [1, 1],
        [2, 2],
      ]),
    }
    expect(toFlat(seek(alg, opts, 2).centroids)).toEqual(toFlat(run(alg, opts, 2).centroids))
    const long = trace(alg, opts, 6, { record: { i: (s) => s.inertia } })
    expect(toFlat(extend(trace(alg, opts, 2, { record: { i: (s) => s.inertia } }), alg, opts, 4).series.i)).toEqual(
      toFlat(long.series.i),
    )
  })
  it('k-means++ is reproducible and its first pick is uniform', () => {
    const a = kmeansPlusPlus(stream(4), X, 3)
    expect(toFlat(kmeansPlusPlus(stream(4), X, 3).indices)).toEqual(toFlat(a.indices))
    const p = toRows(a.probabilities)
    expect(p[0].every((u) => Math.abs(u - 1 / 45) < 1e-15)).toBe(true)
    expect(p[1][toFlat(a.indices)[0]]).toBe(0)
    for (const row of p) expect(row.reduce((u, v) => u + v, 0)).toBeCloseTo(1, 12)
  })
  it('mini-batch k-means and k-medoids find the three groups', () => {
    const mb = miniBatchKMeans({ k: 3, steps: 60, batchSize: 10 }).fit({ x: X }, { stream: stream(1) })
    expect(partition(mb.labels)).toEqual(partition(fx.kmeans.labels))
    const pam = kMedoids({ k: 3 }).fit({ x: X })
    expect(partition(pam.labels)).toEqual(partition(fx.kmeans.labels))
    const costs = toFlat(pam.training.series.cost)
    for (let i = 1; i < costs.length; i++) expect(costs[i]).toBeLessThanOrEqual(costs[i - 1])
  })
})

describe('Gaussian mixtures', () => {
  for (const kind of ['full', 'diagonal', 'spherical'] as const) {
    it(`EM from a fixed start matches scikit-learn (${kind})`, () => {
      const g = fx.gmm[kind]
      const alg = gaussianMixtureSteps(X, { k: 3, covariance: kind, tol: -1 })
      const init = { weights: tensor(g.weights), means: tensor(fx.init), covariances: tensor(g.covariances) }
      const t = trace(alg, init, 3)
      ;[1, 3].forEach((it, j) => {
        const s = t.steps[it]
        close(s.weights, g.steps[j].weights, 7)
        close(s.means, g.steps[j].means, 7)
        close(s.covariances, g.steps[j].covariances, 7)
        expect(t.steps[it - 1].logLikelihood).toBeCloseTo(g.steps[j].lower_bound, 7)
      })
    })
  }
  it('the log-likelihood rises and responsibilities sum to one', () => {
    const m = gaussianMixture({ k: 3 }).fit({ x: X }, { stream: stream(3) })
    const ll = toFlat(m.training.series.logLikelihood)
    for (let i = 1; i < ll.length; i++) expect(ll[i]).toBeGreaterThanOrEqual(ll[i - 1] - 1e-10)
    for (const row of toRows(m.responsibilities(X))) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const draws = m.sampleMixture(stream(9), 500)
    expect(draws.x.shape).toEqual([500, 2])
  })
})

describe('agglomerative clustering', () => {
  for (const method of ['single', 'complete', 'average', 'ward'] as const) {
    it(`${method} linkage matches SciPy`, () => close(linkage(X, method), fx.linkage[method]))
  }
  it('dendrogram leaves match SciPy; cuts give the expected partitions', () => {
    const Z = linkage(X, 'ward')
    expect(toFlat(dendrogram(Z).order)).toEqual(fx.leaves)
    const three = cutTree(Z, { clusters: 3 })
    expect(new Set(toFlat(three)).size).toBe(3)
    const h = toRows(Z)[41][2]
    expect(toFlat(cutTree(Z, { height: h }))).toEqual(toFlat(three))
    const tree = mergeTree(Z)
    expect(leaves(tree).sort((a, b) => a - b)).toEqual(Array.from({ length: 45 }, (_, i) => i))
    expect(tree.nodes[tree.root].size).toBe(45)
    const m = agglomerative({ linkage: 'ward', clusters: 3 }).fit({ x: X })
    expect(toFlat(m.labels)).toEqual(toFlat(three))
    const alg = agglomerativeSteps(X, { linkage: 'average' })
    expect(toFlat(seek(alg, {}, 10).labels)).toEqual(toFlat(run(alg, {}, 10).labels))
  })
})

describe('density clustering', () => {
  it('DBSCAN matches scikit-learn', () => {
    const m = dbscan({ eps: 0.6, minSamples: 4 }).fit({ x: X })
    expect(toFlat(m.labels)).toEqual(fx.dbscan.labels)
    const core = toFlat(m.roles).flatMap((r, i) => (r === CORE ? [i] : []))
    expect(core).toEqual(fx.dbscan.core)
  })
  it('OPTICS matches scikit-learn', () => {
    const m = optics({ minSamples: 4 }).fit({ x: X })
    expect(toFlat(m.ordering)).toEqual(fx.optics.ordering)
    close(m.reachability, fx.optics.reachability)
    close(m.coreDistances, fx.optics.core)
    expect(toFlat(m.predecessor)).toEqual(fx.optics.predecessor)
  })
  it('spectral clustering and mean shift match scikit-learn partitions', () => {
    const s = spectralClustering({ k: 3, affinity: { kind: 'rbf', lengthscale: 1 } }).fit(
      { x: tensor(fx.xs) },
      { stream: stream(0) },
    )
    expect(partition(s.labels)).toEqual(partition(fx.spectral))
    expect(s.components).toBe(1)
    const ms = meanShift({ bandwidth: 1.5 }).fit({ x: X })
    close(ms.centres, fx.meanshift.centres, 2)
    expect(toFlat(ms.labels)).toEqual(fx.meanshift.labels)
  })
})
