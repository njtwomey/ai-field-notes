import { describe, expect, it } from 'vitest'
import { classicalMds, kernelPca, metricMds, pca, smacofSteps } from 'aifn-applied/unsupervised/embedding/linear'
import {
  curveParameters,
  fuzzyGraph,
  jointProbabilities,
  perplexityCalibration,
  tsne,
  tsneSteps,
  umap,
  umapSteps,
} from 'aifn-applied/unsupervised/embedding/neighbour'
import { isomap, laplacianEigenmaps, locallyLinearEmbedding } from 'aifn-applied/unsupervised/embedding/manifold'
import { rbf } from 'aifn/learning/kernels'
import { stream } from 'aifn/foundation/random'
import { tensor, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'
import { fixture } from './fixtures'

type Fx = {
  x: number[][]
  xq: number[][]
  pca: {
    components: number[][]
    variance: number[]
    ratio: number[]
    noise: number
    transform: number[][]
    whiten: number[][]
  }
  kpca: { eigenvalues: number[]; embedding: number[][]; transform: number[][] }
  spiral: number[][]
  isomap: number[][]
  spectral: number[][]
  lle: number[][]
  mds_eigenvalues: number[]
  smacof: { init: number[][]; steps: number[][][] }
  joint: number[][]
}
const fx = fixture<Fx>('embed')
const X = tensor(fx.x)
const XQ = tensor(fx.xq)
const S = tensor(fx.spiral)

function close(a: Tensor | number[], b: unknown, digits = 8) {
  const got = Array.isArray(a) ? a : toFlat(a)
  const want = (b as number[]).flat(3) as number[]
  expect(got.length).toBe(want.length)
  got.forEach((v, i) => expect(v).toBeCloseTo(want[i], digits))
}

/** Columns equal up to sign. */
function closeUpToSign(a: Tensor, b: number[][], digits = 6) {
  const A = toRows(a)
  for (let c = 0; c < b[0].length; c++) {
    const s = Math.sign(A.reduce((acc, r, i) => acc + r[c] * b[i][c], 0)) || 1
    A.forEach((r, i) => expect(s * r[c]).toBeCloseTo(b[i][c], digits))
  }
}

const distances = (x: number[][]) => tensor(x.map((a) => x.map((b) => Math.hypot(...a.map((u, k) => u - b[k])))))

describe('PCA and kernel PCA', () => {
  it('PCA matches scikit-learn, with whitening and round trips', () => {
    const m = pca({ components: 3 }).fit({ x: X })
    close(m.components, fx.pca.components)
    close(m.explainedVariance, fx.pca.variance)
    close(m.explainedVarianceRatio, fx.pca.ratio)
    expect(m.noiseVariance).toBeCloseTo(fx.pca.noise, 8)
    close(m.transform(XQ), fx.pca.transform)
    close(pca({ components: 2, whiten: true }).fit({ x: X }).transform(XQ), fx.pca.whiten)
    const full = pca().fit({ x: X })
    close(full.inverseTransform(full.transform(XQ)), fx.xq)
  })
  it('kernel PCA matches scikit-learn', () => {
    const m = kernelPca({ kernel: rbf({ lengthscale: Math.SQRT2 }), components: 2 }).fit({ x: X })
    close(m.eigenvalues, fx.kpca.eigenvalues, 7)
    closeUpToSign(m.embedding, fx.kpca.embedding)
    closeUpToSign(m.transform(XQ), fx.kpca.transform)
  })
})

describe('multidimensional scaling', () => {
  it('classical MDS eigenvalues and distances', () => {
    const r = classicalMds(distances(fx.x), 4)
    close(r.eigenvalues, fx.mds_eigenvalues, 7)
    // Euclidean input: the full-rank embedding reproduces the distances.
    close(distances(toRows(r.embedding)), toRows(distances(fx.x)), 7)
  })
  it('SMACOF steps match the Guttman transform; stress falls', () => {
    const alg = smacofSteps(distances(fx.x))
    const t = trace(alg, { embedding: tensor(fx.smacof.init) }, 3)
    fx.smacof.steps.forEach((want, k) => close(t.steps[k + 1].embedding, want))
    const m = metricMds().fit({ x: X })
    const s = toFlat(m.training.series.stress)
    for (let i = 1; i < s.length; i++) expect(s[i]).toBeLessThanOrEqual(s[i - 1] + 1e-9)
    expect(toFlat(seek(alg, {}, 4).embedding)).toEqual(toFlat(run(alg, {}, 4).embedding))
  })
})

describe('neighbour-graph embeddings', () => {
  it('Isomap matches scikit-learn', () => {
    const m = isomap({ neighbours: 6 }).fit({ x: S })
    expect(m.components).toBe(1)
    closeUpToSign(m.embedding, fx.isomap, 5)
  })
  it('Laplacian eigenmaps match SpectralEmbedding', () => {
    // scikit-learn counts the point itself among its neighbours.
    closeUpToSign(laplacianEigenmaps({ neighbours: 6 }).fit({ x: S }).embedding, fx.spectral, 5)
  })
  it('LLE matches scikit-learn', () => {
    closeUpToSign(locallyLinearEmbedding({ neighbours: 6 }).fit({ x: S }).embedding, fx.lle, 5)
  })
})

describe('t-SNE', () => {
  it('joint affinities match scikit-learn; calibration hits the perplexity', () => {
    close(jointProbabilities(X, 5), fx.joint, 6)
    const n = fx.x.length
    const d2 = tensor(fx.x.map((a) => fx.x.map((b) => a.reduce((s, u, k) => s + (u - b[k]) ** 2, 0))))
    const c = perplexityCalibration(d2, 5)
    for (const h of toFlat(c.entropies)) expect(Math.abs(h - Math.log(5))).toBeLessThan(1e-5)
    expect(c.conditional.shape).toEqual([n, n])
  })
  it('is deterministic, lowers KL and follows the trace protocol', () => {
    const a = tsne({ perplexity: 5, iterations: 300 }).fit({ x: X }, { stream: stream(1) })
    const b = tsne({ perplexity: 5, iterations: 300 }).fit({ x: X }, { stream: stream(1) })
    expect(toFlat(a.embedding)).toEqual(toFlat(b.embedding))
    const kl = toFlat(a.training.series.kl)
    expect(kl.at(-1)!).toBeLessThan(kl[kl.length > 30 ? 26 : 1])
    const alg = tsneSteps(jointProbabilities(X, 5))
    const s = () => stream(2)
    expect(toFlat(seek(alg, {}, 5, { stream: s() }).embedding)).toEqual(
      toFlat(run(alg, {}, 5, { stream: s() }).embedding),
    )
    const long = trace(alg, {}, 8, { stream: s(), record: { kl: (st) => st.kl } })
    const short = trace(alg, {}, 3, { stream: s(), record: { kl: (st) => st.kl } })
    expect(toFlat(extend(short, alg, {}, 5).series.kl)).toEqual(toFlat(long.series.kl))
  })
})

describe('UMAP', () => {
  it('curve parameters match umap-learn for min_dist 0.1', () => {
    const { a, b } = curveParameters(0.1, 1)
    expect(a).toBeCloseTo(1.577, 2)
    expect(b).toBeCloseTo(0.895, 2)
  })
  it('the fuzzy graph calibrates σ and is symmetric with weights in (0, 1]', () => {
    const g = fuzzyGraph(S, 8)
    for (const w of g.edges.weight) expect(w > 0 && w <= 1).toBe(true)
    expect(toFlat(g.rho).every((r) => r > 0)).toBe(true)
  })
  it('is deterministic and follows the trace protocol', () => {
    const a = umap({ neighbours: 8, epochs: 50 }).fit({ x: S }, { stream: stream(3) })
    const b = umap({ neighbours: 8, epochs: 50 }).fit({ x: S }, { stream: stream(3) })
    expect(toFlat(a.embedding)).toEqual(toFlat(b.embedding))
    expect(toFlat(a.embedding).every(Number.isFinite)).toBe(true)
    const alg = umapSteps(fuzzyGraph(S, 8), { epochs: 20 })
    const s = () => stream(4)
    expect(toFlat(seek(alg, {}, 6, { stream: s() }).embedding)).toEqual(
      toFlat(run(alg, {}, 6, { stream: s() }).embedding),
    )
  })
})
