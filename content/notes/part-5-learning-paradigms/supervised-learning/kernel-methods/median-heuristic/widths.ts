/**
 * The median-heuristic experiment of the figure, as a pure function so it can be checked outside the page: one of the
 * engine's synthetic data sets, distances over random pairs, three candidate RBF widths from their quantiles, and the
 * five-fold cross-validated accuracy of the engine's SVM at each candidate and along a grid of widths.
 */
import { child, integers, stream } from 'aifn-compute/foundation/random'
import { linspace, tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { rbf } from 'aifn-compute/learning/kernels'
import { accuracy } from 'aifn-compute/learning/metrics'
import { crossValidate, stratifiedKFold } from 'aifn-compute/learning/validate'
import { histogram, quantile } from 'aifn-compute/probability/stats'
import { checkerboard, circles, moons, spirals, xor } from 'aifn-methods/data/synthetic'
import { supportVectorMachine } from 'aifn-methods/learning/kernel-methods'

export type DataName = 'moons' | 'circles' | 'xor' | 'spirals' | 'checkerboard'
export const DATA: { value: DataName; label: string }[] = [
  { value: 'moons', label: 'two moons' },
  { value: 'circles', label: 'two circles' },
  { value: 'xor', label: 'xor' },
  { value: 'spirals', label: 'two spirals' },
  { value: 'checkerboard', label: 'checkerboard' },
]
export const QUANTILES = [0.1, 0.5, 0.9] as const
export const NAMES = ['0.1 quantile', 'median', '0.9 quantile']
export const PAIRS = 1000
export const FOLDS = 5
export const GRID = toFlat(linspace(-1.5, 1, 13)).map((v) => 10 ** v)

/** One of the engine's synthetic two-class data sets, with the noise level applied where the generator has one. */
function makeData(name: DataName, n: number, noise: number, seed: number) {
  const s = child(stream(seed), 'data')
  if (name === 'moons') return moons(s, { n, noise })
  if (name === 'circles') return circles(s, { n, noise, factor: 0.5 })
  if (name === 'xor') return xor(s, { n, kind: 'gaussian', sd: 0.2 + noise })
  if (name === 'spirals') return spirals(s, { n, noise })
  return checkerboard(s, { n })
}

export const BINS = 30

export type Params = { data: DataName; n: number; noise: number; C: number; seed: number }

/** Everything the figure draws, for one setting of its controls. */
export function experiment(p: Params) {
  const d = makeData(p.data, p.n, p.noise, p.seed)
  const labels = d.y
  if (!labels) throw new Error('the synthetic classification sets always have labels')
  const train = dataset(d.x, labels)
  const X = toRows(d.x) as number[][]
  const y = toFlat(labels)
  const n = X.length

  // Distances between random pairs of distinct points.
  const ia = toFlat(integers(child(stream(p.seed), 'pair-a'), n, { shape: [PAIRS] }))
  const ib = toFlat(integers(child(stream(p.seed), 'pair-b'), n, { shape: [PAIRS] }))
  const dist = ia
    .map((a, k) => (a === ib[k] ? NaN : Math.hypot(X[a][0] - X[ib[k]][0], X[a][1] - X[ib[k]][1])))
    .filter((v) => !Number.isNaN(v))
  const q = QUANTILES.map((p) => quantile(dist, p))

  // exp(−λ²r²) with λ = 1/d is the engine's rbf exp(−r²/2ℓ²) with lengthscale ℓ = d/√2.
  const fit = (ell: number) =>
    crossValidate(
      supportVectorMachine({ C: p.C, kernel: rbf({ lengthscale: ell }) }),
      train,
      stratifiedKFold({ k: FOLDS, shuffle: true }),
      [accuracy],
      { stream: child(stream(p.seed), 'cv') },
    )
  const candidates = q.map((v) => {
    const cv = fit(v / Math.SQRT2)
    return { lengthscale: v / Math.SQRT2, accuracy: cv.mean.accuracy, std: cv.std.accuracy }
  })
  const grid = GRID.map((ell) => fit(ell).mean.accuracy)
  const best = candidates.reduce((b, c, i) => (c.accuracy > candidates[b].accuracy ? i : b), 0)

  // The decision function of each candidate, fitted on all the data, on a grid around it.
  const lo = [0, 1].map((j) => Math.min(...X.map((p) => p[j])))
  const hi = [0, 1].map((j) => Math.max(...X.map((p) => p[j])))
  const pad = 0.1 * Math.max(hi[0] - lo[0], hi[1] - lo[1])
  const gx = toFlat(linspace(lo[0] - pad, hi[0] + pad, 41))
  const gy = toFlat(linspace(lo[1] - pad, hi[1] + pad, 41))
  const at = tensor(gy.flatMap((b) => gx.map((a) => [a, b])))
  const fields = candidates.map((c) => {
    const model = supportVectorMachine({ C: p.C, kernel: rbf({ lengthscale: c.lengthscale }) }).fit(train)
    const f = toFlat(model.score(at))
    return gy.map((_, r) => f.slice(r * gx.length, (r + 1) * gx.length))
  })
  const histTop = Math.max(...toFlat(histogram(dist, { bins: BINS }).counts))
  return { X, y, dist, histTop, q, candidates, grid, best, gx, gy, fields }
}
