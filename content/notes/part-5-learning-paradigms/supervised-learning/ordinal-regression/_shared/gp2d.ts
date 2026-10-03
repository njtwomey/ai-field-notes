/**
 * Gaussian-process ordinal regression on the shared 2-D data: the Laplace fit of laplace.ts with a squared-exponential
 * kernel, on a class-stratified subsample, with thresholds fixed at unit spacing around zero.
 */
import { argmaxIndex, specKey, type DataSpec, type Dataset, type Fitted, type Point } from './ordinal'
import { fitGpOrdinalWith, type GpOrdinalFit } from './laplace'

/** At most this many training points enter the GP: the Laplace fit costs O(n³) per Newton step. */
export const GP_CAP = 150

/** The first ⌊cap/K⌋ points of each class (the points of a class are already in random order). */
export function stratified(d: Dataset, cap = GP_CAP): { x: Point[]; y: number[] } {
  const per = Math.max(1, Math.floor(cap / d.k))
  const seen = Array(d.k).fill(0)
  const x: Point[] = []
  const y: number[] = []
  d.train.x.forEach((p, i) => {
    const c = d.train.y[i]
    if (seen[c]++ < per) {
      x.push(p)
      y.push(c)
    }
  })
  return { x, y }
}

/** Thresholds b_1 < … < b_{K−1} one unit apart and centred on zero; the kernel amplitude K/2 lets f reach them all. */
export const gpThresholds = (k: number) => Array.from({ length: k - 1 }, (_, j) => j - (k - 2) / 2)

const rbf = (lengthscale: number, amplitude: number) => (u: Point, v: Point) =>
  amplitude * amplitude * Math.exp(-((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2) / (2 * lengthscale * lengthscale))

export type Gp2dFit = { fitted: Fitted; logEvidence: number; n: number }

const cache = new Map<string, Gp2dFit>()
/** The last Newton iterate per dataset: the inputs are the same when only the lengthscale or noise changes. */
const warm = new Map<string, number[]>()

export function fitGp2d(d: Dataset, spec: DataSpec, lengthscale: number, sigma: number): Gp2dFit {
  const key = `${specKey(spec)}/${lengthscale}/${sigma}`
  const hit = cache.get(key)
  if (hit) return hit
  const { x, y } = stratified(d)
  const gp: GpOrdinalFit<Point> = fitGpOrdinalWith(
    x,
    y,
    gpThresholds(d.k),
    rbf(lengthscale, d.k / 2),
    sigma,
    warm.get(specKey(spec)),
  )
  warm.set(specKey(spec), gp.a)
  // The map asks for the decision and the probabilities at the same point in turn; keep the last answer.
  let last: Point | null = null
  let lastProbs: number[] = []
  const probs = (p: Point) => {
    if (!last || last[0] !== p[0] || last[1] !== p[1]) {
      last = p
      lastProbs = gp.probs(p)
    }
    return lastProbs
  }
  const out = {
    fitted: { probs, predict: (p: Point) => argmaxIndex(probs(p)) },
    logEvidence: gp.logEvidence,
    n: x.length,
  }
  if (cache.size > 32) cache.delete(cache.keys().next().value as string)
  cache.set(key, out)
  return out
}

/** Lengthscales tried by the evidence fit. */
export const LENGTHSCALES = [0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5]
