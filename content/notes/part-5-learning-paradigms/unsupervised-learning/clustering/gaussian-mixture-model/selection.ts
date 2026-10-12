/**
 * Best-of-several EM fits of a diagonal Gaussian mixture for k = 1 … K_MAX on the first n points of `blobData`, with
 * the aifn-engine's `gaussianMixture`. Fits are made one at a time (`fitNext`), so a figure can spread them over
 * several frames, and kept, so changing the restart budget or returning to an n refits nothing already fitted.
 */
import { gaussianMixture, type GaussianMixtureModel } from 'aifn-methods/unsupervised/clustering'
import { child, stream } from 'aifn-compute/foundation/random'
import { dataset } from 'aifn-compute/learning/estimators'
import { fromData, toArray, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Vec2 as Point } from 'aifn-compute/numerics/linalg'
import { blobData } from './data'

export const K_MAX = 8

/**
 * Added to every variance in each M-step. It keeps a component from collapsing onto one point, where the likelihood
 * diverges.
 */
const REGULARISATION = 1e-3

export type KFit = {
  k: number
  /** The deviance −2 log L of the best fit: −2n times its mean log-likelihood. */
  deviance: number
  /** The best fit's component means and 2 × 2 covariances. */
  means: Point[]
  covariances: number[][][]
  model: GaussianMixtureModel
}

const ROOT = stream('gaussian-mixture-model/model-selection')

/** The first n points as an n × 2 tensor, one per n. */
const inputs = new Map<number, Tensor>()
const firstPoints = (n: number) => {
  let x = inputs.get(n)
  if (!x) inputs.set(n, (x = fromData(Float64Array.from(blobData().points.slice(0, n).flat()), [n, 2])))
  return x
}

/** EM from k-means++ seeding number r, which draws from the same child stream at every n and k. */
const fits = new Map<string, GaussianMixtureModel>()
const key = (n: number, k: number, r: number) => `${n}/${k}/${r}`

function fit(n: number, k: number, r: number): GaussianMixtureModel {
  const estimator = gaussianMixture({
    k,
    covariance: 'diagonal',
    regularisation: REGULARISATION,
    tolerance: 1e-5,
    maxSteps: 200,
  })
  const model = estimator.fit(dataset(firstPoints(n)), { stream: child(ROOT, 'restart', r), trace: { every: 1000 } })
  fits.set(key(n, k, r), model)
  return model
}

/** The fits a budget of `restarts` needs on the first n points, restart by restart. */
const needed = (n: number, restarts: number) =>
  Array.from({ length: restarts * K_MAX }, (_, i) => [n, (i % K_MAX) + 1, Math.floor(i / K_MAX)] as const)

/** Whether any fit for this n and restart budget is still to be made. */
export const pending = (n: number, restarts: number) => needed(n, restarts).some(([, k, r]) => !fits.has(key(n, k, r)))

/** Make the next missing fit, restart 0 for every k first. */
export function fitNext(n: number, restarts: number): void {
  const next = needed(n, restarts).find(([, k, r]) => !fits.has(key(n, k, r)))
  if (next) fit(...next)
}

/** How many restarts are fitted for every k: the budget `bestFits` can be asked for without fitting anything. */
export function restartsDone(n: number, restarts: number): number {
  let r = 0
  while (r < restarts && Array.from({ length: K_MAX }, (_, i) => fits.has(key(n, i + 1, r))).every(Boolean)) r++
  return r
}

/** For each k, the highest-likelihood fit of `restarts` EM runs on the first n points. */
export function bestFits(n: number, restarts: number): KFit[] {
  return Array.from({ length: K_MAX }, (_, i) => {
    const k = i + 1
    const runs = Array.from({ length: restarts }, (_, r) => fits.get(key(n, k, r)) ?? fit(n, k, r))
    const best = runs.reduce((a, b) => (b.logLikelihood > a.logLikelihood ? b : a))
    return {
      k,
      deviance: -2 * n * best.logLikelihood,
      means: toRows(best.means) as Point[],
      covariances: toArray(best.covariances) as number[][][],
      model: best,
    }
  })
}

/** The most likely component of each of the first n points under a fit. */
export const labelsOf = (fit: KFit, n: number) =>
  (toRows(fit.model.responsibilities(firstPoints(n))) as number[][]).map((r) => r.indexOf(Math.max(...r)))
