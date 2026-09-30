/**
 * Standard targets with analytic gradients, for demonstrations and tests: a banana (twisted Gaussian), a Gaussian, an
 * isotropic Gaussian mixture and Neal's funnel.
 */

import { inverse, logDet } from 'aifn/linalg'
import { tensor, type Tensor } from 'aifn/tensor'
import type { Target, VectorLike } from './types'
import { data, logSumExp, toF64, type F64 } from './util'

const LOG_2PI = Math.log(2 * Math.PI)

/**
 * The banana (twisted Gaussian of Haario, Saksman & Tamminen, 1999) in two dimensions:
 * log π(x, y) = −½(x/a)² − ½(y − b(x² − a²))² + const. x ~ N(0, a²) and y | x ~ N(b(x² − a²), 1), so the mean is
 * (0, 0), Var x = a², Var y = 1 + 2b²a⁴, and b bends the ridge (b = 0 is a Gaussian). Default a = 1, b = 1.
 */
export function banana(options: { a?: number; b?: number } = {}): Target & { mean: Tensor; variance: Tensor } {
  const { a = 1, b = 1 } = options
  const read = (t: Tensor) => {
    const d = data(t)
    return [d[0], d[1]] as const
  }
  return {
    name: 'banana',
    dim: 2,
    logDensity: (theta) => {
      const [x, y] = read(theta)
      const r = y - b * (x * x - a * a)
      return -0.5 * (x / a) ** 2 - 0.5 * r * r - LOG_2PI - Math.log(a)
    },
    grad: (theta) => {
      const [x, y] = read(theta)
      const r = y - b * (x * x - a * a)
      return [-x / (a * a) + 2 * b * x * r, -r]
    },
    mean: tensor([0, 0]),
    variance: tensor([a * a, 1 + 2 * b * b * a ** 4]),
  }
}

/**
 * A normalised multivariate Gaussian target N(mean, covariance): log π(θ) = −½(θ − μ)ᵀΣ⁻¹(θ − μ) − ½ log|2πΣ|, with
 * gradient −Σ⁻¹(θ − μ). `precision` is Σ⁻¹ (d×d).
 */
export function gaussianTarget(
  mean: VectorLike,
  covariance: Tensor | readonly (readonly number[])[],
): Target & { mean: Tensor; covariance: Tensor; precision: Tensor } {
  const mu = toF64(mean, 'gaussianTarget')
  const d = mu.length
  const cov = Array.isArray(covariance) ? tensor(covariance as number[][]) : (covariance as Tensor)
  if (cov.shape.length !== 2 || cov.shape[0] !== d || cov.shape[1] !== d)
    throw new Error(`gaussianTarget: covariance must be ${d}×${d}`)
  const precision = inverse(cov)
  const P = data(precision)
  const logNorm = -0.5 * (d * LOG_2PI + (logDet(cov) as number))
  const gradAt = (x: F64): F64 => {
    const g = new Float64Array(d)
    for (let i = 0; i < d; i++) {
      let s = 0
      for (let j = 0; j < d; j++) s += P[i * d + j] * (x[j] - mu[j])
      g[i] = -s
    }
    return g
  }
  return {
    name: 'gaussian',
    dim: d,
    logDensity: (theta) => {
      const x = data(theta)
      const g = gradAt(x)
      let q = 0
      for (let i = 0; i < d; i++) q += (x[i] - mu[i]) * g[i]
      return 0.5 * q + logNorm
    },
    grad: (theta) => gradAt(data(theta)),
    mean: tensor(Array.from(mu)),
    covariance: cov,
    precision,
  }
}

/**
 * An isotropic Gaussian mixture Σₖ wₖ N(θ | mₖ, σ²I) (normalised). `means` is K×d; `weights` default to equal.
 */
export function gaussianMixtureTarget(
  means: readonly (readonly number[])[],
  sd: number,
  weights?: readonly number[],
): Target & { means: Tensor; weights: Tensor } {
  const K = means.length
  const d = means[0].length
  const w = weights ? weights.map((v) => v / weights.reduce((a, b) => a + b, 0)) : new Array<number>(K).fill(1 / K)
  const logW = w.map(Math.log)
  const componentLogs = (x: F64) =>
    means.map((m, k) => {
      let q = 0
      for (let i = 0; i < d; i++) q += (x[i] - m[i]) ** 2
      return logW[k] - (0.5 * q) / (sd * sd) - d * (Math.log(sd) + 0.5 * LOG_2PI)
    })
  return {
    name: 'gaussian-mixture',
    dim: d,
    logDensity: (theta) => logSumExp(componentLogs(data(theta))),
    grad: (theta) => {
      const x = data(theta)
      const logs = componentLogs(x)
      const total = logSumExp(logs)
      const g = new Float64Array(d)
      means.forEach((m, k) => {
        const r = Math.exp(logs[k] - total)
        for (let i = 0; i < d; i++) g[i] -= (r * (x[i] - m[i])) / (sd * sd)
      })
      return g
    },
    means: tensor(means as number[][]),
    weights: tensor(w),
  }
}

/**
 * Neal's funnel (Neal, 2003, §8) in d dimensions: v ~ N(0, s²) and xᵢ | v ~ N(0, eᵛ) for i = 1 … d − 1, with θ = (v,
 * x₁, …). Its neck (v ≪ 0) needs small steps and its mouth large ones, so fixed-step HMC diverges there. Default
 * d = 2, s = 3.
 */
export function funnel(options: { dim?: number; scale?: number } = {}): Target {
  const { dim = 2, scale = 3 } = options
  return {
    name: 'funnel',
    dim,
    logDensity: (theta) => {
      const x = data(theta)
      const v = x[0]
      let q = 0
      for (let i = 1; i < dim; i++) q += x[i] * x[i]
      return (
        -0.5 * (v / scale) ** 2 -
        Math.log(scale) -
        0.5 * LOG_2PI -
        0.5 * q * Math.exp(-v) -
        ((dim - 1) * (v + LOG_2PI)) / 2
      )
    },
    grad: (theta) => {
      const x = data(theta)
      const v = x[0]
      const g = new Float64Array(dim)
      let q = 0
      for (let i = 1; i < dim; i++) {
        q += x[i] * x[i]
        g[i] = -x[i] * Math.exp(-v)
      }
      g[0] = -v / (scale * scale) + 0.5 * q * Math.exp(-v) - (dim - 1) / 2
      return g
    },
  }
}
