/**
 * Sample autocorrelation and partial autocorrelation, and autoregressive estimation: the Levinson–Durbin recursion,
 * Yule–Walker and Burg's method.
 */

import { normalQuantile } from 'aifn/special'
import { autocorrelation, autocovariance } from 'aifn/stats'
import { dense, type Vector } from 'aifn/tensor'
import { meanOf, toVec, vecT, type VectorLike } from './mat'

/** The sample ACF with its reference bands. */
export type SampleAcf = {
  /** ρ̂(0) … ρ̂(maxLag), with ρ̂(0) = 1. */
  acf: Vector
  /** The white-noise band: ρ̂(k) outside ±band rejects ρ(k) = 0 at the chosen level for an iid series (z/√n). */
  band: number
  /**
   * Bartlett's standard error of ρ̂(k) under the hypothesis that the series is MA(k − 1):
   * √((1 + 2Σ_{j<k} ρ̂(j)²)/n) (Box, Jenkins & Reinsel, 2008, eq. 2.1.15). Entry 0 is 0.
   */
  bartlett: Vector
}

/**
 * The sample autocorrelation ρ̂(k) = γ̂(k)/γ̂(0) for k = 0 … maxLag, with the biased (÷ n) autocovariance, as
 * `aifn/stats`' `autocorrelation` computes it (FFT for long series), plus the white-noise band z/√n and Bartlett's
 * standard errors at the given `level` (default 0.95).
 */
export function sampleAcf(x: VectorLike, maxLag: number, { level = 0.95 }: { level?: number } = {}): SampleAcf {
  const xs = toVec(x, 'sampleAcf')
  const n = xs.length
  const lag = Math.min(maxLag, n - 1)
  const acf = dense.data(autocorrelation(xs, { maxLag: lag }))
  const z = normalQuantile(0.5 + level / 2) as number
  const bartlett = new Float64Array(lag + 1)
  let s = 0
  for (let k = 1; k <= lag; k++) {
    if (k > 1) s += acf[k - 1] ** 2
    bartlett[k] = z * Math.sqrt((1 + 2 * s) / n)
  }
  return { acf: vecT(Array.from(acf)), band: z / Math.sqrt(n), bartlett: vecT(Array.from(bartlett)) }
}

/** The result of the Levinson–Durbin recursion. */
export type LevinsonDurbin = {
  /** φ₁ … φ_p of the order-p AR fit x_t = Σ φ_i x_{t−i} + ε_t. */
  ar: Vector
  /** The reflection coefficients φ_kk, k = 1 … p: the partial autocorrelations. */
  reflection: Vector
  /** The prediction-error variances v₀ = γ(0), v₁, …, v_p. */
  variance: Vector
  /** True if some |φ_kk| ≥ 1 (the autocovariance was not positive definite); later orders are then unreliable. */
  singular: boolean
}

/**
 * The Levinson–Durbin recursion (Levinson, 1947; Durbin, 1960): solves the Toeplitz Yule–Walker equations
 * Σ_j φ_j γ(i − j) = γ(i), i = 1 … p, in O(p²) from autocovariances γ(0) … γ(p). Each order k adds the reflection
 * coefficient φ_kk = (γ(k) − Σ_{j<k} φ_{k−1,j} γ(k − j)) / v_{k−1}, updates φ_kj = φ_{k−1,j} − φ_kk φ_{k−1,k−j}, and
 * v_k = v_{k−1}(1 − φ_kk²) (Brockwell & Davis, 1991, Prop. 5.2.1). Autocorrelations work too (v is then relative).
 */
export function levinsonDurbin(acov: VectorLike, order: number): LevinsonDurbin {
  const g = toVec(acov, 'levinsonDurbin')
  if (order < 0 || order >= g.length)
    throw new Error(`levinsonDurbin: need ${order + 1} autocovariances, got ${g.length}`)
  let phi: number[] = []
  const reflection: number[] = []
  const variance = [g[0]]
  let singular = false
  for (let k = 1; k <= order; k++) {
    const v = variance[k - 1]
    let num = g[k]
    for (let j = 1; j < k; j++) num -= phi[j - 1] * g[k - j]
    const kk = v > 0 ? num / v : NaN
    if (!(Math.abs(kk) < 1)) singular = true
    phi = [...phi.map((p, j) => p - kk * phi[k - 2 - j]), kk]
    reflection.push(kk)
    variance.push(v * (1 - kk * kk))
  }
  return { ar: vecT(phi), reflection: vecT(reflection), variance: vecT(variance), singular }
}

/**
 * The sample partial autocorrelations φ̂_kk, k = 1 … maxLag: the reflection coefficients of Levinson–Durbin run on the
 * sample autocovariances (the "Yule–Walker" PACF, statsmodels' `pacf(method='ywm')`). Entry k − 1 is lag k.
 */
export function samplePacf(x: VectorLike, maxLag: number): Vector {
  const xs = toVec(x, 'samplePacf')
  const lag = Math.min(maxLag, xs.length - 1)
  return levinsonDurbin(autocovariance(xs, { maxLag: lag }), lag).reflection
}

/** An autoregressive fit x_t − μ = Σ φ_i (x_{t−i} − μ) + ε_t, ε_t ~ N(0, σ²). */
export type AutoregressiveFit = {
  ar: Vector
  /** Innovation variance σ². */
  sigma2: number
  /** The mean μ that was subtracted (the sample mean, or 0 with `demean: false`). */
  mean: number
  /** The partial autocorrelations (reflection coefficients) of orders 1 … p. */
  reflection: Vector
}

/**
 * Yule–Walker estimates of an AR(p): solve the Yule–Walker equations with the biased (÷ n) sample autocovariances by
 * Levinson–Durbin, so the fitted polynomial is always stationary; σ² is the order-p prediction-error variance
 * (statsmodels' `yule_walker(x, p, method='mle')`).
 */
export function yuleWalker(
  x: VectorLike,
  order: number,
  { demean = true }: { demean?: boolean } = {},
): AutoregressiveFit {
  const xs = toVec(x, 'yuleWalker')
  const mean = demean ? meanOf(xs) : 0
  const g = autocovariance(xs, { maxLag: order, demean })
  const ld = levinsonDurbin(g, order)
  return { ar: ld.ar, sigma2: (ld.variance.data as Float64Array)[order], mean, reflection: ld.reflection }
}

/**
 * Burg's method (Burg, 1968; Kay, 1988, §7.4): at each order choose the reflection coefficient that minimises the sum
 * of forward and backward prediction-error powers, k_m = 2Σ f_t b_{t−1} / Σ (f_t² + b_{t−1}²), then update the errors
 * and the coefficients by the Levinson step. The estimate is always stationary and, unlike Yule–Walker, uses no
 * zero-padded lags, so it resolves sharp spectral peaks in short series. σ² is the final mean error power.
 */
export function burg(x: VectorLike, order: number, { demean = true }: { demean?: boolean } = {}): AutoregressiveFit {
  const xs = toVec(x, 'burg')
  const n = xs.length
  if (order >= n) throw new Error(`burg: order ${order} needs more than ${n} values`)
  const mean = demean ? meanOf(xs) : 0
  let f = xs.map((v) => v - mean)
  let b = [...f]
  let power = f.reduce((s, v) => s + v * v, 0) / n
  let phi: number[] = []
  const reflection: number[] = []
  for (let m = 1; m <= order; m++) {
    // Forward errors f_t for t = m … n−1 against backward errors b_{t−1}.
    let num = 0
    let den = 0
    for (let t = m; t < n; t++) {
      num += f[t] * b[t - 1]
      den += f[t] * f[t] + b[t - 1] * b[t - 1]
    }
    const k = den > 0 ? (2 * num) / den : 0
    const nf = [...f]
    const nb = [...b]
    for (let t = m; t < n; t++) {
      nf[t] = f[t] - k * b[t - 1]
      nb[t] = b[t - 1] - k * f[t]
    }
    f = nf
    b = nb
    phi = [...phi.map((p, j) => p - k * phi[m - 2 - j]), k]
    reflection.push(k)
    power *= 1 - k * k
  }
  return { ar: vecT(phi), sigma2: power, mean, reflection: vecT(reflection) }
}
