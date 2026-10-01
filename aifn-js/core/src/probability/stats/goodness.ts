/**
 * The Kolmogorov–Smirnov statistic and test, part of `aifn/probability/stats`: one sample against a continuous
 * reference cdf, or two samples against each other, as `scipy.stats.kstest` and `scipy.stats.ks_2samp` (two-sided).
 *
 * - One sample: D = supₓ |Fₙ(x) − F(x)|, attained at a sorted draw x₍ᵢ₎ as max(i/n − F(x₍ᵢ₎), F(x₍ᵢ₎) − (i − 1)/n). The
 *   exact null distribution of D for continuous F is computed by Marsaglia, Tsang and Wang's matrix-power algorithm
 *   (2003, J. Stat. Softw. 8(18)); far in the tail (n d² > 7.24, or > 3.76 with n > 99) their asymptotic expansion
 *   is used, as in their paper, and for n d ≥ 100 Kolmogorov's limit with Stephens' correction (≈0.3% relative). `method: 'asymp'` uses Kolmogorov's limit P(√n D > λ) → 2 Σₖ (−1)ᵏ⁻¹ e^{−2k²λ²}.
 * - Two samples of sizes m and n: D = supₓ |F_m(x) − G_n(x)|. Under the null every interleaving of the pooled sorted
 *   samples is equally likely, so P(D ≥ d) is one minus the probability that a uniformly random monotone lattice path
 *   from (0, 0) to (m, n) keeps |i/m − j/n| < d, computed exactly by dynamic programming in integers (Hodges, 1958,
 *   Ark. Mat. 3). Above m·n = 10⁶ the one-sample exact law at n = round(mn/(m + n)) is used, as scipy's 'asymp'.
 */

import { DomainError } from 'aifn/foundation/errors'
import type { Data } from './input'
import { toSequence } from './input'

/** The two-sided Kolmogorov–Smirnov statistic with where it is attained. */
export type KsStatistic = {
  /** D = sup |difference of the cdfs|. */
  statistic: number
  /** The x at which D is attained. */
  location: number
  /** +1 when the (first) empirical cdf is above the reference there, −1 when below. */
  sign: 1 | -1
}

/** The result of `ksTest`. */
export type KsTest = KsStatistic & {
  /** P(D ≥ observed) under the null hypothesis. */
  pValue: number
  /** How the p-value was computed. */
  method: 'exact' | 'asymp'
}

/** A reference distribution: a continuous cdf (one-sample test) or a second sample (two-sample test). */
export type KsReference = ((x: number) => number) | Data

function sortedValues(x: Data, what: string): Float64Array {
  const v = Float64Array.from(toSequence(x, what))
  if (v.length === 0) throw new DomainError(what, `${what}: needs at least one value`)
  if (v.some(Number.isNaN)) throw new DomainError(what, `${what}: the sample contains NaN`)
  return v.sort()
}

/** D for one sorted sample against a cdf. */
function oneSample(x: Float64Array, cdf: (x: number) => number): KsStatistic {
  const n = x.length
  let best: KsStatistic = { statistic: -1, location: x[0], sign: 1 }
  for (let i = 0; i < n; i++) {
    const f = cdf(x[i])
    const above = (i + 1) / n - f
    const below = f - i / n
    if (above > best.statistic) best = { statistic: above, location: x[i], sign: 1 }
    if (below > best.statistic) best = { statistic: below, location: x[i], sign: -1 }
  }
  return best
}

/** D for two sorted samples: the largest gap between the empirical cdfs, read just after each distinct value. */
function twoSample(x: Float64Array, y: Float64Array): KsStatistic {
  const m = x.length
  const n = y.length
  let i = 0
  let j = 0
  let best: KsStatistic = { statistic: 0, location: Math.min(x[0], y[0]), sign: 1 }
  while (i < m || j < n) {
    const v = Math.min(i < m ? x[i] : Infinity, j < n ? y[j] : Infinity)
    while (i < m && x[i] === v) i++
    while (j < n && y[j] === v) j++
    const d = i / m - j / n
    if (Math.abs(d) > best.statistic) best = { statistic: Math.abs(d), location: v, sign: d > 0 ? 1 : -1 }
  }
  return best
}

/**
 * The two-sided Kolmogorov–Smirnov statistic of a sample against a continuous cdf, or of two samples, with the x at
 * which it is attained (as scipy's `statistic_location` and `statistic_sign`).
 */
export function ksStatistic(x: Data, reference: KsReference): KsStatistic {
  const xs = sortedValues(x, 'ksStatistic')
  return typeof reference === 'function'
    ? oneSample(xs, reference)
    : twoSample(xs, sortedValues(reference, 'ksStatistic'))
}

/**
 * The two-sided Kolmogorov–Smirnov test of a sample against a continuous cdf (`scipy.stats.kstest`) or of two samples
 * (`scipy.stats.ks_2samp`). `method` 'exact' (default: the exact null law; see the module notes for the tail and large
 * two-sample cases) or 'asymp' (Kolmogorov's limit for one sample; the one-sample exact law at the effective size for
 * two). Ties in a two-sample test make the p-value conservative.
 */
export function ksTest(x: Data, reference: KsReference, options: { method?: 'exact' | 'asymp' } = {}): KsTest {
  const method = options.method ?? 'exact'
  const xs = sortedValues(x, 'ksTest')
  if (typeof reference === 'function') {
    const s = oneSample(xs, reference)
    const n = xs.length
    const pValue = method === 'exact' ? kolmogorovSf(s.statistic, n) : kolmogorovLimitSf(s.statistic * Math.sqrt(n))
    return { ...s, pValue: clamp01(pValue), method }
  }
  const ys = sortedValues(reference, 'ksTest')
  const s = twoSample(xs, ys)
  const [m, n] = [xs.length, ys.length]
  const exact = method === 'exact' && m * n <= 1e6
  const pValue = exact ? twoSampleSf(s.statistic, m, n) : kolmogorovSf(s.statistic, Math.round((m * n) / (m + n)))
  return { ...s, pValue: clamp01(pValue), method: exact ? 'exact' : 'asymp' }
}

const clamp01 = (p: number) => Math.min(1, Math.max(0, p))

// ── Null distributions ───────────────────────────────────────────────────────────────────────────────────────────────

/** Kolmogorov's limit P(K > λ) = 2 Σₖ (−1)ᵏ⁻¹ e^{−2k²λ²} (scipy's `kstwobign.sf`). */
export function kolmogorovLimitSf(lambda: number): number {
  if (!(lambda > 0)) return 1
  // For small λ the alternating series converges slowly; use the Jacobi-transformed form √(2π)/λ Σ e^{−(2k−1)²π²/(8λ²)}.
  if (lambda < 1) {
    let s = 0
    for (let k = 1; k <= 20; k++) s += Math.exp((-((2 * k - 1) ** 2) * Math.PI ** 2) / (8 * lambda * lambda))
    return 1 - (Math.sqrt(2 * Math.PI) / lambda) * s
  }
  let s = 0
  for (let k = 1; k <= 100; k++) {
    const term = Math.exp(-2 * k * k * lambda * lambda)
    s += (k % 2 === 1 ? 2 : -2) * term
    if (term < 1e-17) break
  }
  return s
}

/**
 * P(Dₙ ≥ d) for the one-sample statistic of n draws from a continuous distribution (scipy's `kstwo.sf`), by
 * Marsaglia, Tsang and Wang (2003): P(Dₙ < d) = n!/nⁿ · (Hⁿ)ₖₖ with k = ⌊nd⌋ + 1 and H the (2k − 1)-square matrix of
 * their paper, its power taken by squaring with a decimal exponent kept aside so that nothing overflows.
 */
export function kolmogorovSf(d: number, n: number): number {
  if (!(d > 0)) return 1
  if (d >= 1) return 0
  const s = d * d * n
  if (s > 7.24 || (s > 3.76 && n > 99)) return 2 * Math.exp(-(2.000071 + 0.331 / Math.sqrt(n) + 1.409 / n) * s)
  const k = Math.floor(n * d) + 1
  // Past k = 100 the (2k − 1)-square power costs ~log₂n · 8k³ operations; Kolmogorov's limit at Stephens' corrected
  // λ = (√n + 0.12 + 0.11/√n) d (Stephens, 1970, JRSS B 32(1)) is within ~0.3% relative there (n ≥ 1000).
  if (k > 100) return kolmogorovLimitSf((Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * d)
  const m = 2 * k - 1
  const h = k - n * d
  const H = new Float64Array(m * m)
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) H[i * m + j] = i - j + 1 < 0 ? 0 : 1
  for (let i = 0; i < m; i++) {
    H[i * m] -= h ** (i + 1)
    H[(m - 1) * m + i] -= h ** (m - i)
  }
  H[(m - 1) * m] += 2 * h - 1 > 0 ? (2 * h - 1) ** m : 0
  for (let i = 0; i < m; i++)
    for (let j = 0; j < m; j++) if (i - j + 1 > 0) for (let g = 1; g <= i - j + 1; g++) H[i * m + j] /= g
  const { Q, exponent } = matrixPower(H, m, n)
  let v = Q[(k - 1) * m + k - 1]
  let e = exponent
  for (let i = 1; i <= n; i++) {
    v = (v * i) / n
    if (v < 1e-140) {
      v *= 1e140
      e -= 140
    }
  }
  return 1 - v * 10 ** e
}

/** A·B for m-square row-major matrices. */
function square(a: Float64Array, b: Float64Array, m: number): Float64Array {
  const out = new Float64Array(m * m)
  for (let i = 0; i < m; i++)
    for (let l = 0; l < m; l++) {
      const ail = a[i * m + l]
      for (let j = 0; j < m; j++) out[i * m + j] += ail * b[l * m + j]
    }
  return out
}

/** Aᵖ as Q · 10^exponent, by repeated squaring, rescaling by 10⁻¹⁴⁰ whenever the centre entry exceeds 10¹⁴⁰. */
function matrixPower(a: Float64Array, m: number, p: number): { Q: Float64Array; exponent: number } {
  if (p === 1) return { Q: Float64Array.from(a), exponent: 0 }
  const half = matrixPower(a, m, Math.floor(p / 2))
  let Q = square(half.Q, half.Q, m)
  let exponent = 2 * half.exponent
  if (p % 2 === 1) Q = square(a, Q, m)
  const centre = Math.floor(m / 2)
  if (Q[centre * m + centre] > 1e140) {
    for (let i = 0; i < Q.length; i++) Q[i] *= 1e-140
    exponent += 140
  }
  return { Q, exponent }
}

/**
 * P(D ≥ d) for two samples of sizes m and n under the null, exactly: one minus the probability that a uniformly random
 * monotone lattice path from (0, 0) to (m, n) stays strictly inside |i/m − j/n| < d. Compared in integers,
 * |i·n − j·m| < d·m·n, since every attainable D is a multiple of 1/(mn). Each point's probability is accumulated with
 * the path's step probabilities, (m − i)/(m − i + n − j) for a step in i, so nothing overflows.
 */
function twoSampleSf(d: number, m: number, n: number): number {
  const bound = Math.round(d * m * n)
  const inside = (i: number, j: number) => Math.abs(i * n - j * m) < bound
  // p[j]: probability of reaching (i, j) without leaving the band, for the current i.
  let p = new Float64Array(n + 1)
  p[0] = 1
  for (let j = 1; j <= n; j++) p[j] = inside(0, j) ? (p[j - 1] * (n - j + 1)) / (m + n - j + 1) : 0
  for (let i = 1; i <= m; i++) {
    const next = new Float64Array(n + 1)
    for (let j = 0; j <= n; j++) {
      if (!inside(i, j)) continue
      // Arrive from (i − 1, j) by a step in i, or from (i, j − 1) by a step in j.
      const fromI = (p[j] * (m - i + 1)) / (m - i + 1 + n - j)
      const fromJ = j > 0 ? (next[j - 1] * (n - j + 1)) / (m - i + n - j + 1) : 0
      next[j] = fromI + fromJ
    }
    p = next
  }
  return 1 - p[n]
}
