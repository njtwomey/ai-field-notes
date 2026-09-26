/** Special functions for the distribution registry. Accurate to roughly 1e-7, which is ample for figures. */

const LOG_FACTORIAL: number[] = [0]

/** log(n!) for integer n ≥ 0, cached. */
export function logFactorial(n: number): number {
  for (let i = LOG_FACTORIAL.length; i <= n; i++) LOG_FACTORIAL[i] = LOG_FACTORIAL[i - 1] + Math.log(i)
  return LOG_FACTORIAL[n]
}

/** log of the binomial coefficient n choose k. */
export const logChoose = (n: number, k: number) => logFactorial(n) - logFactorial(k) - logFactorial(n - k)

/** Error function, Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7). */
export function erf(x: number): number {
  const s = Math.sign(x)
  const a = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * a)
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)
  return s * y
}

export const normalPdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)
export const normalCdf = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2))

/** Standard normal quantile (Acklam's rational approximation, relative error below 1.2e-9). */
export function normalQuantile(p: number): number {
  const a = [
    -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239,
  ]
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
  const c = [
    -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968,
    2.938163982698783,
  ]
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
  const lo = 0.02425
  if (p <= 0) return -Infinity
  if (p >= 1) return Infinity
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p))
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    )
  }
  if (p > 1 - lo) return -normalQuantile(1 - p)
  const q = p - 0.5
  const r = q * q
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  )
}

/** log Γ(x) for x > 0 (Lanczos approximation, g = 7; about 15 significant digits). */
export function logGamma(x: number): number {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x)
  x -= 1
  let a = c[0]
  const t = x + 7.5
  for (let i = 1; i < 9; i++) a += c[i] / (x + i)
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
}

/** Continued fraction for the incomplete beta function (modified Lentz's method). */
function betaContinuedFraction(a: number, b: number, x: number): number {
  const tiny = 1e-300
  let c = 1
  let d = 1 - ((a + b) * x) / (a + 1)
  if (Math.abs(d) < tiny) d = tiny
  d = 1 / d
  let h = d
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2))
    d = 1 + aa * d
    if (Math.abs(d) < tiny) d = tiny
    c = 1 + aa / c
    if (Math.abs(c) < tiny) c = tiny
    d = 1 / d
    h *= d * c
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1))
    d = 1 + aa * d
    if (Math.abs(d) < tiny) d = tiny
    c = 1 + aa / c
    if (Math.abs(c) < tiny) c = tiny
    d = 1 / d
    const delta = d * c
    h *= delta
    if (Math.abs(delta - 1) < 1e-14) break
  }
  return h
}

/** Regularised incomplete beta function I_x(a, b). */
export function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x))
  // The continued fraction converges fastest on this side of the mean; use symmetry otherwise.
  return x < (a + 1) / (a + b + 2)
    ? (front * betaContinuedFraction(a, b, x)) / a
    : 1 - (front * betaContinuedFraction(b, a, 1 - x)) / b
}

/** Student t cdf with ν degrees of freedom (ν may be fractional, as in Welch's test). */
export function studentTCdf(t: number, df: number): number {
  const tail = 0.5 * incompleteBeta(df / (df + t * t), df / 2, 0.5)
  return t >= 0 ? 1 - tail : tail
}

/** Regularised lower incomplete gamma P(a, x) = γ(a, x)/Γ(a), for a > 0 and x ≥ 0. */
export function incompleteGamma(a: number, x: number): number {
  if (x <= 0) return 0
  if (!Number.isFinite(x)) return 1
  const front = Math.exp(a * Math.log(x) - x - logGamma(a))
  if (x < a + 1) {
    // Series: P(a, x) = front · Σ xⁿ / (a (a + 1) ⋯ (a + n)).
    let term = 1 / a
    let total = term
    for (let n = 1; n < 500; n++) {
      term *= x / (a + n)
      total += term
      if (term < total * 1e-15) break
    }
    return Math.min(front * total, 1)
  }
  // Continued fraction for Q(a, x) = 1 − P(a, x) (modified Lentz's method).
  const tiny = 1e-300
  let b = x + 1 - a
  let c = 1 / tiny
  let d = 1 / b
  let h = d
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a)
    b += 2
    d = an * d + b
    if (Math.abs(d) < tiny) d = tiny
    c = b + an / c
    if (Math.abs(c) < tiny) c = tiny
    d = 1 / d
    const delta = d * c
    h *= delta
    if (Math.abs(delta - 1) < 1e-15) break
  }
  return Math.max(1 - front * h, 0)
}

/**
 * Invert an increasing cdf by bisection: the x with cdf(x) = u. `lo` must satisfy cdf(lo) ≤ u; `hi` is a starting
 * guess and doubles until cdf(hi) ≥ u. For plot ranges and quantiles without a closed form.
 */
export function invertCdf(cdf: (x: number) => number, u: number, lo: number, hi: number): number {
  for (let i = 0; i < 200 && cdf(hi) < u; i++) hi = lo + 2 * (hi - lo)
  for (let i = 0; i < 100; i++) {
    const mid = 0.5 * (lo + hi)
    if (cdf(mid) < u) lo = mid
    else hi = mid
  }
  return 0.5 * (lo + hi)
}
