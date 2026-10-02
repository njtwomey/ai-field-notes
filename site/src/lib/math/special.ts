/** Special functions backed by aifn/numerics/special primitives. */
import {
  erf as aifnErf,
  logFactorial as aifnLogFactorial,
  logChoose as aifnLogChoose,
  normalPdf as aifnNormalPdf,
  normalCdf as aifnNormalCdf,
  normalQuantile as aifnNormalQuantile,
  logGamma as aifnLogGamma,
  regularisedBeta,
  regularisedGammaP,
  studentTCdf as aifnStudentTCdf,
} from 'aifn/numerics/special'

/** log(n!) for integer n ≥ 0. */
export const logFactorial = (n: number): number => aifnLogFactorial(n) as number

/** log of the binomial coefficient n choose k. */
export const logChoose = (n: number, k: number): number => aifnLogChoose(n, k) as number

/** Error function. */
export const erf = (x: number): number => aifnErf(x) as number

export const normalPdf = (z: number): number => aifnNormalPdf(z) as number
export const normalCdf = (z: number): number => aifnNormalCdf(z) as number

/** Standard normal quantile. */
export const normalQuantile = (p: number): number => aifnNormalQuantile(p) as number

/** log Γ(x) for x > 0. */
export const logGamma = (x: number): number => aifnLogGamma(x) as number

/** Regularised incomplete beta function I_x(a, b). */
export const incompleteBeta = (x: number, a: number, b: number): number => regularisedBeta(a, b, x) as number

/** Student t cdf with ν degrees of freedom. */
export const studentTCdf = (t: number, df: number): number => aifnStudentTCdf(t, df) as number

/** Regularised lower incomplete gamma P(a, x) = γ(a, x)/Γ(a), for a > 0 and x ≥ 0. */
export const incompleteGamma = (a: number, x: number): number => regularisedGammaP(a, x) as number

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
