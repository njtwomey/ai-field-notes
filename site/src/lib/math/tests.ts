/** Helpers for statistical-test widgets, backed by aifn/numerics/special. */
import { logChoose, logGamma, normalCdf } from './special'
import { studentTQuantile as aifnStudentTQuantile } from 'aifn/numerics/special'

/** Quantile of Student's t with ν degrees of freedom. */
export const studentTQuantile = (p: number, df: number): number => aifnStudentTQuantile(p, df) as number

/** Binomial pmf P(S = k) for S ~ Binom(n, p). */
export function binomialPmf(k: number, n: number, p: number): number {
  if (k < 0 || k > n) return 0
  if (p === 0) return k === 0 ? 1 : 0
  if (p === 1) return k === n ? 1 : 0
  return Math.exp(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p))
}

/** Upper tail P(S ≥ k) for S ~ Binom(n, p). */
export function binomialUpper(k: number, n: number, p: number): number {
  let total = 0
  for (let j = Math.max(0, k); j <= n; j++) total += binomialPmf(j, n, p)
  return Math.min(1, total)
}

/** Lower tail P(S ≤ k) for S ~ Binom(n, p). */
export function binomialLower(k: number, n: number, p: number): number {
  let total = 0
  for (let j = 0; j <= Math.min(n, k); j++) total += binomialPmf(j, n, p)
  return Math.min(1, total)
}

/** Two-sided z-test power for a standardised shift λ = δ√n (in standard errors) and critical value c. */
export const zPower = (shift: number, crit: number) => 1 - normalCdf(crit - shift) + normalCdf(-crit - shift)

/**
 * Two-sided t-test power: P(|T'| > t_crit) for a noncentral t with ν degrees of freedom and noncentrality λ.
 * T' = (Z + λ) / √(V/ν) with V ~ χ²_ν, so the power is the average over V of a normal tail probability. The average is
 * taken by Simpson's rule on the χ² density.
 */
export function tPower(shift: number, df: number, alpha: number): number {
  const crit = studentTQuantile(1 - alpha / 2, df)
  const logNorm = -(df / 2) * Math.log(2) - logGamma(df / 2)
  const density = (v: number) => (v <= 0 ? 0 : Math.exp(logNorm + (df / 2 - 1) * Math.log(v) - v / 2))
  const top = df + 12 * Math.sqrt(2 * df) + 20
  const steps = 800
  const h = top / steps
  let total = 0
  for (let i = 0; i <= steps; i++) {
    const v = i * h
    const scale = crit * Math.sqrt(v / df)
    const f = density(v) * (1 - normalCdf(scale - shift) + normalCdf(-scale - shift))
    total += f * (i === 0 || i === steps ? 1 : i % 2 ? 4 : 2)
  }
  return Math.min(1, (total * h) / 3)
}
