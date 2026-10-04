import { normalCdf, normalQuantile } from 'aifn/numerics/special'
/** Helpers shared by the A/A, A/B and sample-ratio-mismatch examples: binomial draws and the two-proportion z-test. */

/**
 * An exact Binomial(n, p) draw by geometric skips: the gap before each success is geometric, so the cost is about np
 * uniforms rather than n.
 */
export function sampleBinomial(n: number, p: number, uniform: () => number): number {
  if (p <= 0 || n <= 0) return 0
  if (p >= 1) return n
  if (p > 0.5) return n - sampleBinomial(n, 1 - p, uniform)
  const logQ = Math.log(1 - p)
  let count = 0
  let position = 0
  for (;;) {
    position += Math.floor(Math.log(1 - uniform()) / logQ) + 1
    if (position > n) return count
    count++
  }
}

export type ZTest = {
  /** Observed conversion rates of A and B. */
  pa: number
  pb: number
  diff: number
  /** Standard error of the difference under H₀ (pooled rate), used for the test. */
  se0: number
  z: number
  p: number
  /** Unpooled standard error, used for the confidence interval. */
  se: number
  ci: [number, number]
}

/** Two-sided two-proportion z-test of B against A, with a 1 − α confidence interval for p_B − p_A. */
export function twoProportionZ(xa: number, na: number, xb: number, nb: number, alpha: number): ZTest {
  const pa = xa / na
  const pb = xb / nb
  const pooled = (xa + xb) / (na + nb)
  const se0 = Math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
  const diff = pb - pa
  const z = se0 > 0 ? diff / se0 : 0
  const se = Math.sqrt((pa * (1 - pa)) / na + (pb * (1 - pb)) / nb)
  const zc = normalQuantile(1 - alpha / 2)
  return { pa, pb, diff, se0, z, p: 2 * (1 - normalCdf(Math.abs(z))), se, ci: [diff - zc * se, diff + zc * se] }
}

/** Users per arm for a two-sided test at level α to detect p₀ → p₁ with the given power. */
export function requiredN(p0: number, p1: number, alpha: number, power: number): number {
  const za = normalQuantile(1 - alpha / 2)
  const zb = normalQuantile(power)
  const bar = (p0 + p1) / 2
  const top = za * Math.sqrt(2 * bar * (1 - bar)) + zb * Math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
  return Math.ceil((top * top) / (p1 - p0) ** 2)
}

/** Power of the two-sided pooled z-test for p₀ → p₁ with n users per arm (normal approximation). */
export function zTestPower(p0: number, p1: number, n: number, alpha: number): number {
  const za = normalQuantile(1 - alpha / 2)
  const bar = (p0 + p1) / 2
  const s0 = Math.sqrt((2 * bar * (1 - bar)) / n)
  const s1 = Math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n)
  const d = p1 - p0
  return 1 - normalCdf((za * s0 - d) / s1) + normalCdf((-za * s0 - d) / s1)
}
