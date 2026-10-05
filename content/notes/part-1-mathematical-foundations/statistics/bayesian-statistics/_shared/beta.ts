import { logGamma, regularisedBeta, regularisedBetaInverse } from 'aifn-compute/numerics/special'

/** Beta(a, b) density; handles the boundary so that a U-shaped density does not produce NaN at 0 or 1. */
export function betaPdf(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return 0
  return Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x))
}

export const betaCdf = (x: number, a: number, b: number) => regularisedBeta(a, b, x)

export const betaQuantile = (p: number, a: number, b: number) => regularisedBetaInverse(a, b, p)

/**
 * Shortest interval holding mass `level`: minimise q(p + level) − q(p) over the lower-tail mass p ∈ [0, 1 − level]
 * by golden-section search. For a unimodal density the shortest interval is the highest-density interval.
 */
export function betaHpd(a: number, b: number, level: number): [number, number] {
  const width = (p: number) => betaQuantile(p + level, a, b) - betaQuantile(p, a, b)
  let lo = 0
  let hi = 1 - level
  const g = (Math.sqrt(5) - 1) / 2
  for (let i = 0; i < 40; i++) {
    const m1 = hi - g * (hi - lo)
    const m2 = lo + g * (hi - lo)
    if (width(m1) < width(m2)) hi = m2
    else lo = m1
  }
  const p = 0.5 * (lo + hi)
  return [betaQuantile(p, a, b), betaQuantile(p + level, a, b)]
}
