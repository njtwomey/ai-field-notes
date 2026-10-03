/**
 * Gaussian helpers shared by the model-based machine learning figures: densities, a tail-accurate normal cdf, and the
 * truncation functions v and w that every threshold (probit) factor uses.
 */

const SQRT2PI = Math.sqrt(2 * Math.PI)

/** Complementary error function with relative error below 1.2e-7 everywhere, including far into the tails. */
export function erfc(x: number): number {
  const z = Math.abs(x)
  const t = 1 / (1 + 0.5 * z)
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    )
  return x >= 0 ? r : 2 - r
}

export const phi = (x: number) => Math.exp(-0.5 * x * x) / SQRT2PI
export const Phi = (x: number) => 0.5 * erfc(-x / Math.SQRT2)

/** Density of N(mean, variance) at x. */
export const gaussPdf = (x: number, mean: number, variance: number) =>
  Math.exp((-0.5 * (x - mean) ** 2) / variance) / Math.sqrt(2 * Math.PI * variance)

/**
 * v(t) = φ(t)/Φ(t): the mean shift of a standard normal truncated to (−t, ∞). For very negative t the ratio is computed
 * from its asymptotic series, where both φ and Φ underflow.
 */
export function vTrunc(t: number): number {
  if (t < -30) return -t - 1 / t + 2 / t ** 3
  return phi(t) / Phi(t)
}

/** w(t) = v(t)(v(t) + t): the fractional variance reduction of the same truncation, in (0, 1). */
export function wTrunc(t: number): number {
  const v = vTrunc(t)
  return v * (v + t)
}

/** Grid of n evenly spaced points on [a, b]. */
export function grid(a: number, b: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
}
