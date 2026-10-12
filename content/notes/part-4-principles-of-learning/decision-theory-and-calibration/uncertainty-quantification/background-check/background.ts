/**
 * Background Check's combination step (Perello-Nieto, Silva Filho, Kull and Flach, 2016), as closed-form formulas:
 * a background bias μ maps the relative foreground density q_f to a relative background density q_b = μ(q_f), the
 * familiarity is r = q_f / q_b, and the (k + 1)-class posterior is p(b | x) = 1/(1 + r) and
 * p(c | x) = p(c | f, x) r/(1 + r). Inputs are logs so that far tails (q_f → 0) stay finite.
 *
 * A candidate for the engine: `backgroundCheck(logQf, within, bias)` beside the one-class models in
 * `aifn-methods/unsupervised/anomaly`.
 */

/** μ(z) = μ0 + (μ1 − μ0) z^γ: affine for γ = 1, constant for μ0 = μ1, monotone for any γ > 0. */
export type Bias = { mu0: number; mu1: number; gamma: number }

/** μ(z) itself, for drawing the bias. */
export const bias = (z: number, b: Bias) => b.mu0 + (b.mu1 - b.mu0) * z ** b.gamma

/** log q_b for log q_f under the bias (−∞ where the bias puts no background). */
export function logBackground(logQf: number, b: Bias): number {
  // With μ0 = 0, q_b = μ1 z^γ: work in logs so that a vanishing q_f does not give 0/0 in the familiarity.
  if (b.mu0 === 0) return b.mu1 > 0 ? Math.log(b.mu1) + b.gamma * logQf : -Infinity
  const v = b.mu0 + (b.mu1 - b.mu0) * Math.exp(b.gamma * logQf)
  return v > 0 ? Math.log(v) : -Infinity
}

/** log r = log q_f − log q_b; +∞ where the bias puts no background. */
export function logFamiliarity(logQf: number, b: Bias): number {
  if (b.mu0 === 0 && b.mu1 > 0) return (1 - b.gamma) * logQf - Math.log(b.mu1)
  return logQf - logBackground(logQf, b)
}

/** The k + 1 probabilities (classes first, background last) and p(f | x) = r/(1 + r). */
export function backgroundCheck(logQf: number, within: ArrayLike<number>, b: Bias) {
  const lr = logFamiliarity(logQf, b)
  const pf = lr === Infinity ? 1 : 1 / (1 + Math.exp(-lr))
  const probs = Array.from(within, (p) => p * pf)
  probs.push(1 - pf)
  return { pf, logR: lr, probs }
}

/** Index of the largest probability; the background (last) wins ties so that a tie is a rejection. */
export function decide(probs: readonly number[]): number {
  let best = probs.length - 1
  for (let c = 0; c < probs.length - 1; c++) if (probs[c] > probs[best]) best = c
  return best
}
