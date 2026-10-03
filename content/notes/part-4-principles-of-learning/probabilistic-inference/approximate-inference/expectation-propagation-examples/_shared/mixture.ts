/**
 * Minka's mixture-weight problem with a Beta approximating family: p(x | w) = w p₁(x) + (1 − w) p₂(x) with known
 * components, a uniform prior on w, and Beta sites w^α (1 − w)^β. The projection onto the Beta family matches
 * E[log w] and E[log(1 − w)] (the KL projection) or, as an alternative, the mean and variance.
 */

/** log Γ(x) for x > 0 (Lanczos, g = 7). */
export function logGamma(x: number): number {
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ]
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x)
  const z = x - 1
  let a = c[0]
  const t = z + 7.5
  for (let i = 1; i < 9; i++) a += c[i] / (z + i)
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a)
}

/** Digamma ψ(x) for x > 0: recurrence up to x ≥ 6, then the asymptotic series. */
export function digamma(x: number): number {
  let r = 0
  while (x < 6) {
    r -= 1 / x
    x += 1
  }
  const f = 1 / (x * x)
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))
}

/** Trigamma ψ′(x) for x > 0. */
export function trigamma(x: number): number {
  let r = 0
  while (x < 6) {
    r += 1 / (x * x)
    x += 1
  }
  const f = 1 / (x * x)
  return r + 1 / x + f / 2 + (f / x) * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)))
}

export type BetaParams = { a: number; b: number }

export const betaLogPdf = (w: number, { a, b }: BetaParams) =>
  (a - 1) * Math.log(w) + (b - 1) * Math.log1p(-w) - (logGamma(a) + logGamma(b) - logGamma(a + b))

/** The Beta(a, b) with E[log w] = c1 and E[log(1 − w)] = c2, by Newton's method from a starting point. */
export function solveBeta(c1: number, c2: number, start: BetaParams): BetaParams {
  let { a, b } = start
  for (let it = 0; it < 100; it++) {
    const s = digamma(a + b)
    const f1 = digamma(a) - s - c1
    const f2 = digamma(b) - s - c2
    const t = trigamma(a + b)
    const j11 = trigamma(a) - t
    const j22 = trigamma(b) - t
    const det = j11 * j22 - t * t
    let da = (j22 * f1 + t * f2) / det
    let db = (t * f1 + j11 * f2) / det
    // Halve the step until both parameters stay positive.
    while (a - da <= 0 || b - db <= 0) {
      da /= 2
      db /= 2
    }
    a -= da
    b -= db
    if (Math.abs(da) + Math.abs(db) < 1e-12) break
  }
  return { a, b }
}

export type Projection = 'kl' | 'moments'

/**
 * Tilted distribution Beta(a, b) · (w p₁ + (1 − w) p₂), a two-component Beta mixture with weight π on Beta(a + 1, b),
 * projected back onto the Beta family.
 */
export function mixtureTilted(cavity: BetaParams, p1: number, p2: number, projection: Projection) {
  const { a, b } = cavity
  const pi = (p1 * a) / (p1 * a + p2 * b)
  if (projection === 'kl') {
    const s = digamma(a + b + 1)
    const c1 = pi * (digamma(a + 1) - s) + (1 - pi) * (digamma(a) - s)
    const c2 = pi * (digamma(b) - s) + (1 - pi) * (digamma(b + 1) - s)
    return { pi, next: solveBeta(c1, c2, cavity) }
  }
  const n1 = a + b + 1
  const m = (pi * (a + 1) + (1 - pi) * a) / n1
  const m2 = (pi * (a + 1) * (a + 2) + (1 - pi) * a * (a + 1)) / (n1 * (n1 + 1))
  const total = (m - m2) / (m2 - m * m)
  return { pi, next: { a: m * total, b: (1 - m) * total } }
}

export type MixtureStep = { sweep: number; site: number; ok: boolean; q: BetaParams; alpha: number[]; beta: number[] }

/** EP with Beta sites; the prior is Beta(1, 1). Updates whose cavity is not a proper Beta are skipped. */
export function mixtureEp(p1: number[], p2: number[], sweeps: number, projection: Projection): MixtureStep[] {
  const n = p1.length
  const alpha = new Array<number>(n).fill(0)
  const beta = new Array<number>(n).fill(0)
  const steps: MixtureStep[] = []
  const total = () => ({ a: 1 + alpha.reduce((s, v) => s + v, 0), b: 1 + beta.reduce((s, v) => s + v, 0) })
  for (let sweep = 0; sweep < sweeps; sweep++) {
    for (let i = 0; i < n; i++) {
      const q = total()
      const cavity = { a: q.a - alpha[i], b: q.b - beta[i] }
      if (!(cavity.a > 0 && cavity.b > 0)) {
        steps.push({ sweep, site: i, ok: false, q, alpha: [...alpha], beta: [...beta] })
        continue
      }
      const { next } = mixtureTilted(cavity, p1[i], p2[i], projection)
      alpha[i] = next.a - cavity.a
      beta[i] = next.b - cavity.b
      steps.push({ sweep, site: i, ok: true, q: total(), alpha: [...alpha], beta: [...beta] })
    }
  }
  return steps
}
