/**
 * Skill-rating arithmetic shared by the skill-rating notes: Gaussian tails with good relative accuracy, the TrueSkill
 * v and w functions, a two-player TrueSkill update, Elo, Bradley–Terry by Zermelo's MM iteration, and simulated
 * match streams. Everything is small enough to recompute on every slider move.
 */
import { rng } from '@/lib/math'
import { normalQuantile } from '@/lib/math/special'

/** Standard normal density. */
export const phi = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)

/**
 * Complementary error function with relative error below 1.2e-7 everywhere (Numerical Recipes' erfcc). The
 * site-wide `normalCdf` has absolute, not relative, accuracy, which is not enough for Φ deep in the lower tail, where
 * TrueSkill divides by it after a large upset.
 */
function erfc(x: number): number {
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

/** Standard normal cdf Φ. */
export const Phi = (z: number) => 0.5 * erfc(-z / Math.SQRT2)

/** Mean correction for a win: v(x, a) = φ(x − a)/Φ(x − a), the mean of a standard normal truncated below at a − x. */
export function vWin(x: number, a: number): number {
  const u = x - a
  const den = Phi(u)
  // Far in the tail φ(u)/Φ(u) → −u; guard the division.
  return den < 1e-300 ? -u : phi(u) / den
}

/** Variance correction for a win: w = v (v + x − a), in (0, 1). */
export function wWin(x: number, a: number): number {
  const v = vWin(x, a)
  return v * (v + x - a)
}

/** Mean correction for a draw, |d| ≤ a in standardised units. */
export function vDraw(x: number, a: number): number {
  const den = Phi(a - x) - Phi(-a - x)
  if (den < 1e-300) return x < 0 ? -x - a : -x + a
  return (phi(-a - x) - phi(a - x)) / den
}

/** Variance correction for a draw. */
export function wDraw(x: number, a: number): number {
  const den = Phi(a - x) - Phi(-a - x)
  if (den < 1e-300) return 1
  const v = vDraw(x, a)
  return v * v + ((a - x) * phi(a - x) + (a + x) * phi(a + x)) / den
}

export type Rating = { mu: number; sigma: number }
export type Outcome = 'win' | 'draw' | 'loss'

export const TS_DEFAULTS = { mu: 25, sigma: 25 / 3, beta: 25 / 6, tau: 25 / 300, drawProbability: 0.1 }

/** Draw margin ε for a draw probability between two equally skilled players with no skill uncertainty. */
export function drawMargin(pDraw: number, beta: number, players = 2): number {
  if (pDraw <= 0) return 0
  return normalQuantile((pDraw + 1) / 2) * Math.sqrt(players) * beta
}

export type TrueSkillStep = {
  /** Prior variances after the dynamics noise τ² is added. */
  var1: number
  var2: number
  /** Performance-difference prior d = p1 − p2 ~ N(t, c²). */
  t: number
  c: number
  eps: number
  v: number
  w: number
  /** Moments of d after truncation by the outcome. */
  dMean: number
  dVar: number
  /** Probability of the observed outcome under the prior. */
  pOutcome: number
  p1: Rating
  p2: Rating
}

/** One TrueSkill update of a two-player game, from player 1's point of view. */
export function trueSkill1v1(
  r1: Rating,
  r2: Rating,
  outcome: Outcome,
  { beta, tau, eps }: { beta: number; tau: number; eps: number },
): TrueSkillStep {
  const var1 = r1.sigma ** 2 + tau ** 2
  const var2 = r2.sigma ** 2 + tau ** 2
  const c = Math.sqrt(2 * beta * beta + var1 + var2)
  const t = r1.mu - r2.mu
  const a = eps / c
  let v: number
  let w: number
  let pOutcome: number
  // A loss for player 1 is a win for player 2: flip the sign of the difference and of the mean update.
  const sign = outcome === 'loss' ? -1 : 1
  if (outcome === 'draw') {
    v = vDraw(t / c, a)
    w = wDraw(t / c, a)
    pOutcome = Phi(a - t / c) - Phi(-a - t / c)
  } else {
    v = vWin((sign * t) / c, a)
    w = wWin((sign * t) / c, a)
    pOutcome = Phi((sign * t) / c - a)
  }
  const dMean = t + sign * c * v
  const dVar = c * c * (1 - w)
  return {
    var1,
    var2,
    t,
    c,
    eps,
    v,
    w,
    dMean,
    dVar,
    pOutcome,
    p1: { mu: r1.mu + (sign * var1 * v) / c, sigma: Math.sqrt(var1 * (1 - (var1 / (c * c)) * w)) },
    p2: { mu: r2.mu - (sign * var2 * v) / c, sigma: Math.sqrt(var2 * (1 - (var2 / (c * c)) * w)) },
  }
}

/** Gaussian density N(x; m, s²). */
export const gauss = (x: number, m: number, s: number) => phi((x - m) / s) / s

/** Elo-style expected score on the TrueSkill scale: P(1 beats 2) = Φ((r1 − r2) / (√2 β)), Elo's Gaussian form. */
export const eloExpectedGaussian = (r1: number, r2: number, beta: number) => Phi((r1 - r2) / (Math.SQRT2 * beta))

/** Logistic Elo expected score on the usual 400-point scale. */
export const eloExpected = (r1: number, r2: number) => 1 / (1 + 10 ** (-(r1 - r2) / 400))

export type Game = { i: number; j: number; /** 1 if i won, 0 if j won. */ y: number }

/** A stream of games between random pairs, with outcomes drawn from the Thurstone model with performance noise β. */
export function simulateGames(skills: number[], n: number, beta: number, seed: number): Game[] {
  const r = rng(seed)
  const games: Game[] = []
  const m = skills.length
  for (let g = 0; g < n; g++) {
    const i = Math.floor(r.uniform() * m)
    let j = Math.floor(r.uniform() * (m - 1))
    if (j >= i) j += 1
    const pi = skills[i] + beta * r.normal()
    const pj = skills[j] + beta * r.normal()
    games.push({ i, j, y: pi > pj ? 1 : 0 })
  }
  return games
}

/**
 * Zermelo's iteration (an MM algorithm) for Bradley–Terry strengths. `wins[i][j]` counts wins of i over j. Returns the
 * log-strengths θ after each iteration, centred to mean zero, starting from all zeros.
 */
export function bradleyTerryMM(wins: number[][], iterations: number): number[][] {
  const m = wins.length
  let p = Array(m).fill(1)
  const history = [p.map(() => 0)]
  const W = wins.map((row) => row.reduce((a, b) => a + b, 0))
  for (let k = 0; k < iterations; k++) {
    const next = p.map((pi, i) => {
      let den = 0
      for (let j = 0; j < m; j++) if (j !== i) den += (wins[i][j] + wins[j][i]) / (pi + p[j])
      // A player with no wins has no finite MLE; keep a tiny strength so the others stay defined.
      return den > 0 ? Math.max(W[i], 1e-9) / den : pi
    })
    const logs = next.map(Math.log)
    const centre = logs.reduce((a, b) => a + b, 0) / m
    p = logs.map((l) => Math.exp(l - centre))
    history.push(logs.map((l) => l - centre))
  }
  return history
}

/** Bradley–Terry log-likelihood of a win matrix at log-strengths θ. */
export function bradleyTerryLogLik(wins: number[][], theta: number[]): number {
  let ll = 0
  for (let i = 0; i < wins.length; i++)
    for (let j = 0; j < wins.length; j++) {
      if (i === j || wins[i][j] === 0) continue
      ll += wins[i][j] * (theta[i] - Math.log(Math.exp(theta[i]) + Math.exp(theta[j])))
    }
  return ll
}
