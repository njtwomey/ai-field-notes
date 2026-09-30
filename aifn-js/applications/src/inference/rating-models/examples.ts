import type { Status } from 'aifn/foundation/contracts'
import {
  normalCdf,
  normalQuantile,
  truncatedNormalV,
  truncatedNormalVDraw,
  truncatedNormalW,
  truncatedNormalWDraw,
} from 'aifn/numerics/special'
import { fromData, type Tensor } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import { dist, model, type Model } from 'aifn/inference/model'

// ── TrueSkill ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** A skill belief N(mean, sd²). */
export interface Rating {
  mean: number
  sd: number
}

/** TrueSkill's published defaults: μ₀ = 25, σ₀ = 25/3, β = 25/6, τ = 25/300, draw probability 0.1. */
export const TRUESKILL_DEFAULTS = { mean: 25, sd: 25 / 3, beta: 25 / 6, tau: 25 / 300, drawProbability: 0.1 } as const

/**
 * The draw margin ε for a draw probability p between two equally skilled players: p = 2Φ(ε/(√n β)) − 1, so
 * ε = Φ⁻¹((p + 1)/2) √n β with n players (2).
 */
export function drawMargin(drawProbability: number, beta: number, players = 2): number {
  return (normalQuantile((drawProbability + 1) / 2) as number) * Math.sqrt(players) * beta
}

/** Options of a TrueSkill update. */
export interface TrueSkillOptions {
  beta?: number
  /** Dynamics noise added to each variance before the game (default 25/300; 0 in EP over a fixed match set). */
  tau?: number
  /** ε (default from the draw probability 0.1). */
  drawMargin?: number
}

/** The result of one two-player TrueSkill update, with the intermediate quantities. */
export interface TrueSkillUpdate {
  player1: Rating
  player2: Rating
  /** c² = 2β² + σ₁² + σ₂² (after τ), and t = μ₁ − μ₂. */
  c: number
  t: number
  /** The mean and variance corrections v and w. */
  v: number
  w: number
  /** The probability of the observed outcome under the prior. */
  probability: number
}

/**
 * One TrueSkill update of a two-player game from player 1's side (Herbrich et al. 2007; Moser 2010, eqs. for
 * v_win, w_win, v_draw, w_draw). With σᵢ² ← σᵢ² + τ², c² = 2β² + σ₁² + σ₂², t = μ₁ − μ₂ and ε the draw margin:
 * a win gives v = v_win(t/c − ε/c), w = v(v + t/c − ε/c); μ₁ += σ₁² v/c, μ₂ −= σ₂² v/c, σᵢ² ← σᵢ²(1 − σᵢ² w/c²).
 * A loss is a win for player 2; a draw uses v_draw and w_draw on the interval |d| < ε.
 */
export function trueSkillUpdate(
  r1: Rating,
  r2: Rating,
  outcome: 'win' | 'loss' | 'draw',
  options: TrueSkillOptions = {},
): TrueSkillUpdate {
  const beta = options.beta ?? TRUESKILL_DEFAULTS.beta
  const tau = options.tau ?? TRUESKILL_DEFAULTS.tau
  const eps = options.drawMargin ?? drawMargin(TRUESKILL_DEFAULTS.drawProbability, beta)
  const var1 = r1.sd ** 2 + tau ** 2
  const var2 = r2.sd ** 2 + tau ** 2
  const c = Math.sqrt(2 * beta * beta + var1 + var2)
  const t = r1.mean - r2.mean
  let v: number
  let w: number
  let probability: number
  let sign = 1
  if (outcome === 'draw') {
    v = truncatedNormalVDraw(t / c, eps / c) as number
    w = truncatedNormalWDraw(t / c, eps / c) as number
    probability = (normalCdf((eps - t) / c) as number) - (normalCdf((-eps - t) / c) as number)
  } else {
    sign = outcome === 'win' ? 1 : -1
    const u = (sign * t) / c - eps / c
    v = truncatedNormalV(u) as number
    w = truncatedNormalW(u) as number
    probability = normalCdf(u) as number
  }
  return {
    player1: { mean: r1.mean + (sign * var1 * v) / c, sd: Math.sqrt(var1 * (1 - (var1 / (c * c)) * w)) },
    player2: { mean: r2.mean - (sign * var2 * v) / c, sd: Math.sqrt(var2 * (1 - (var2 / (c * c)) * w)) },
    c,
    t,
    v,
    w,
    probability,
  }
}

/**
 * TrueSkill's structure in the model language (Herbrich, Minka and Graepel 2007, fig. 1, with the performances
 * integrated out): a plate of P players with skills s_p ~ N(μ₀, σ₀²), and a plate of G games, each naming its winner
 * w_g and loser l_g (per-game constants) and observing y_g = 1 ~ Bernoulli(Φ((s_{w_g} − s_{l_g})/(√2 β))), the
 * probability that the winner's performance N(s_w, β²) exceeds the loser's. Draws (the margin ε, |d| < ε) have no
 * family in the model language; `trueSkillEp` handles them with its truncated-Gaussian match factor.
 */
export function trueSkillModel(options: { mean?: number; sd?: number; beta?: number } = {}): Model {
  const { mean = TRUESKILL_DEFAULTS.mean, sd = TRUESKILL_DEFAULTS.sd, beta = TRUESKILL_DEFAULTS.beta } = options
  return model('TrueSkill', (m) => {
    const players = m.plate('players', 'P', { label: 'P', index: 'p' })
    const games = m.plate('games', 'G', { label: 'G', index: 'g' })
    const skill = players.variable('s', dist.Normal(mean, sd), { label: 's_p' })
    const winner = games.constant('w', undefined, { label: 'w_g' })
    const loser = games.constant('l', undefined, { label: 'l_g' })
    const d = games.deterministic('d', 'difference', [skill.at(winner), skill.at(loser)], { label: 'd_g' })
    const z = games.deterministic('z', 'product', [d, 1 / (Math.SQRT2 * beta)], { label: 'z_g' })
    const p = games.deterministic('π', 'probit', [z], { label: '\\pi_g' })
    games.observed('y', dist.Bernoulli(p), { label: 'y_g' })
  })
}

/** A match between two players: the first won, or it was a draw. */
export interface Match {
  winner: number
  loser: number
  draw?: boolean
}

/** Options of {@link trueSkillEp}. */
export interface TrueSkillEpOptions {
  /** Prior ratings, one per player. */
  players: readonly Rating[]
  matches: readonly Match[]
  beta?: number
  drawMargin?: number
  /** Weight of the old site, in [0, 1). Default 0. */
  damping?: number
  tolerance?: number
}

/** The state of EP over a fixed set of matches (skills constant over the matches, as in TrueSkill through time). */
export interface TrueSkillEpState extends Status {
  /** Match updates done. */
  t: number
  /** Site precisions and shifts per match: [M, 2] (column 0 on the first-listed player, 1 on the second). */
  sitePrecision: Tensor
  siteShift: Tensor
  /** Posterior means and standard deviations per player. */
  means: Tensor
  sds: Tensor
  /** The match updated last (−1 at the start), and the update's details. */
  match: number
  update: TrueSkillUpdate | null
  ok: boolean
  sweep: number
  position: number
  change: number
  sweepChange: number
  lastSweepChange: number
  converged: boolean
}

function marginals(players: readonly Rating[], matches: readonly Match[], tau: Float64Array, nu: Float64Array) {
  const P = players.map((r) => 1 / r.sd ** 2)
  const N = players.map((r) => r.mean / r.sd ** 2)
  matches.forEach((m, k) => {
    P[m.winner] += tau[2 * k]
    N[m.winner] += nu[2 * k]
    P[m.loser] += tau[2 * k + 1]
    N[m.loser] += nu[2 * k + 1]
  })
  return { P, N }
}

/**
 * EP for TrueSkill over a fixed set of matches: one match per step. The cavity of each player removes the match's
 * site; the match factor is applied by the two-player update (with τ = 0); the new sites are the updated ratings
 * divided by the cavities. The first sweep is TrueSkill's online (ADF) pass without dynamics; later sweeps let early
 * matches learn from later ones (Herbrich, Minka and Graepel, 2007; Dangauthier et al., 2008). No start.
 */
export function trueSkillEp(o: TrueSkillEpOptions): Algorithm<void, TrueSkillEpState> {
  const { players, matches } = o
  const M = matches.length
  const beta = o.beta ?? TRUESKILL_DEFAULTS.beta
  const margin = o.drawMargin ?? drawMargin(TRUESKILL_DEFAULTS.drawProbability, beta)
  const damping = o.damping ?? 0
  const tolerance = o.tolerance ?? 1e-8
  return {
    name: 'ep.trueskill',
    init: () => {
      const tau = new Float64Array(2 * M)
      const { P, N } = marginals(players, matches, tau, tau)
      return {
        t: 0,
        sitePrecision: fromData(new Float64Array(2 * M), [M, 2]),
        siteShift: fromData(new Float64Array(2 * M), [M, 2]),
        means: fromData(
          Float64Array.from(P, (p, i) => N[i] / p),
          [P.length],
        ),
        sds: fromData(
          Float64Array.from(P, (p) => Math.sqrt(1 / p)),
          [P.length],
        ),
        match: -1,
        update: null,
        ok: true,
        sweep: 0,
        position: 0,
        change: 0,
        sweepChange: 0,
        lastSweepChange: Infinity,
        converged: M === 0,
      }
    },
    step: (s) => {
      if (s.converged) return { ...s, t: s.t + 1 }
      const k = s.position
      const m = matches[k]
      const tau = Float64Array.from(s.sitePrecision.data)
      const nu = Float64Array.from(s.siteShift.data)
      const { P, N } = marginals(players, matches, tau, nu)
      const cav = [
        { p: P[m.winner] - tau[2 * k], n: N[m.winner] - nu[2 * k] },
        { p: P[m.loser] - tau[2 * k + 1], n: N[m.loser] - nu[2 * k + 1] },
      ]
      let ok = cav.every((c) => c.p > 0)
      let change = 0
      let update: TrueSkillUpdate | null = null
      if (ok) {
        update = trueSkillUpdate(
          { mean: cav[0].n / cav[0].p, sd: Math.sqrt(1 / cav[0].p) },
          { mean: cav[1].n / cav[1].p, sd: Math.sqrt(1 / cav[1].p) },
          m.draw ? 'draw' : 'win',
          { beta, tau: 0, drawMargin: margin },
        )
        const after = [update.player1, update.player2]
        after.forEach((r, j) => {
          const nt = 1 / r.sd ** 2 - cav[j].p
          const nn = r.mean / r.sd ** 2 - cav[j].n
          const dt = (1 - damping) * nt + damping * tau[2 * k + j]
          const dn = (1 - damping) * nn + damping * nu[2 * k + j]
          change = Math.max(change, Math.abs(dt - tau[2 * k + j]), Math.abs(dn - nu[2 * k + j]))
          tau[2 * k + j] = dt
          nu[2 * k + j] = dn
        })
        ok = Number.isFinite(change)
      }
      const mg = marginals(players, matches, tau, nu)
      let position = k + 1
      let { sweep, sweepChange, lastSweepChange } = s
      let converged: boolean = s.converged
      sweepChange = Math.max(sweepChange, change)
      if (position >= matches.length) {
        converged = sweepChange < tolerance
        lastSweepChange = sweepChange
        sweepChange = 0
        position = 0
        sweep++
      }
      return {
        ...s,
        t: s.t + 1,
        sitePrecision: fromData(tau, s.sitePrecision.shape),
        siteShift: fromData(nu, s.siteShift.shape),
        means: fromData(
          Float64Array.from(mg.P, (p, i) => mg.N[i] / p),
          [mg.P.length],
        ),
        sds: fromData(
          Float64Array.from(mg.P, (p) => Math.sqrt(1 / p)),
          [mg.P.length],
        ),
        match: k,
        update,
        ok,
        sweep,
        position,
        change,
        sweepChange,
        lastSweepChange,
        converged,
      }
    },
  }
}
