/**
 * EP's worked examples: Minka's clutter problem (Minka 2001, UAI, §4), TrueSkill for two-player matches (Herbrich,
 * Minka & Graepel 2007, "TrueSkill: a Bayesian skill rating system", NIPS; the update formulas as in Moser 2010,
 * "Computing your skill"), and the Bayes point machine (Herbrich, Graepel & Campbell 2001, JMLR 1; trained by EP as in
 * Minka 2001, thesis §5).
 */

import { normal, type Stream } from 'aifn/random'
import {
  normalCdf,
  normalQuantile,
  truncatedNormalV,
  truncatedNormalVDraw,
  truncatedNormalW,
  truncatedNormalWDraw,
} from 'aifn/special'
import { fromData, toRows, type Matrix, type Tensor, type Vector } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { inverse } from 'aifn/linalg'
import type { EpOptions } from './ep'
import { clutterTilted, probitTilted, stepTilted, tiltedByQuadrature, type Tilted } from './tilted'

// ── The clutter problem ─────────────────────────────────────────────────────────────────────────────────────────────

/** Minka's clutter problem: θ ~ N(0, priorVariance); each xᵢ is N(θ, 1) with probability 1 − w, else N(0, c²). */
export interface ClutterProblem {
  /** w, the clutter probability. */
  weight: number
  /** Default 100. */
  priorVariance?: number
  /** Default 10. */
  clutterVariance?: number
}

const defaults = (p: ClutterProblem) => ({
  weight: p.weight,
  priorVariance: p.priorVariance ?? 100,
  clutterVariance: p.clutterVariance ?? 10,
})

/** log p(x | θ) = Σᵢ log[(1 − w) N(xᵢ; θ, 1) + w N(xᵢ; 0, c²)]. */
export function clutterLogLikelihood(theta: number, x: ArrayLike<number>, problem: ClutterProblem): number {
  const { weight, clutterVariance } = defaults(problem)
  let total = 0
  for (let i = 0; i < x.length; i++) {
    const s = ((1 - weight) * Math.exp(-0.5 * (x[i] - theta) ** 2)) / Math.sqrt(2 * Math.PI)
    const c = (weight * Math.exp((-0.5 * x[i] ** 2) / clutterVariance)) / Math.sqrt(2 * Math.PI * clutterVariance)
    total += Math.log(s + c)
  }
  return total
}

/** Draw n observations of the clutter problem at a true θ (float64 vector). */
export function sampleClutter(s: Stream, n: number, theta: number, problem: ClutterProblem): Vector {
  const { weight, clutterVariance } = defaults(problem)
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const r = s.child(i)
    out[i] =
      r.child('which').uniform() < weight
        ? normal(r.child('x'), 0, Math.sqrt(clutterVariance))
        : normal(r.child('x'), theta, 1)
  }
  return fromData(out, [n])
}

/**
 * EP options for the clutter problem, ready for `expectationPropagationSteps`: closed-form tilted moments for α = 1,
 * quadrature for power EP.
 */
export function clutterEp(
  x: ArrayLike<number>,
  problem: ClutterProblem,
  options: Pick<EpOptions, 'damping' | 'power' | 'order' | 'tolerance'> = {},
): EpOptions {
  const p = defaults(problem)
  const xs = Array.from(x)
  const factor = (i: number) => (theta: number) => clutterLogLikelihood(theta, [xs[i]], p)
  return {
    prior: { mean: 0, variance: p.priorVariance },
    factors: xs.length,
    tilted: (i, c, power) =>
      power === 1
        ? clutterTilted(xs[i], c.mean, c.variance, { weight: p.weight, clutterVariance: p.clutterVariance })
        : tiltedByQuadrature(c.mean, c.variance, factor(i), { power }),
    ...options,
  }
}

/**
 * The exact posterior of the clutter problem on an even grid (trapezoid rule): density, mean, variance and
 * log evidence. The grid must hold the posterior mass; the default ±40 does for the usual prior.
 */
export function clutterPosterior(
  x: ArrayLike<number>,
  problem: ClutterProblem,
  { lower = -40, upper = 40, points = 4001 }: { lower?: number; upper?: number; points?: number } = {},
): { grid: Vector; density: Vector; mean: number; variance: number; logEvidence: number } {
  const p = defaults(problem)
  const h = (upper - lower) / (points - 1)
  const grid = Float64Array.from({ length: points }, (_, i) => lower + i * h)
  const logs = grid.map(
    (t) =>
      -0.5 * Math.log(2 * Math.PI * p.priorVariance) - (0.5 * t * t) / p.priorVariance + clutterLogLikelihood(t, x, p),
  )
  const top = Math.max(...logs)
  const f = logs.map((l) => Math.exp(l - top))
  const trap = (g: ArrayLike<number>) => {
    let s = 0
    for (let i = 0; i < g.length; i++) s += g[i] * (i === 0 || i === g.length - 1 ? 0.5 : 1)
    return s * h
  }
  const Z = trap(f)
  const density = f.map((v) => v / Z)
  const mean = trap(density.map((d, i) => d * grid[i]))
  const variance = trap(density.map((d, i) => d * (grid[i] - mean) ** 2))
  return {
    grid: fromData(grid, [points]),
    density: fromData(density, [points]),
    mean,
    variance,
    logEvidence: top + Math.log(Z),
  }
}

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

/** A match between two players: the first won, or it was a draw. */
export interface Match {
  winner: number
  loser: number
  draw?: boolean
}

/** Options of {@link trueSkillEpSteps}. */
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
export interface TrueSkillEpState {
  players: readonly Rating[]
  matches: readonly Match[]
  beta: number
  drawMargin: number
  damping: number
  tolerance: number
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
 * matches learn from later ones.
 */
export const trueSkillEpSteps: Algorithm<TrueSkillEpOptions, TrueSkillEpState> = {
  name: 'ep.trueskill',
  init: (o) => {
    const M = o.matches.length
    const beta = o.beta ?? TRUESKILL_DEFAULTS.beta
    const tau = new Float64Array(2 * M)
    const { P, N } = marginals(o.players, o.matches, tau, tau)
    return {
      players: o.players,
      matches: o.matches,
      beta,
      drawMargin: o.drawMargin ?? drawMargin(TRUESKILL_DEFAULTS.drawProbability, beta),
      damping: o.damping ?? 0,
      tolerance: o.tolerance ?? 1e-8,
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
    if (s.converged) return s
    const k = s.position
    const m = s.matches[k]
    const tau = Float64Array.from(s.sitePrecision.data)
    const nu = Float64Array.from(s.siteShift.data)
    const { P, N } = marginals(s.players, s.matches, tau, nu)
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
        { beta: s.beta, tau: 0, drawMargin: s.drawMargin },
      )
      const after = [update.player1, update.player2]
      after.forEach((r, j) => {
        const nt = 1 / r.sd ** 2 - cav[j].p
        const nn = r.mean / r.sd ** 2 - cav[j].n
        const dt = (1 - s.damping) * nt + s.damping * tau[2 * k + j]
        const dn = (1 - s.damping) * nn + s.damping * nu[2 * k + j]
        change = Math.max(change, Math.abs(dt - tau[2 * k + j]), Math.abs(dn - nu[2 * k + j]))
        tau[2 * k + j] = dt
        nu[2 * k + j] = dn
      })
      ok = Number.isFinite(change)
    }
    const mg = marginals(s.players, s.matches, tau, nu)
    let position = k + 1
    let { sweep, sweepChange, lastSweepChange } = s
    let converged: boolean = s.converged
    sweepChange = Math.max(sweepChange, change)
    if (position >= s.matches.length) {
      converged = sweepChange < s.tolerance
      lastSweepChange = sweepChange
      sweepChange = 0
      position = 0
      sweep++
    }
    return {
      ...s,
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
  done: (s) => s.converged,
}

// ── Bayes point machine ─────────────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link bayesPointMachineSteps}. */
export interface BayesPointMachineOptions {
  /** Inputs, n × d (append a column of ones for a bias). */
  x: Matrix | readonly (readonly number[])[]
  /** Labels ±1, length n. */
  y: ArrayLike<number>
  /** Prior w ~ N(0, priorVariance I). Default 1. */
  priorVariance?: number
  /** `step` (noise-free: 𝟙(y wᵀx > 0)) or `probit` (Φ(y wᵀx / s)). Default `probit`. */
  likelihood?: 'step' | 'probit'
  /** s² for the probit likelihood. Default 0.1. */
  noiseVariance?: number
  damping?: number
  tolerance?: number
}

/** The state of EP for the Bayes point machine. */
export interface BayesPointMachineState {
  x: number[][]
  y: number[]
  priorVariance: number
  likelihood: 'step' | 'probit'
  noiseVariance: number
  damping: number
  tolerance: number
  /** One-dimensional sites on the projections sᵢ = wᵀxᵢ: precision and shift (length n). */
  sitePrecision: Tensor
  siteShift: Tensor
  /** The posterior over w: mean (the Bayes point) and covariance. */
  mean: Vector
  covariance: Matrix
  /** The point updated last, its cavity on sᵢ and tilted moments. */
  point: number
  cavity: { mean: number; variance: number }
  tilted: Tilted
  ok: boolean
  sweep: number
  position: number
  change: number
  sweepChange: number
  lastSweepChange: number
  converged: boolean
}

/** Σ = (I/σ₀² + Σᵢ τᵢ xᵢxᵢᵀ)⁻¹ and m = Σ Σᵢ νᵢ xᵢ. */
function bpmPosterior(x: number[][], tau: Float64Array, nu: Float64Array, priorVariance: number) {
  const d = x[0]?.length ?? 0
  const A = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1 / priorVariance : 0)))
  const b = new Array<number>(d).fill(0)
  x.forEach((xi, i) => {
    for (let p = 0; p < d; p++) {
      b[p] += nu[i] * xi[p]
      for (let q = 0; q < d; q++) A[p][q] += tau[i] * xi[p] * xi[q]
    }
  })
  const inv = toRows(inverse(matrixOf(A)))
  const mean = inv.map((row) => row.reduce((s, v, j) => s + v * b[j], 0))
  return { mean, covariance: inv }
}

const matrixOf = (rows: number[][]): Matrix =>
  fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])

/**
 * The Bayes point machine by EP (Minka 2001, thesis §5.2): each point's likelihood acts on w only through
 * sᵢ = wᵀxᵢ, so its site is a one-dimensional Gaussian on sᵢ. An update projects q(w) onto sᵢ (mean mᵀxᵢ, variance
 * xᵢᵀΣxᵢ), removes the site, applies the step or probit factor, and refits the site. One point per step.
 */
export const bayesPointMachineSteps: Algorithm<BayesPointMachineOptions, BayesPointMachineState> = {
  name: 'ep.bayes-point-machine',
  init: (o) => {
    const x = 'shape' in o.x ? toRows(o.x as Matrix) : (o.x as readonly (readonly number[])[]).map((r) => [...r])
    const n = x.length
    const priorVariance = o.priorVariance ?? 1
    const zero = new Float64Array(n)
    const post = bpmPosterior(x, zero, zero, priorVariance)
    return {
      x,
      y: Array.from(o.y),
      priorVariance,
      likelihood: o.likelihood ?? 'probit',
      noiseVariance: o.noiseVariance ?? 0.1,
      damping: o.damping ?? 0,
      tolerance: o.tolerance ?? 1e-8,
      sitePrecision: fromData(new Float64Array(n), [n]),
      siteShift: fromData(new Float64Array(n), [n]),
      mean: fromData(Float64Array.from(post.mean), [post.mean.length]),
      covariance: matrixOf(post.covariance),
      point: -1,
      cavity: { mean: NaN, variance: NaN },
      tilted: { logZ: NaN, mean: NaN, variance: NaN },
      ok: true,
      sweep: 0,
      position: 0,
      change: 0,
      sweepChange: 0,
      lastSweepChange: Infinity,
      converged: n === 0,
    }
  },
  step: (s) => {
    if (s.converged) return s
    const i = s.position
    const xi = s.x[i]
    const S = toRows(s.covariance)
    const m = Array.from(s.mean.data)
    const tau = Float64Array.from(s.sitePrecision.data)
    const nu = Float64Array.from(s.siteShift.data)
    // The marginal of sᵢ = wᵀxᵢ under q, then its cavity.
    const Sx = S.map((row) => row.reduce((acc, v, j) => acc + v * xi[j], 0))
    const vi = Sx.reduce((acc, v, j) => acc + v * xi[j], 0)
    const mi = m.reduce((acc, v, j) => acc + v * xi[j], 0)
    const ct = 1 / vi - tau[i]
    const cn = mi / vi - nu[i]
    let ok = ct > 0
    let change = 0
    let cavity = { mean: NaN, variance: NaN }
    let t: Tilted = { logZ: NaN, mean: NaN, variance: NaN }
    if (ok) {
      cavity = { mean: cn / ct, variance: 1 / ct }
      t =
        s.likelihood === 'step'
          ? stepTilted(s.y[i] * cavity.mean, cavity.variance)
          : probitTilted(cavity.mean, cavity.variance, s.y[i], { noiseVariance: s.noiseVariance })
      // The step factor is written on y·s; map its mean back to s.
      if (s.likelihood === 'step') t = { ...t, mean: s.y[i] * t.mean }
      ok = t.variance > 0
    }
    if (ok) {
      const nt = 1 / t.variance - ct
      const nn = t.mean / t.variance - cn
      const dt = (1 - s.damping) * nt + s.damping * tau[i]
      const dn = (1 - s.damping) * nn + s.damping * nu[i]
      change = Math.max(Math.abs(dt - tau[i]), Math.abs(dn - nu[i]))
      tau[i] = dt
      nu[i] = dn
    }
    const post = bpmPosterior(s.x, tau, nu, s.priorVariance)
    let position = i + 1
    let { sweep, sweepChange, lastSweepChange } = s
    let converged: boolean = s.converged
    sweepChange = Math.max(sweepChange, change)
    if (position >= s.x.length) {
      converged = sweepChange < s.tolerance
      lastSweepChange = sweepChange
      sweepChange = 0
      position = 0
      sweep++
    }
    return {
      ...s,
      sitePrecision: fromData(tau, [tau.length]),
      siteShift: fromData(nu, [nu.length]),
      mean: fromData(Float64Array.from(post.mean), [post.mean.length]),
      covariance: matrixOf(post.covariance),
      point: i,
      cavity,
      tilted: t,
      ok,
      sweep,
      position,
      change,
      sweepChange,
      lastSweepChange,
      converged,
    }
  },
  done: (s) => s.converged,
}

/**
 * Predictive probabilities P(y = +1 | x) under the BPM posterior: Φ(mᵀx / √(xᵀΣx + s²)) for the probit likelihood,
 * Φ(mᵀx / √(xᵀΣx)) for the step. `points` is n × d; returns a length-n vector.
 */
export function bayesPointMachinePredict(
  s: BayesPointMachineState,
  points: Matrix | readonly (readonly number[])[],
): Vector {
  const X = 'shape' in points ? toRows(points as Matrix) : (points as readonly (readonly number[])[])
  const S = toRows(s.covariance)
  const m = s.mean.data
  const noise = s.likelihood === 'probit' ? s.noiseVariance : 0
  const out = Float64Array.from(X, (x) => {
    let mu = 0
    let v = 0
    for (let p = 0; p < x.length; p++) {
      mu += m[p] * x[p]
      for (let q = 0; q < x.length; q++) v += x[p] * S[p][q] * x[q]
    }
    return normalCdf(mu / Math.sqrt(v + noise)) as number
  })
  return fromData(out, [out.length])
}
