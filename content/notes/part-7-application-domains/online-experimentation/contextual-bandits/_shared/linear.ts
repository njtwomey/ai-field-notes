/**
 * Two-dimensional linear bandits for the figures in this category: ridge statistics, LinUCB, linear Thompson sampling
 * and the greedy ridge policy, run on seeded problems. Two dimensions keep every matrix 2×2 and closed-form, and let the
 * figures draw the parameter space.
 *
 * The random source is passed in (`rng(seed)` from '@/lib/math' in the widgets), so this module has no imports.
 */

export type Rand = { uniform: () => number; normal: () => number }
export type Vec = [number, number]

export type LinearPolicy = 'linucb' | 'lints' | 'greedy'
export type ArmMode = 'fixed' | 'context'

export const ARMS = 5
export const HORIZON = 300
export const NOISE = 0.3

/** The design matrix V = λI + Σ x xᵀ as [p, q, r] = [[p, q], [q, r]], and b = Σ r x. */
export type Ridge = { p: number; q: number; r: number; b: Vec }

export const initRidge = (lambda: number): Ridge => ({ p: lambda, q: 0, r: lambda, b: [0, 0] })

export function addObservation(s: Ridge, x: Vec, reward: number): Ridge {
  return {
    p: s.p + x[0] * x[0],
    q: s.q + x[0] * x[1],
    r: s.r + x[1] * x[1],
    b: [s.b[0] + reward * x[0], s.b[1] + reward * x[1]],
  }
}

/** V⁻¹ as [p, q, r]. */
export function inverse(s: Ridge): [number, number, number] {
  const det = s.p * s.r - s.q * s.q
  return [s.r / det, -s.q / det, s.p / det]
}

/** The ridge estimate θ̂ = V⁻¹ b. */
export function estimate(s: Ridge): Vec {
  const [a, c, d] = inverse(s)
  return [a * s.b[0] + c * s.b[1], c * s.b[0] + d * s.b[1]]
}

/** ‖x‖_{V⁻¹} = √(xᵀ V⁻¹ x), the width of the confidence set in direction x. */
export function width(s: Ridge, x: Vec): number {
  const [a, c, d] = inverse(s)
  return Math.sqrt(a * x[0] * x[0] + 2 * c * x[0] * x[1] + d * x[1] * x[1])
}

export const dot = (x: Vec, y: Vec) => x[0] * y[0] + x[1] * y[1]

/** Points on the ellipse {θ : (θ − c)ᵀ V (θ − c) = radius²}. */
export function ellipse(s: Ridge, centre: Vec, radius: number, points = 90): Vec[] {
  const mid = (s.p + s.r) / 2
  const rad = Math.hypot((s.p - s.r) / 2, s.q)
  const angle = 0.5 * Math.atan2(2 * s.q, s.p - s.r)
  const u: Vec = [Math.cos(angle), Math.sin(angle)]
  const a1 = radius / Math.sqrt(mid + rad)
  const a2 = radius / Math.sqrt(mid - rad)
  return Array.from({ length: points + 1 }, (_, k) => {
    const phi = (2 * Math.PI * k) / points
    const c1 = a1 * Math.cos(phi)
    const c2 = a2 * Math.sin(phi)
    return [centre[0] + c1 * u[0] - c2 * u[1], centre[1] + c1 * u[1] + c2 * u[0]]
  })
}

/** A draw from N(θ̂, v² V⁻¹), using the Cholesky factor of V⁻¹. */
export function samplePosterior(s: Ridge, v: number, r: Rand): Vec {
  const [a, c, d] = inverse(s)
  const l11 = Math.sqrt(a)
  const l21 = c / l11
  const l22 = Math.sqrt(Math.max(d - l21 * l21, 0))
  const z1 = r.normal()
  const z2 = r.normal()
  const m = estimate(s)
  return [m[0] + v * l11 * z1, m[1] + v * (l21 * z1 + l22 * z2)]
}

/** The arms offered in round t: fixed directions, or fresh random directions and lengths (a context). */
export function armsFor(mode: ArmMode, r: Rand): Vec[] {
  return Array.from({ length: ARMS }, (_, i): Vec => {
    if (mode === 'fixed') {
      const angle = (2 * Math.PI * i) / ARMS + 0.3
      return [Math.cos(angle), Math.sin(angle)]
    }
    const angle = 2 * Math.PI * r.uniform()
    const length = 0.5 + 0.5 * r.uniform()
    return [length * Math.cos(angle), length * Math.sin(angle)]
  })
}

export type Round = {
  /** Statistics before the round's choice. */
  stats: Ridge
  arms: Vec[]
  chosen: number
  /** Linear Thompson sampling's draw for this round. */
  draw?: Vec
}

export type Settings = { mode: ArmMode; lambda: number; alpha: number; v: number }

/**
 * Plays HORIZON rounds of one policy on the problem θ*. The arms and noise of run `seed` do not depend on the policy,
 * so policies can be compared on identical rounds. Returns the rounds (if asked) and the cumulative regret.
 */
export function play(
  policy: LinearPolicy,
  theta: Vec,
  settings: Settings,
  seed: number,
  makeRand: (seed: number) => Rand,
  keepRounds: boolean,
): { rounds: Round[]; regret: number[] } {
  const world = makeRand(seed * 7919 + 17)
  const agent = makeRand(seed * 104729 + 29)
  let stats = initRidge(settings.lambda)
  const rounds: Round[] = []
  const regret: number[] = []
  let cum = 0
  for (let t = 0; t < HORIZON; t++) {
    const arms = armsFor(settings.mode, world)
    const noise = world.normal() * NOISE
    let chosen = 0
    let draw: Vec | undefined
    if (policy === 'lints') {
      draw = samplePosterior(stats, settings.v, agent)
      const d = draw
      chosen = argmax(arms.map((x) => dot(x, d)))
    } else {
      const m = estimate(stats)
      const bonus = policy === 'linucb' ? settings.alpha : 0
      chosen = argmax(arms.map((x) => dot(x, m) + bonus * width(stats, x)))
    }
    if (keepRounds) rounds.push({ stats, arms, chosen, draw })
    const x = arms[chosen]
    const values = arms.map((a) => dot(a, theta))
    cum += Math.max(...values) - values[chosen]
    regret.push(cum)
    stats = addObservation(stats, x, values[chosen] + noise)
  }
  return { rounds, regret }
}

function argmax(xs: number[]): number {
  let best = 0
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[best]) best = i
  return best
}
