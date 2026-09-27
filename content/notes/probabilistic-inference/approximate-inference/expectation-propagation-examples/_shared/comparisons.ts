/**
 * EP as message passing on a factor graph of paired comparisons: skills s_j ~ N(μ₀, σ₀²), and a game won by a over b
 * contributes the factor Φ((s_a − s_b)/σ_n). The approximation is fully factorised, one Gaussian per player, and each
 * game factor sends a Gaussian message to each of its two players.
 */
import { divide, toMoments, toNat, vFn, wFn, type Moments, type Nat } from './ep.ts'

export type Game = { winner: number; loser: number }

export type MessageStep = {
  sweep: number
  game: number
  /** Cavities of winner and loser: the prior times every other message to that player. */
  cavity: [Moments, Moments]
  t: number
  /** Marginals of every player after the update. */
  marginals: Moments[]
  /** The two messages just sent, to the winner and to the loser. */
  messages: [Nat, Nat]
}

export function comparisonEp(players: number, games: Game[], prior: Moments, noiseVar: number, sweeps: number) {
  const p0 = toNat(prior)
  // messages[g] = [to winner, to loser], natural parameters; all start at 1 (τ = ν = 0).
  const messages: [Nat, Nat][] = games.map(() => [
    { tau: 0, nu: 0 },
    { tau: 0, nu: 0 },
  ])
  const marginal = (j: number): Nat =>
    games.reduce(
      (acc, g, k) => {
        const m = g.winner === j ? messages[k][0] : g.loser === j ? messages[k][1] : null
        return m ? { tau: acc.tau + m.tau, nu: acc.nu + m.nu } : acc
      },
      { ...p0 },
    )
  const steps: MessageStep[] = []
  for (let sweep = 0; sweep < sweeps; sweep++) {
    games.forEach((g, k) => {
      const ca = toMoments(divide(marginal(g.winner), messages[k][0]))
      const cb = toMoments(divide(marginal(g.loser), messages[k][1]))
      // The factor depends on d = s_a − s_b, Gaussian under the cavities with variance σ_a² + σ_b²; adding σ_n² turns the
      // probit into a step on d plus noise. Each player's marginal moves along its own share of that variance.
      const c = Math.sqrt(ca.variance + cb.variance + noiseVar)
      const t = (ca.mean - cb.mean) / c
      const v = vFn(t)
      const w = wFn(t)
      const na = { mean: ca.mean + (ca.variance * v) / c, variance: ca.variance * (1 - (ca.variance * w) / (c * c)) }
      const nb = { mean: cb.mean - (cb.variance * v) / c, variance: cb.variance * (1 - (cb.variance * w) / (c * c)) }
      messages[k] = [divide(toNat(na), toNat(ca)), divide(toNat(nb), toNat(cb))]
      steps.push({
        sweep,
        game: k,
        cavity: [ca, cb],
        t,
        marginals: Array.from({ length: players }, (_, j) => toMoments(marginal(j))),
        messages: [messages[k][0], messages[k][1]],
      })
    })
  }
  return steps
}
