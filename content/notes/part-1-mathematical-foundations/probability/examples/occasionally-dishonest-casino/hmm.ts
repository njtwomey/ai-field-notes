/** A two-state hidden Markov model with six-sided dice: state 0 is the fair die, state 1 the loaded die. */
import { rng } from '@/lib/math'

export type Casino = {
  /** P(fair → loaded) per roll. */
  toLoaded: number
  /** P(loaded → fair) per roll. */
  toFair: number
  /** P(six | loaded). The other five faces share the rest equally. */
  loadedSix: number
}

export const transition = (m: Casino): number[][] => [
  [1 - m.toLoaded, m.toLoaded],
  [m.toFair, 1 - m.toFair],
]

/** Emission probability of face `x` (1 to 6) from state `s`. */
export const emission = (m: Casino, s: number, x: number): number =>
  s === 0 ? 1 / 6 : x === 6 ? m.loadedSix : (1 - m.loadedSix) / 5

/** The chain starts from its stationary distribution. */
export const initial = (m: Casino): number[] => {
  const loaded = m.toLoaded / (m.toLoaded + m.toFair)
  return [1 - loaded, loaded]
}

/** Draw `n` hidden states and rolls. */
export function simulate(m: Casino, n: number, seed: number): { states: number[]; rolls: number[] } {
  const { uniform } = rng(seed)
  const A = transition(m)
  const states: number[] = []
  const rolls: number[] = []
  let s = uniform() < initial(m)[1] ? 1 : 0
  for (let t = 0; t < n; t++) {
    if (t > 0) s = uniform() < A[s][1] ? 1 : 0
    states.push(s)
    // Inverse transform over the six faces.
    let u = uniform()
    let face = 1
    while (face < 6 && u >= emission(m, s, face)) {
      u -= emission(m, s, face)
      face++
    }
    rolls.push(face)
  }
  return { states, rolls }
}

/**
 * Forward–backward with per-step scaling. Returns P(state_t = loaded | all rolls) for every t and the log-likelihood
 * log P(rolls). Scaling each forward vector to sum to one keeps the numbers in range for long sequences.
 */
export function forwardBackward(m: Casino, rolls: number[]): { posterior: number[]; logLikelihood: number } {
  const A = transition(m)
  const n = rolls.length
  const alpha: number[][] = []
  const scale: number[] = []
  for (let t = 0; t < n; t++) {
    const prev = t === 0 ? null : alpha[t - 1]
    const a = [0, 1].map(
      (s) => (prev ? prev[0] * A[0][s] + prev[1] * A[1][s] : initial(m)[s]) * emission(m, s, rolls[t]),
    )
    const c = a[0] + a[1]
    scale.push(c)
    alpha.push([a[0] / c, a[1] / c])
  }
  const beta: number[][] = Array.from({ length: n }, () => [1, 1])
  for (let t = n - 2; t >= 0; t--) {
    beta[t] = [0, 1].map(
      (r) =>
        (A[r][0] * emission(m, 0, rolls[t + 1]) * beta[t + 1][0] +
          A[r][1] * emission(m, 1, rolls[t + 1]) * beta[t + 1][1]) /
        scale[t + 1],
    )
  }
  const posterior = alpha.map((a, t) => {
    const loaded = a[1] * beta[t][1]
    return loaded / (a[0] * beta[t][0] + loaded)
  })
  return { posterior, logLikelihood: scale.reduce((s, c) => s + Math.log(c), 0) }
}

/** Viterbi: the single most probable state sequence, by dynamic programming over log-probabilities. */
export function viterbi(m: Casino, rolls: number[]): number[] {
  const logA = transition(m).map((row) => row.map(Math.log))
  const n = rolls.length
  let score = [0, 1].map((s) => Math.log(initial(m)[s]) + Math.log(emission(m, s, rolls[0])))
  const back: number[][] = []
  for (let t = 1; t < n; t++) {
    const pointers: number[] = []
    score = [0, 1].map((s) => {
      const from0 = score[0] + logA[0][s]
      const from1 = score[1] + logA[1][s]
      pointers.push(from1 > from0 ? 1 : 0)
      return Math.max(from0, from1) + Math.log(emission(m, s, rolls[t]))
    })
    back.push(pointers)
  }
  const path = [score[1] > score[0] ? 1 : 0]
  for (let t = n - 2; t >= 0; t--) path.unshift(back[t][path[0]])
  return path
}
