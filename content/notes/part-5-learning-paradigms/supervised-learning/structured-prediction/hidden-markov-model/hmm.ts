/**
 * A discrete-emission hidden Markov model in the chain notation of the structured-prediction notes: node potentials
 * ψ_n(v) = b_v(x_n) (with the initial distribution π folded into ψ_1), edge potentials Ψ_n = A, and forward and
 * backward vectors α_n = Ψᵀ(α_{n−1} ⊙ ψ_{n−1}), β_n = Ψ(β_{n+1} ⊙ ψ_{n+1}) with α_1 = β_N = 1.
 */

export type Hmm = {
  /** Initial distribution π over the K states. */
  initial: number[]
  /** Transition matrix A[u][v] = p(y_{n+1} = v | y_n = u). */
  transition: number[][]
  /** Emission probabilities emission[v][k] = p(x_n = k | y_n = v). */
  emission: number[][]
}

/** Node potentials ψ_n for an observation sequence, with π folded into ψ_1. */
export function nodePotentials(m: Hmm, xs: number[]): number[][] {
  return xs.map((x, n) => m.emission.map((row, v) => (n === 0 ? m.initial[v] : 1) * row[x]))
}

export type ForwardBackward = {
  /** α_n, rescaled to sum to one at each position (the scale does not change any ratio used below). */
  alpha: number[][]
  /** β_n, rescaled with the same constants as α so that α_n ⊙ ψ_n ⊙ β_n sums to one. */
  beta: number[][]
  /** Node potentials ψ_n. */
  psi: number[][]
  /** Filtering distribution p(y_n | x_1..x_n) ∝ α_n ⊙ ψ_n. */
  filtered: number[][]
  /** Smoothing distribution p(y_n | x) ∝ α_n ⊙ ψ_n ⊙ β_n. */
  marginal: number[][]
  /** log p(x) = log Z. */
  logZ: number
}

/**
 * Forward–backward with per-step scaling. With c_n the normaliser of α_n ⊙ ψ_n, log Z = Σ log c_n, and dividing β by
 * the same constants keeps every vector in range for long sequences.
 */
export function forwardBackward(m: Hmm, xs: number[]): ForwardBackward {
  const K = m.initial.length
  const N = xs.length
  const psi = nodePotentials(m, xs)
  const A = m.transition
  const alpha: number[][] = []
  const filtered: number[][] = []
  const scale: number[] = []
  for (let n = 0; n < N; n++) {
    // α_n = Aᵀ γ_{n−1}, with γ_{n−1} = α_{n−1} ⊙ ψ_{n−1} already normalised in `filtered`.
    const a =
      n === 0
        ? Array(K).fill(1)
        : Array.from({ length: K }, (_, v) => filtered[n - 1].reduce((s, g, u) => s + g * A[u][v], 0))
    const g = a.map((ai, v) => ai * psi[n][v])
    const c = g.reduce((s, gi) => s + gi, 0)
    scale.push(c)
    alpha.push(a)
    filtered.push(g.map((gi) => gi / c))
  }
  const beta: number[][] = Array.from({ length: N }, () => Array(K).fill(1))
  for (let n = N - 2; n >= 0; n--) {
    // β_n = A δ_{n+1}, with δ_{n+1} = β_{n+1} ⊙ ψ_{n+1}.
    const d = beta[n + 1].map((b, v) => b * psi[n + 1][v])
    beta[n] = Array.from({ length: K }, (_, u) => A[u].reduce((s, auv, v) => s + auv * d[v], 0) / scale[n + 1])
  }
  const marginal = filtered.map((f, n) => {
    const p = f.map((fi, v) => fi * beta[n][v])
    const z = p.reduce((s, pi) => s + pi, 0)
    return p.map((pi) => pi / z)
  })
  return { alpha, beta, psi, filtered, marginal, logZ: scale.reduce((s, c) => s + Math.log(c), 0) }
}

/** Viterbi: max-product with back-pointers, in log space. Returns the path and its log joint probability. */
export function viterbi(m: Hmm, xs: number[]): { path: number[]; logProbability: number } {
  const K = m.initial.length
  const logPsi = nodePotentials(m, xs).map((row) => row.map(Math.log))
  const logA = m.transition.map((row) => row.map(Math.log))
  let score = logPsi[0]
  const back: number[][] = []
  for (let n = 1; n < xs.length; n++) {
    const pointers: number[] = []
    score = Array.from({ length: K }, (_, v) => {
      let best = 0
      for (let u = 1; u < K; u++) if (score[u] + logA[u][v] > score[best] + logA[best][v]) best = u
      pointers.push(best)
      return score[best] + logA[best][v] + logPsi[n][v]
    })
    back.push(pointers)
  }
  let last = 0
  for (let v = 1; v < K; v++) if (score[v] > score[last]) last = v
  const path = [last]
  for (let n = back.length - 1; n >= 0; n--) path.unshift(back[n][path[0]])
  return { path, logProbability: score[last] }
}
