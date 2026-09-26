/**
 * Linear-chain inference in the notation of Twomey, Diethe & Flach (2016): node potentials ψ_n (vectors over labels) and
 * edge potentials Ψ_n (matrices linking positions n and n+1). Positions are 0-based in code, 1-based in the notes.
 */
export type Vec = number[]
export type Mat = number[][]

const dot = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + v * b[i], 0)
const hadamard = (a: Vec, b: Vec) => a.map((v, i) => v * b[i])
/** Ψᵀγ: (Ψᵀγ)(v) = Σ_u Ψ(u, v) γ(u). */
const transposeTimes = (m: Mat, g: Vec) => m[0].map((_, v) => m.reduce((s, row, u) => s + row[v] * g[u], 0))
/** Ψδ: (Ψδ)(u) = Σ_v Ψ(u, v) δ(v). */
const times = (m: Mat, d: Vec) => m.map((row) => dot(row, d))

export type Inference = {
  alpha: Vec[]
  beta: Vec[]
  /** marginals[n][y] = P(Y_n = y | x). */
  marginals: Vec[]
  z: number
}

/**
 * Forward–backward: γ_{n−1} = α_{n−1} ⊙ ψ_{n−1}, α_n = Ψ_{n−1}ᵀ γ_{n−1}; δ_{n+1} = β_{n+1} ⊙ ψ_{n+1}, β_n = Ψ_n δ_{n+1};
 * α_1 = β_N = 1. The unnormalised marginal α_n ⊙ ψ_n ⊙ β_n sums to Z at every position.
 */
export function forwardBackward(psi: Vec[], edge: Mat[]): Inference {
  const n = psi.length
  const k = psi[0].length
  const alpha: Vec[] = [Array(k).fill(1)]
  for (let i = 1; i < n; i++) alpha.push(transposeTimes(edge[i - 1], hadamard(alpha[i - 1], psi[i - 1])))
  const beta: Vec[] = Array(n)
  beta[n - 1] = Array(k).fill(1)
  for (let i = n - 2; i >= 0; i--) beta[i] = times(edge[i], hadamard(beta[i + 1], psi[i + 1]))
  const unnormalised = psi.map((p, i) => hadamard(hadamard(alpha[i], p), beta[i]))
  const z = unnormalised[0].reduce((s, v) => s + v, 0)
  return { alpha, beta, marginals: unnormalised.map((u) => u.map((v) => v / z)), z }
}

/** The most probable label sequence (max-product in log space), and its log score log Π ψ Ψ. */
export function viterbi(psi: Vec[], edge: Mat[]): { path: number[]; score: number } {
  const n = psi.length
  const k = psi[0].length
  let best = psi[0].map(Math.log)
  const back: number[][] = []
  for (let i = 1; i < n; i++) {
    const next: Vec = []
    const arg: number[] = []
    for (let v = 0; v < k; v++) {
      let top = -Infinity
      let at = 0
      for (let u = 0; u < k; u++) {
        const s = best[u] + Math.log(edge[i - 1][u][v])
        if (s > top) {
          top = s
          at = u
        }
      }
      next.push(top + Math.log(psi[i][v]))
      arg.push(at)
    }
    back.push(arg)
    best = next
  }
  let last = best.indexOf(Math.max(...best))
  const score = best[last]
  const path = [last]
  for (let i = n - 2; i >= 0; i--) {
    last = back[i][last]
    path.unshift(last)
  }
  return { path, score }
}

/**
 * Best rank-1 approximation σ u vᵀ of a positive matrix, by power iteration on MᵀM. For a positive matrix the leading
 * singular vectors are positive (Perron–Frobenius), so the result is again a valid potential.
 */
export function rankOne(m: Mat): Mat {
  let v: Vec = m[0].map(() => 1)
  for (let it = 0; it < 200; it++) {
    const u = times(m, v)
    const w = transposeTimes(m, u)
    const norm = Math.hypot(...w)
    v = w.map((x) => x / norm)
  }
  const mv = times(m, v)
  const sigma = Math.hypot(...mv)
  const u = mv.map((x) => x / sigma)
  return u.map((ui) => v.map((vj) => sigma * ui * vj))
}
