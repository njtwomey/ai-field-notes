/**
 * Chain models: hidden Markov models with scaled forward–backward, Viterbi, sampling and forward-filtering
 * backward-sampling (Rabiner 1989, "A tutorial on hidden Markov models", Proc. IEEE 77(2), §III; Durbin, Eddy, Krogh
 * & Mitchison 1998, "Biological Sequence Analysis", ch. 3), and generic log-space sum-product and max-product on a
 * chain of log-potentials, which the linear-chain CRF uses.
 *
 * Notation (Twomey, Diethe & Flach 2016): positions n = 0 … N − 1, states k = 0 … K − 1. α_n is the forward message
 * into position n and β_n the backward message; the posterior marginal is ∝ α_n ⊙ ψ_n ⊙ β_n with ψ_n the emission
 * likelihoods (π folded into ψ_0).
 */

import { categorical, type Stream } from 'aifn/random'
import { logSumExp } from 'aifn/special'
import { fromData, toRows, type Matrix, type Tensor, type Vector } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'

/** A discrete-emission HMM: π (K), A[u, v] = p(y_{n+1} = v | y_n = u) (K × K), B[k, m] = p(x = m | y = k) (K × M). */
export interface Hmm {
  initial: Vector
  transition: Matrix
  emission: Matrix
  /** Names for display. */
  stateNames?: readonly string[]
  symbolNames?: readonly string[]
}

type Rows = readonly (readonly number[])[]
const asVector = (v: Tensor | readonly number[]): Vector =>
  'shape' in v ? (v as Tensor) : fromData(Float64Array.from(v as readonly number[]), [(v as readonly number[]).length])
const asMatrix = (m: Tensor | Rows): Matrix =>
  'shape' in m
    ? (m as Tensor)
    : fromData(Float64Array.from((m as Rows).flat()), [(m as Rows).length, (m as Rows)[0].length])
const rowsOf = (m: Tensor | Rows): number[][] => ('shape' in m ? toRows(m as Tensor) : (m as Rows).map((r) => [...r]))
const matrix = (rows: readonly ArrayLike<number>[], k = rows[0]?.length ?? 0): Matrix => {
  const out = new Float64Array(rows.length * k)
  rows.forEach((r, i) => out.set(r, i * k))
  return fromData(out, [rows.length, k])
}

/** Build an HMM from arrays or tensors, checking that π and each row of A and B sum to one (to 1e-9). */
export function hmm(
  initial: Tensor | readonly number[],
  transition: Tensor | Rows,
  emission: Tensor | Rows,
  names: { stateNames?: readonly string[]; symbolNames?: readonly string[] } = {},
): Hmm {
  const pi = asVector(initial)
  const A = asMatrix(transition)
  const B = asMatrix(emission)
  const K = pi.shape[0]
  if (A.shape[0] !== K || A.shape[1] !== K || B.shape[0] !== K) throw new RangeError('hmm: inconsistent shapes')
  const sums = [[...pi.data], ...rowsOf(A), ...rowsOf(B)].map((r) => r.reduce((a, b) => a + b, 0))
  if (sums.some((s) => Math.abs(s - 1) > 1e-9)) throw new RangeError('hmm: π and the rows of A and B must sum to 1')
  return { initial: pi, transition: A, emission: B, ...names }
}

/**
 * The occasionally dishonest casino (Durbin et al. 1998, §3.2): a fair die (state 0) and a loaded die (state 1) that
 * rolls a six half the time; the casino switches from fair to loaded with probability 0.05 and back with 0.1. Symbols
 * 0 … 5 are the faces 1 … 6.
 */
export function dishonestCasino(): Hmm {
  return hmm(
    [0.5, 0.5],
    [
      [0.95, 0.05],
      [0.1, 0.9],
    ],
    [Array(6).fill(1 / 6), [...Array(5).fill(0.1), 0.5]],
    { stateNames: ['fair', 'loaded'], symbolNames: ['1', '2', '3', '4', '5', '6'] },
  )
}

/** The emission likelihoods ψ_n(k) = B[k, x_n] with π folded into ψ_0, as an N × K matrix. */
export function hmmNodePotentials(model: Hmm, observations: ArrayLike<number>): Matrix {
  const K = model.initial.shape[0]
  const B = rowsOf(model.emission)
  const pi = model.initial.data
  return matrix(
    Array.from(observations, (x, n) => Array.from({ length: K }, (_, k) => (n === 0 ? pi[k] : 1) * B[k][x])),
    K,
  )
}

/** The result of the scaled forward–backward recursions. */
export interface ForwardBackwardResult {
  /** α_n = Aᵀ γ_{n−1} (N × K), with γ the filtered distribution, so rows n ≥ 1 sum to one; α_0 = 1 (π is in ψ_0). */
  alpha: Matrix
  /** β_n (N × K), scaled by the same constants so that α_n ⊙ ψ_n ⊙ β_n sums to one; β_{N−1} = 1. */
  beta: Matrix
  /** ψ_n (N × K). */
  psi: Matrix
  /** Filtering distributions p(y_n | x_{0..n}) (N × K). */
  filtered: Matrix
  /** Smoothing marginals p(y_n | x) (N × K). */
  marginals: Matrix
  /** Pairwise marginals ξ_n(u, v) = p(y_n = u, y_{n+1} = v | x), (N − 1) × K × K. */
  pairwise: Tensor
  /** c_n, the normaliser of α_n ⊙ ψ_n (length N); log p(x) = Σ log c_n. */
  scale: Vector
  logLikelihood: number
}

/**
 * Forward–backward with per-step scaling (Rabiner 1989, §V.A): α_n = Aᵀ γ_{n−1} with γ_{n−1} the filtered
 * distribution, c_n = Σ α_n ⊙ ψ_n, and β_n = A (β_{n+1} ⊙ ψ_{n+1}) / c_{n+1}. log p(x) = Σ log c_n.
 */
export function forwardBackward(model: Hmm, observations: ArrayLike<number>): ForwardBackwardResult {
  const psi = toRows(hmmNodePotentials(model, observations))
  return scaledForwardBackward(psi, rowsOf(model.transition))
}

function scaledForwardBackward(psi: number[][], A: number[][]): ForwardBackwardResult {
  const N = psi.length
  const K = A.length
  const alpha: number[][] = []
  const filtered: number[][] = []
  const scale: number[] = []
  for (let n = 0; n < N; n++) {
    const a =
      n === 0
        ? Array(K).fill(1)
        : Array.from({ length: K }, (_, v) => filtered[n - 1].reduce((s, g, u) => s + g * A[u][v], 0))
    const g = a.map((ai, v) => ai * psi[n][v])
    const c = g.reduce((s, x) => s + x, 0)
    scale.push(c)
    alpha.push(a)
    filtered.push(g.map((x) => x / c))
  }
  const beta: number[][] = Array.from({ length: N }, () => Array(K).fill(1))
  for (let n = N - 2; n >= 0; n--) {
    const d = beta[n + 1].map((b, v) => b * psi[n + 1][v])
    beta[n] = Array.from({ length: K }, (_, u) => A[u].reduce((s, auv, v) => s + auv * d[v], 0) / scale[n + 1])
  }
  const marginals = filtered.map((f, n) => {
    const p = f.map((fi, v) => fi * beta[n][v])
    const z = p.reduce((s, x) => s + x, 0)
    return p.map((x) => x / z)
  })
  const pairwise = new Float64Array(Math.max(N - 1, 0) * K * K)
  for (let n = 0; n + 1 < N; n++) {
    let z = 0
    for (let u = 0; u < K; u++)
      for (let v = 0; v < K; v++) {
        const x = filtered[n][u] * A[u][v] * psi[n + 1][v] * beta[n + 1][v]
        pairwise[(n * K + u) * K + v] = x
        z += x
      }
    for (let i = 0; i < K * K; i++) pairwise[n * K * K + i] /= z
  }
  return {
    alpha: matrix(alpha, K),
    beta: matrix(beta, K),
    psi: matrix(psi, K),
    filtered: matrix(filtered, K),
    marginals: matrix(marginals, K),
    pairwise: fromData(pairwise, [Math.max(N - 1, 0), K, K]),
    scale: fromData(Float64Array.from(scale), [N]),
    logLikelihood: scale.reduce((s, c) => s + Math.log(c), 0),
  }
}

/** The result of Viterbi decoding. */
export interface ViterbiResult {
  /** The most probable state path (int32, length N). */
  path: Vector
  /** Its log-probability (log joint p(x, y) for an HMM; the unnormalised log score for a chain of potentials). */
  logProbability: number
  /** δ_n(k): the best log score of a path ending in k at n (N × K). */
  delta: Matrix
  /** Back-pointers: the best predecessor of k at n (int32, N × K; row 0 is −1). */
  backpointers: Tensor
}

/**
 * Max-product on a chain in log space: logUnary (N × K) and logPairwise (K × K shared, or (N − 1) × K × K).
 * δ_0 = u_0, δ_n(v) = max_u δ_{n−1}(u) + P(u, v) + u_n(v); ties go to the smaller state.
 */
export function chainViterbi(logUnary: Matrix, logPairwise: Tensor): ViterbiResult {
  const [N, K] = logUnary.shape
  const U = toRows(logUnary)
  const pair = pairwiseAt(logPairwise, K)
  const delta: number[][] = [U[0]]
  const back = new Int32Array(N * K).fill(-1)
  for (let n = 1; n < N; n++) {
    const prev = delta[n - 1]
    const P = pair(n - 1)
    delta.push(
      Array.from({ length: K }, (_, v) => {
        let best = 0
        for (let u = 1; u < K; u++) if (prev[u] + P[u][v] > prev[best] + P[best][v]) best = u
        back[n * K + v] = best
        return prev[best] + P[best][v] + U[n][v]
      }),
    )
  }
  let last = 0
  for (let k = 1; k < K; k++) if (delta[N - 1][k] > delta[N - 1][last]) last = k
  const path = new Int32Array(N)
  path[N - 1] = last
  for (let n = N - 1; n > 0; n--) path[n - 1] = back[n * K + path[n]]
  return {
    path: fromData(path, [N]),
    logProbability: delta[N - 1][last],
    delta: matrix(delta, K),
    backpointers: fromData(back, [N, K]),
  }
}

/** The pairwise log-potential matrix between positions n and n + 1. */
function pairwiseAt(logPairwise: Tensor, K: number): (n: number) => number[][] {
  if (logPairwise.shape.length === 2) {
    const P = toRows(logPairwise)
    return () => P
  }
  const all = Array.from(logPairwise.data)
  return (n) => Array.from({ length: K }, (_, u) => all.slice((n * K + u) * K, (n * K + u + 1) * K))
}

/** Viterbi decoding of an HMM: the most probable hidden path and log p(x, y*). */
export function viterbi(model: Hmm, observations: ArrayLike<number>): ViterbiResult {
  const logPsi = matrix(toRows(hmmNodePotentials(model, observations)).map((r) => r.map(Math.log)))
  const logA = matrix(rowsOf(model.transition).map((r) => r.map(Math.log)))
  return chainViterbi(logPsi, logA)
}

/** The result of log-space forward–backward on a chain of potentials. */
export interface ChainMarginals {
  /** log α_n (N × K): log Σ over paths to n, excluding u_n. */
  logAlpha: Matrix
  /** log β_n (N × K): log Σ over paths from n, excluding u_n. */
  logBeta: Matrix
  marginals: Matrix
  /** (N − 1) × K × K. */
  pairwise: Tensor
  logZ: number
}

/**
 * Sum-product on a chain in log space: log α_0 = 0, log α_n(v) = logsumexp_u [log α_{n−1}(u) + u_{n−1}(u) + P(u, v)],
 * log β_{N−1} = 0, log β_n(u) = logsumexp_v [P(u, v) + u_{n+1}(v) + log β_{n+1}(v)], log Z = logsumexp(α + u + β).
 */
export function chainForwardBackward(logUnary: Matrix, logPairwise: Tensor): ChainMarginals {
  const [N, K] = logUnary.shape
  const U = toRows(logUnary)
  const pair = pairwiseAt(logPairwise, K)
  const lse = (xs: number[]) => logSumExp(fromData(Float64Array.from(xs), [xs.length])) as number
  const la: number[][] = [Array(K).fill(0)]
  for (let n = 1; n < N; n++) {
    const P = pair(n - 1)
    la.push(Array.from({ length: K }, (_, v) => lse(la[n - 1].map((a, u) => a + U[n - 1][u] + P[u][v]))))
  }
  const lb: number[][] = Array.from({ length: N }, () => Array(K).fill(0))
  for (let n = N - 2; n >= 0; n--) {
    const P = pair(n)
    lb[n] = Array.from({ length: K }, (_, u) => lse(lb[n + 1].map((b, v) => P[u][v] + U[n + 1][v] + b)))
  }
  const logZ = lse(la[0].map((a, k) => a + U[0][k] + lb[0][k]))
  const marginals = la.map((row, n) => row.map((a, k) => Math.exp(a + U[n][k] + lb[n][k] - logZ)))
  const pairwise = new Float64Array(Math.max(N - 1, 0) * K * K)
  for (let n = 0; n + 1 < N; n++) {
    const P = pair(n)
    for (let u = 0; u < K; u++)
      for (let v = 0; v < K; v++)
        pairwise[(n * K + u) * K + v] = Math.exp(la[n][u] + U[n][u] + P[u][v] + U[n + 1][v] + lb[n + 1][v] - logZ)
  }
  return {
    logAlpha: matrix(la, K),
    logBeta: matrix(lb, K),
    marginals: matrix(marginals, K),
    pairwise: fromData(pairwise, [Math.max(N - 1, 0), K, K]),
    logZ,
  }
}

// ── Stepping ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link forwardBackwardSteps} and {@link viterbiSteps}. */
export interface ChainStepOptions {
  model: Hmm
  observations: ArrayLike<number>
}

/**
 * Forward–backward stepped one position at a time: steps 1 … N fill the forward rows (α_n and the filtered
 * distribution), steps N + 1 … 2N − 1 the backward rows from the end; then the marginals are complete. Rows not yet
 * computed are NaN.
 */
export interface ForwardBackwardState {
  model: Hmm
  observations: Int32Array
  phase: 'forward' | 'backward' | 'done'
  /** The position computed by the last step (−1 before the first). */
  position: number
  alpha: Matrix
  filtered: Matrix
  beta: Matrix
  marginals: Matrix
  /** log p(x_0 … x_n) up to the last forward position. */
  logLikelihood: number
  /** The full result, computed once at init; the partial matrices above reveal it row by row. */
  full: ForwardBackwardResult
}

function reveal(full: Matrix, rows: (n: number) => boolean): Matrix {
  const [N, K] = full.shape
  const out = Float64Array.from(full.data)
  for (let n = 0; n < N; n++) if (!rows(n)) out.fill(NaN, n * K, (n + 1) * K)
  return fromData(out, [N, K])
}

function fbState(
  base: Omit<ForwardBackwardState, 'alpha' | 'filtered' | 'beta' | 'marginals' | 'logLikelihood'>,
): ForwardBackwardState {
  const N = base.observations.length
  const f = base.full
  const fwd = base.phase === 'forward' ? base.position : N - 1
  const bwdFrom = base.phase === 'forward' ? N : base.phase === 'backward' ? base.position : 0
  let ll = 0
  for (let n = 0; n <= fwd; n++) ll += Math.log(f.scale.data[n])
  return {
    ...base,
    alpha: reveal(f.alpha, (n) => n <= fwd),
    filtered: reveal(f.filtered, (n) => n <= fwd),
    beta: reveal(f.beta, (n) => n >= bwdFrom),
    marginals: reveal(f.marginals, (n) => n >= bwdFrom),
    logLikelihood: ll,
  }
}

/** Forward–backward as a traceable algorithm (see {@link ForwardBackwardState}). */
export const forwardBackwardSteps: Algorithm<ChainStepOptions, ForwardBackwardState> = {
  name: 'pgm.forward-backward',
  init: ({ model, observations }) =>
    fbState({
      model,
      observations: Int32Array.from(observations),
      phase: 'forward',
      position: -1,
      full: forwardBackward(model, observations),
    }),
  step: (s) => {
    const N = s.observations.length
    if (s.phase === 'done') return s
    if (s.phase === 'forward') {
      const position = s.position + 1
      if (position < N - 1) return fbState({ ...s, position })
      return fbState({ ...s, phase: N > 1 ? 'backward' : 'done', position: N - 1 })
    }
    const position = s.position - 1
    return fbState({ ...s, position, phase: position <= 0 ? 'done' : 'backward' })
  },
  done: (s) => s.phase === 'done',
}

/** Viterbi stepped one position at a time: δ rows fill forwards, then the back-trace fills the path backwards. */
export interface ViterbiState {
  model: Hmm
  observations: Int32Array
  phase: 'forward' | 'backtrack' | 'done'
  position: number
  /** δ rows computed so far (NaN elsewhere). */
  delta: Matrix
  /** The path entries recovered so far (−1 elsewhere). */
  path: Vector
  full: ViterbiResult
}

/** Viterbi as a traceable algorithm (see {@link ViterbiState}). */
export const viterbiSteps: Algorithm<ChainStepOptions, ViterbiState> = {
  name: 'pgm.viterbi',
  init: ({ model, observations }) => {
    const full = viterbi(model, observations)
    const N = full.path.shape[0]
    return {
      model,
      observations: Int32Array.from(observations),
      phase: 'forward',
      position: 0,
      delta: reveal(full.delta, (n) => n === 0),
      path: fromData(new Int32Array(N).fill(-1), [N]),
      full,
    }
  },
  step: (s) => {
    const N = s.observations.length
    if (s.phase === 'done') return s
    if (s.phase === 'forward') {
      const position = s.position + 1
      if (position < N) return { ...s, position, delta: reveal(s.full.delta, (n) => n <= position) }
      const path = Int32Array.from(s.path.data)
      path[N - 1] = s.full.path.data[N - 1]
      return { ...s, phase: N > 1 ? 'backtrack' : 'done', position: N - 1, path: fromData(path, [N]) }
    }
    const position = s.position - 1
    const path = Int32Array.from(s.path.data)
    path[position] = s.full.path.data[position]
    return { ...s, position, path: fromData(path, [N]), phase: position === 0 ? 'done' : 'backtrack' }
  },
  done: (s) => s.phase === 'done',
}

// ── Sampling ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** Draw a state path and observations of length n from an HMM (int32 vectors). */
export function sampleHmm(s: Stream, model: Hmm, n: number): { states: Vector; observations: Vector } {
  const A = rowsOf(model.transition)
  const B = rowsOf(model.emission)
  const states = new Int32Array(n)
  const obs = new Int32Array(n)
  for (let t = 0; t < n; t++) {
    const st = s.child('step', t)
    states[t] =
      t === 0 ? categorical(st.child('state'), model.initial.data) : categorical(st.child('state'), A[states[t - 1]])
    obs[t] = categorical(st.child('symbol'), B[states[t]])
  }
  return { states: fromData(states, [n]), observations: fromData(obs, [n]) }
}

/**
 * Forward-filtering backward-sampling (Carter & Kohn 1994; Frühwirth-Schnatter 1994): a draw of the hidden path from
 * p(y | x), using the filtered distributions and y_n | y_{n+1} ∝ γ_n(y_n) A(y_n, y_{n+1}).
 */
export function sampleHiddenPath(s: Stream, model: Hmm, observations: ArrayLike<number>): Vector {
  const fb = forwardBackward(model, observations)
  const F = toRows(fb.filtered)
  const A = rowsOf(model.transition)
  const N = F.length
  const path = new Int32Array(N)
  path[N - 1] = categorical(s.child(N - 1), F[N - 1])
  for (let n = N - 2; n >= 0; n--)
    path[n] = categorical(
      s.child(n),
      F[n].map((g, u) => g * A[u][path[n + 1]]),
    )
  return fromData(path, [N])
}
