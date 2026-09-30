/**
 * Algebraic Riccati equations and the linear-quadratic regulator. Continuous (CARE) by Kleinman's Newton iteration
 * or the matrix sign function of the Hamiltonian; discrete (DARE) by the Riccati recursion (value iteration) or the
 * structure-preserving doubling algorithm. Every solver is a traceable algorithm whose state carries the current P,
 * the gain K, the residual of the equation and a convergence flag.
 */

import { eig } from 'aifn/ode'
import { lstsq } from 'aifn/linalg'
import { toFlat, type Matrix, type Vector } from 'aifn/tensor'
import { run, type Algorithm } from 'aifn/trace'
import {
  add,
  allFiniteM,
  asM,
  block,
  eyeM,
  hcat,
  invM,
  maxAbsM,
  mul,
  scaleM,
  solveM,
  sub,
  symmetrise,
  toT,
  tr,
  vcat,
  type M,
} from './dense'
import { lyapunov } from './structure'
import type { StateSpace } from './system'
import type { MatrixLike } from './vector'

/** The data of a Riccati equation: dynamics (A, B) and quadratic costs xᵀQx + uᵀRu (Q ⪰ 0, R ≻ 0). */
export type RiccatiProblem = { A: MatrixLike; B: MatrixLike; Q: MatrixLike; R: MatrixLike }

/** Fields every Riccati solver state carries. */
export type RiccatiState = {
  step: number
  /** The current solution estimate P (n×n, symmetric). */
  P: Matrix
  /** The gain it gives: K = R⁻¹BᵀP (continuous) or (R + BᵀPB)⁻¹BᵀPA (discrete), m×n. */
  K: Matrix
  /** max |residual| of the Riccati equation at P, relative to 1 + max |P|. */
  residual: number
  /** max |P_t − P_{t−1}| (Infinity at step 0). */
  change: number
  converged: boolean
  /** Why the method cannot continue, or null: `'singular'`, `'not finite'`, `'not stabilisable'`. */
  failure: string | null
}

type Data = { A: M; B: M; Q: M; R: M; n: number; m: number; Rinv: M; G: M }

function problem({ A, B, Q, R }: RiccatiProblem, where: string): Data {
  const a = asM(A, `${where} A`)
  const b = asM(B, `${where} B`)
  const q = asM(Q, `${where} Q`)
  let r = asM(R, `${where} R`)
  const n = a.r
  const m = b.c
  if (a.c !== n || b.r !== n || q.r !== n || q.c !== n) throw new Error(`${where}: A, B and Q must be n×n, n×m, n×n`)
  if (r.r * r.c === m * m) r = { d: r.d, r: m, c: m }
  else throw new Error(`${where}: R must be ${m}×${m}`)
  const Rinv = invM(r)
  if (!Rinv) throw new Error(`${where}: R is singular (it must be positive definite)`)
  return { A: a, B: b, Q: q, R: r, n, m, Rinv, G: mul(mul(b, Rinv), tr(b)) }
}

/** The CARE residual AᵀP + PA − PBR⁻¹BᵀP + Q. */
function careResidual(d: Data, P: M): M {
  return add(add(add(mul(tr(d.A), P), mul(P, d.A)), scaleM(mul(mul(P, d.G), P), -1)), d.Q)
}

/** The DARE residual AᵀPA − P − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q, and the gain, or null when R + BᵀPB is singular. */
function dareParts(d: Data, P: M): { residual: M; K: M } | null {
  const BtP = mul(tr(d.B), P)
  const K = solveM(add(d.R, mul(BtP, d.B)), mul(BtP, d.A))
  if (!K) return null
  const AtP = mul(tr(d.A), P)
  const residual = add(sub(sub(mul(AtP, d.A), P), mul(mul(AtP, d.B), K)), d.Q)
  return { residual, K }
}

const relative = (res: M, P: M) => maxAbsM(res) / (1 + maxAbsM(P))

function careState(d: Data, step: number, P: M, prev: M | null, tol: number, failure: string | null): RiccatiState {
  const Ps = symmetrise(P)
  const residual = allFiniteM(Ps) ? relative(careResidual(d, Ps), Ps) : NaN
  const change = prev ? maxAbsM(sub(Ps, prev)) : Infinity
  return {
    step,
    P: toT(Ps),
    K: toT(mul(mul(d.Rinv, tr(d.B)), Ps)),
    residual,
    change,
    converged: failure === null && (residual <= tol || change <= tol * (1 + maxAbsM(Ps))),
    failure: failure ?? (allFiniteM(Ps) ? null : 'not finite'),
  }
}

const maxReal = (A: M) => Math.max(...toFlat(eig(toT(A), { vectors: false }).real))

/**
 * An initial stabilising gain K₀ for Kleinman's iteration by Bass's method (Armstrong, 1975, "An extension of Bass'
 * algorithm for stabilizing linear continuous constant systems", IEEE TAC 20(1)): with β > max Re λ(A), solve
 * (A + βI)Z + Z(A + βI)ᵀ = 2BBᵀ; then K₀ = BᵀZ⁻¹ makes A − BK₀ stable when (A, B) is controllable.
 */
function bassGain(d: Data): M | null {
  const beta = Math.max(0, maxReal(d.A)) + 1
  const Ab = add(d.A, scaleM(eyeM(d.n), beta))
  const Z = lyapunov(toT(Ab), toT(scaleM(mul(d.B, tr(d.B)), -2)))
  if (!Z.X) return null
  const Zi = invM(asM(Z.X, 'Z'))
  return Zi ? mul(tr(d.B), Zi) : null
}

/** Options for the Riccati solvers. */
export type RiccatiOptions = {
  /** Stop when the relative residual or the change in P is below this. Default 1e-12. */
  tol?: number
}

/**
 * Kleinman's Newton iteration for the CARE AᵀP + PA − PBR⁻¹BᵀP + Q = 0 (Kleinman, 1968, "On an iterative technique
 * for Riccati equation computations", IEEE TAC 13(1)): given a stabilising gain K_k, solve the Lyapunov equation
 * (A − BK_k)ᵀP + P(A − BK_k) + Q + K_kᵀRK_k = 0 (the cost of the policy u = −K_k x), then set K_{k+1} = R⁻¹BᵀP.
 * Each P is a policy evaluation, P decreases monotonically, and convergence is quadratic near the solution. `init`
 * takes `{ K0 }` (default: Bass's stabilising gain; K₀ = 0 when A is already stable).
 */
export function kleinman(
  prob: RiccatiProblem,
  options: RiccatiOptions = {},
): Algorithm<{ K0?: MatrixLike }, RiccatiState> {
  const d = problem(prob, 'kleinman')
  const tol = options.tol ?? 1e-12
  const evaluate = (K: M): M | null => {
    const Acl = sub(d.A, mul(d.B, K))
    const X = lyapunov(toT(tr(Acl)), toT(add(d.Q, mul(mul(tr(K), d.R), K))))
    return X.X ? asM(X.X, 'P') : null
  }
  const failed = (step: number, K: M, prev: M | null, failure: string): RiccatiState => ({
    step,
    P: toT(prev ?? { d: new Float64Array(d.n * d.n).fill(NaN), r: d.n, c: d.n }),
    K: toT(K),
    residual: NaN,
    change: NaN,
    converged: false,
    failure,
  })
  const fromGain = (step: number, K: M, prev: M | null): RiccatiState => {
    if (!allFiniteM(K) || maxReal(sub(d.A, mul(d.B, K))) >= 0) return failed(step, K, prev, 'not stabilisable')
    const P = evaluate(K)
    if (!P) return failed(step, K, prev, 'singular')
    return careState(d, step, P, prev, tol, null)
  }
  return {
    name: 'kleinman-care',
    init: ({ K0 } = {}) => {
      let K: M | null
      if (K0 !== undefined) {
        K = asM(K0, 'kleinman K0')
        if (K.r * K.c === d.m * d.n) K = { d: K.d, r: d.m, c: d.n }
      } else K = maxReal(d.A) < 0 ? { d: new Float64Array(d.m * d.n), r: d.m, c: d.n } : bassGain(d)
      if (!K) return failed(0, { d: new Float64Array(d.m * d.n).fill(NaN), r: d.m, c: d.n }, null, 'not stabilisable')
      return fromGain(0, K, null)
    },
    step: (s) => fromGain(s.step + 1, asM(s.K, 'K'), asM(s.P, 'P')),
    done: (s) => s.converged || s.failure !== null,
  }
}

/** The state of `hamiltonianSign`: a Riccati state plus the sign-function iterate. */
export type SignState = RiccatiState & {
  /** The iterate Z_k → sign(H), 2n×2n. */
  Z: Matrix
  /** The determinant scaling c_k = |det Z_k|^{1/2n}. */
  scaling: number
}

/**
 * The CARE by the matrix sign function of the Hamiltonian H = [[A, −BR⁻¹Bᵀ], [−Q, −Aᵀ]] (Roberts, 1971, "Linear
 * model reduction and solution of the algebraic Riccati equation by use of the sign function"; Byers, 1987, "Solving
 * the algebraic Riccati equation with the matrix sign function"). Newton's iteration Z ← (Z/c + cZ⁻¹)/2 with
 * determinant scaling c = |det Z|^{1/2n} converges quadratically to W = sign(H), which is −1 on the stable invariant
 * subspace span[I; P]; hence [W₁₂; W₂₂ + I] P = −[W₁₁ + I; W₂₁], solved by least squares. No eigenvectors are needed.
 * `init` takes `{}`.
 */
export function hamiltonianSign(prob: RiccatiProblem, options: RiccatiOptions = {}): Algorithm<object, SignState> {
  const d = problem(prob, 'hamiltonianSign')
  const tol = options.tol ?? 1e-12
  const n = d.n
  const H = vcat([hcat([d.A, scaleM(d.G, -1)]), hcat([scaleM(d.Q, -1), scaleM(tr(d.A), -1)])])
  const extract = (W: M): M | null => {
    const I = eyeM(n)
    const lhs = vcat([block(W, 0, n, n, 2 * n), add(block(W, n, 2 * n, n, 2 * n), I)])
    const rhs = scaleM(vcat([add(block(W, 0, n, 0, n), I), block(W, n, 2 * n, 0, n)]), -1)
    if (!allFiniteM(lhs) || !allFiniteM(rhs)) return null
    const x = lstsq(toT(lhs), toT(rhs)).x
    return asM(x, 'P')
  }
  const make = (step: number, Z: M, scaling: number, prev: M | null, change: number): SignState => {
    const P = extract(Z)
    const base = P ? careState(d, step, P, prev, tol, null) : careState(d, step, eyeM(n), prev, tol, 'not finite')
    // Converged when the sign iterate has settled or P already solves the CARE.
    const converged = base.failure === null && step > 0 && (change <= 1e-13 * (1 + maxAbsM(Z)) || base.residual <= tol)
    return { ...base, converged, Z: toT(Z), scaling }
  }
  return {
    name: 'hamiltonian-sign-care',
    init: () => make(0, H, 1, null, Infinity),
    step: (s) => {
      const Z = asM(s.Z, 'Z')
      const Zi = invM(Z)
      if (!Zi) return { ...s, step: s.step + 1, failure: 'singular' }
      const c = detScaling(Z)
      const next = scaleM(add(scaleM(Z, 1 / c), scaleM(Zi, c)), 0.5)
      return make(s.step + 1, next, c, asM(s.P, 'P'), maxAbsM(sub(next, Z)))
    },
    done: (s) => s.converged || s.failure !== null,
  }
}

/** |det Z|^{1/N} for an N×N matrix, from the eigenvalue moduli (a product of moduli, taken in log space). */
function detScaling(Z: M): number {
  const e = eig(toT(Z), { vectors: false })
  const re = toFlat(e.real)
  const im = toFlat(e.imag)
  let logAbs = 0
  for (let i = 0; i < re.length; i++) logAbs += Math.log(Math.hypot(re[i], im[i]))
  const c = Math.exp(logAbs / re.length)
  return Number.isFinite(c) && c > 0 ? c : 1
}

function dareState(d: Data, step: number, P: M, prev: M | null, tol: number): RiccatiState {
  const Ps = symmetrise(P)
  const parts = allFiniteM(Ps) ? dareParts(d, Ps) : null
  const residual = parts ? relative(parts.residual, Ps) : NaN
  const change = prev ? maxAbsM(sub(Ps, prev)) : Infinity
  const failure = !allFiniteM(Ps) ? 'not finite' : parts ? null : 'singular'
  return {
    step,
    P: toT(Ps),
    K: toT(parts ? parts.K : { d: new Float64Array(d.m * d.n).fill(NaN), r: d.m, c: d.n }),
    residual,
    change,
    converged: failure === null && (residual <= tol || change <= tol * (1 + maxAbsM(Ps))),
    failure,
  }
}

/**
 * The DARE P = Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA by the Riccati recursion (value iteration, Bellman's dynamic
 * programming backwards in time): P_t is the optimal cost-to-go of a horizon-t problem, starting from P₀ (default 0).
 * Converges linearly, at a rate set by the slowest closed-loop pole, when (A, B) is stabilisable and (A, Q^{1/2})
 * detectable. `init` takes `{ P0 }`.
 */
export function riccatiRecursion(
  prob: RiccatiProblem,
  options: RiccatiOptions = {},
): Algorithm<{ P0?: MatrixLike }, RiccatiState> {
  const d = problem(prob, 'riccatiRecursion')
  const tol = options.tol ?? 1e-12
  return {
    name: 'riccati-recursion-dare',
    init: ({ P0 } = {}) => dareState(d, 0, P0 === undefined ? scaleM(eyeM(d.n), 0) : asM(P0, 'P0'), null, tol),
    step: (s) => {
      const P = asM(s.P, 'P')
      const K = asM(s.K, 'K')
      const AtP = mul(tr(d.A), P)
      const next = add(d.Q, sub(mul(AtP, d.A), mul(mul(AtP, d.B), K)))
      return dareState(d, s.step + 1, next, P, tol)
    },
    done: (s) => s.converged || s.failure !== null,
  }
}

/** The state of `doubling`: a Riccati state plus the doubling iterates. */
export type DoublingState = RiccatiState & { Ak: Matrix; Gk: Matrix }

/**
 * The DARE by the structure-preserving doubling algorithm (Chu, Fan, Lin & Wang, 2004, "Structure-preserving
 * algorithms for periodic discrete-time algebraic Riccati equations", Int. J. Control 77(8)): with G = BR⁻¹Bᵀ and
 * W = (I + G_kH_k)⁻¹, A_{k+1} = A_k W A_k, G_{k+1} = G_k + A_k W G_k A_kᵀ, H_{k+1} = H_k + A_kᵀ H_k W A_k. H_k → P and
 * each step doubles the horizon (H_k is the cost-to-go of horizon 2^k), so convergence is quadratic. `init` takes `{}`.
 */
export function doubling(prob: RiccatiProblem, options: RiccatiOptions = {}): Algorithm<object, DoublingState> {
  const d = problem(prob, 'doubling')
  const tol = options.tol ?? 1e-12
  return {
    name: 'doubling-dare',
    init: () => ({ ...dareState(d, 0, d.Q, null, tol), Ak: toT(d.A), Gk: toT(d.G) }),
    step: (s) => {
      const A = asM(s.Ak, 'A')
      const G = asM(s.Gk, 'G')
      const Hm = asM(s.P, 'H')
      const W = invM(add(eyeM(d.n), mul(G, Hm)))
      if (!W) return { ...s, step: s.step + 1, failure: 'singular' }
      const AW = mul(A, W)
      const nextA = mul(AW, A)
      const nextG = add(G, mul(mul(AW, G), tr(A)))
      const nextH = add(Hm, mul(mul(mul(tr(A), Hm), W), A))
      const st = dareState(d, s.step + 1, nextH, Hm, tol)
      return { ...st, Ak: toT(nextA), Gk: toT(symmetrise(nextG)) }
    },
    done: (s) => s.converged || s.failure !== null,
  }
}

/** The solution of an LQR problem. */
export type LqrResult = {
  /** The stabilising Riccati solution P (the optimal cost is x₀ᵀPx₀). */
  P: Matrix
  /** The optimal gain: u = −Kx. */
  K: Matrix
  /** Eigenvalues of A − BK. */
  closedLoop: { real: Vector; imag: Vector }
  residual: number
  steps: number
  converged: boolean
  failure: string | null
}

function finishLqr(A: MatrixLike, B: MatrixLike, s: RiccatiState): LqrResult {
  const Acl = sub(asM(A, 'A'), mul(asM(B, 'B'), asM(s.K, 'K')))
  const cl = allFiniteM(Acl) ? eig(toT(Acl), { vectors: false }) : null
  const empty = toT({ d: new Float64Array(0), r: 0, c: 1 })
  return {
    P: s.P,
    K: s.K,
    closedLoop: cl ? { real: cl.real, imag: cl.imag } : { real: empty, imag: empty },
    residual: s.residual,
    steps: s.step,
    converged: s.converged,
    failure: s.failure,
  }
}

/**
 * The continuous-time LQR: minimise ∫₀^∞ (xᵀQx + uᵀRu) dt subject to x′ = Ax + Bu. The optimum is u = −Kx with
 * K = R⁻¹BᵀP and P the stabilising solution of the CARE (Kalman, 1960, "Contributions to the theory of optimal
 * control"). Solved by `kleinman` (default) or `hamiltonianSign`, for at most `maxSteps` (default 100) steps.
 */
export function lqr(
  sys: StateSpace | { A: MatrixLike; B: MatrixLike },
  Q: MatrixLike,
  R: MatrixLike,
  options: RiccatiOptions & { method?: 'kleinman' | 'sign'; maxSteps?: number } = {},
): LqrResult {
  const prob = { A: sys.A, B: sys.B, Q, R }
  const s =
    options.method === 'sign'
      ? run(hamiltonianSign(prob, options), {}, options.maxSteps ?? 100)
      : run(kleinman(prob, options), {}, options.maxSteps ?? 100)
  return finishLqr(sys.A, sys.B, s)
}

/**
 * The discrete-time LQR: minimise Σ (x_kᵀQx_k + u_kᵀRu_k) subject to x_{k+1} = Ax_k + Bu_k; u = −Kx with
 * K = (R + BᵀPB)⁻¹BᵀPA and P the stabilising DARE solution. Solved by `doubling` (default) or `riccatiRecursion`,
 * for at most `maxSteps` steps (default 100 for doubling, 10 000 for the recursion).
 */
export function dlqr(
  sys: StateSpace | { A: MatrixLike; B: MatrixLike },
  Q: MatrixLike,
  R: MatrixLike,
  options: RiccatiOptions & { method?: 'doubling' | 'recursion'; maxSteps?: number } = {},
): LqrResult {
  const prob = { A: sys.A, B: sys.B, Q, R }
  const s =
    options.method === 'recursion'
      ? run(riccatiRecursion(prob, options), {}, options.maxSteps ?? 10000)
      : run(doubling(prob, options), {}, options.maxSteps ?? 100)
  return finishLqr(sys.A, sys.B, s)
}
