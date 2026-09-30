/**
 * Structural properties of linear systems: controllability and observability (Kalman's rank tests), Lyapunov
 * equations and Gramians, and single-input pole placement by Ackermann's formula.
 */

import { eig } from 'aifn/ode'
import { toFlat, type Matrix, type Vector } from 'aifn/tensor'
import { add, asM, eyeM, hcat, kronM, mul, rankM, solveM, sub, symmetrise, toT, toV, tr, zerosM, type M } from './dense'
import type { StateSpace } from './system'
import type { MatrixLike, VectorLike } from './vector'

const matrices = (sys: StateSpace) => ({ A: asM(sys.A, 'A'), B: asM(sys.B, 'B'), C: asM(sys.C, 'C') })

/** The Krylov matrix [B, AB, A²B, …, A^{n−1}B] (n×nm). */
function krylov(A: M, B: M): M {
  const blocks = [B]
  for (let k = 1; k < A.r; k++) blocks.push(mul(A, blocks[k - 1]))
  return hcat(blocks)
}

/** A rank test's result. */
export type RankTest = {
  /** The test matrix. */
  matrix: Matrix
  /** Its numerical rank and singular values (descending). */
  rank: number
  singularValues: Vector
  /** True when the rank equals the state dimension n. */
  full: boolean
}

const rankTest = (m: M, n: number, tol?: number): RankTest => {
  const { rank, singularValues } = rankM(m, tol)
  return { matrix: toT(m), rank, singularValues: toV(singularValues), full: rank === n }
}

/**
 * Kalman's controllability test: the pair (A, B) is controllable (every state reachable from the origin) iff
 * 𝒞 = [B, AB, …, A^{n−1}B] has rank n (Kalman, 1960, "On the general theory of control systems"). Rank by SVD with
 * numpy's default tolerance unless `tol` is given; the smallest singular value says how nearly uncontrollable it is.
 */
export function controllability(sys: StateSpace, { tol }: { tol?: number } = {}): RankTest {
  const { A, B } = matrices(sys)
  return rankTest(krylov(A, B), sys.states, tol)
}

/**
 * Kalman's observability test: (A, C) is observable (the initial state is determined by the output) iff
 * 𝒪 = [C; CA; …; CA^{n−1}] has rank n. The dual of controllability: 𝒪 = 𝒞(Aᵀ, Cᵀ)ᵀ.
 */
export function observability(sys: StateSpace, { tol }: { tol?: number } = {}): RankTest {
  const { A, C } = matrices(sys)
  return rankTest(tr(krylov(tr(A), tr(C))), sys.states, tol)
}

/** The solution of a Lyapunov equation, or `singular` when the equation has no unique solution. */
export type LyapunovSolution = { X: Matrix | null; singular: boolean }

/**
 * Solves the continuous Lyapunov equation A X + X Aᵀ + Q = 0 (the Gramian convention; scipy's
 * `solve_continuous_lyapunov(A, −Q)`), or with `discrete` the Stein equation X = A X Aᵀ + Q. Solved directly as the
 * n²×n² linear system (I ⊗ A + A ⊗ I) vec X = −vec Q, or (I − A ⊗ A) vec X = vec Q, which suits the small n here
 * (Bartels–Stewart is the method for large n). Unique iff no two eigenvalues of A sum to 0 (continuous) or multiply
 * to 1 (discrete); otherwise `singular`.
 */
export function lyapunov(
  a: MatrixLike,
  q: MatrixLike,
  { discrete = false }: { discrete?: boolean } = {},
): LyapunovSolution {
  const A = asM(a, 'lyapunov A')
  const Q = asM(q, 'lyapunov Q')
  const n = A.r
  if (A.c !== n || Q.r !== n || Q.c !== n) throw new Error('lyapunov: A and Q must be square and the same size')
  const I = eyeM(n)
  // Row-major vec: vec(A X) = (A ⊗ I) vec X and vec(X Aᵀ) = (I ⊗ A) vec X.
  const K = discrete ? sub(eyeM(n * n), kronM(A, A)) : add(kronM(A, I), kronM(I, A))
  const rhs = { d: discrete ? Float64Array.from(Q.d) : Q.d.map((v) => -v), r: n * n, c: 1 }
  const x = solveM(K, rhs)
  if (!x) return { X: null, singular: true }
  return { X: toT(symmetrise({ d: x.d, r: n, c: n })), singular: false }
}

/** A Gramian, or null when the system is not stable (the defining integral diverges). */
export type Gramian = { W: Matrix | null; stable: boolean }

/**
 * The controllability Gramian W_c = ∫₀^∞ e^{At}BBᵀe^{Aᵀt} dt (or Σ A^k BBᵀ A^{kᵀ} for a discrete system), which
 * solves A W + W Aᵀ + BBᵀ = 0. It exists for a stable A; xᵀ W_c⁻¹ x is the least input energy that reaches x.
 */
export function controllabilityGramian(sys: StateSpace): Gramian {
  const { A, B } = matrices(sys)
  return gramian(A, mul(B, tr(B)), sys.dt !== null)
}

/** The observability Gramian W_o, solving Aᵀ W + W A + CᵀC = 0 (discrete: W = AᵀWA + CᵀC). */
export function observabilityGramian(sys: StateSpace): Gramian {
  const { A, C } = matrices(sys)
  return gramian(tr(A), mul(tr(C), C), sys.dt !== null)
}

function gramian(A: M, Q: M, discrete: boolean): Gramian {
  const e = eig(toT(A), { vectors: false })
  const re = toFlat(e.real)
  const im = toFlat(e.imag)
  const stable = discrete ? re.every((r, i) => Math.hypot(r, im[i]) < 1) : re.every((r) => r < 0)
  if (!stable) return { W: null, stable }
  return { W: lyapunov(toT(A), toT(Q), { discrete }).X, stable }
}

// ---------------------------------------------------------------------------------------------------------------------
// Pole placement

/** Desired closed-loop poles: real parts and imaginary parts (complex poles in conjugate pairs). */
export type PoleSet = { real: VectorLike; imag?: VectorLike }

/**
 * The monic polynomial with the given roots, coefficients highest degree first ([1, c₁, …, cₙ]). Complex roots must
 * come in conjugate pairs so the coefficients are real; throws otherwise.
 */
export function polynomialFromRoots({ real, imag }: PoleSet): Vector {
  const re = Array.from(asM(real, 'roots').d)
  const im = imag === undefined ? re.map(() => 0) : Array.from(asM(imag, 'roots').d)
  if (im.length !== re.length) throw new Error('polynomialFromRoots: real and imag differ in length')
  let pr = [1]
  let pi = [0]
  re.forEach((r, k) => {
    const i = im[k]
    // Multiply by (s − (r + i·j)).
    const nr = new Array(pr.length + 1).fill(0)
    const ni = new Array(pr.length + 1).fill(0)
    for (let j = 0; j < pr.length; j++) {
      nr[j] += pr[j]
      ni[j] += pi[j]
      nr[j + 1] -= pr[j] * r - pi[j] * i
      ni[j + 1] -= pr[j] * i + pi[j] * r
    }
    pr = nr
    pi = ni
  })
  const scale = Math.max(1, ...pr.map(Math.abs))
  if (pi.some((v) => Math.abs(v) > 1e-9 * scale))
    throw new Error('polynomialFromRoots: complex roots must come in conjugate pairs')
  return toV(pr)
}

/** The result of `ackermann`. */
export type PolePlacement = {
  /** The gain K (1×n) with eig(A − BK) = the requested poles, or null when (A, b) is not controllable. */
  K: Matrix | null
  /** The desired characteristic polynomial [1, α₁, …, αₙ]. */
  characteristic: Vector
  /** The eigenvalues of A − BK actually obtained (empty when K is null). */
  closedLoop: { real: Vector; imag: Vector }
  controllable: boolean
  /** The condition number σ_max/σ_min of the controllability matrix; large means the gain is sensitive. */
  conditioning: number
}

/**
 * Single-input pole placement by Ackermann's formula (Ackermann, 1972): K = [0 … 0 1] 𝒞⁻¹ φ(A), where 𝒞 is the
 * controllability matrix and φ(s) = Πᵢ(s − pᵢ) the desired characteristic polynomial, so that u = −Kx gives
 * eig(A − bK) = {pᵢ}. Numerically poor for large n or nearly uncontrollable pairs (it inverts 𝒞): the conditioning is
 * reported, and the achieved poles are returned for checking.
 */
export function ackermann(sys: StateSpace, desired: PoleSet): PolePlacement {
  if (sys.inputs !== 1) throw new Error('ackermann: single-input systems only')
  const { A, B } = matrices(sys)
  const n = sys.states
  const phi = polynomialFromRoots(desired)
  const alpha = Array.from(toFlat(phi))
  if (alpha.length !== n + 1) throw new Error(`ackermann: need ${n} poles, got ${alpha.length - 1}`)
  const Cm = krylov(A, B)
  const { rank, singularValues } = rankM(Cm)
  const conditioning = singularValues[0] / singularValues[singularValues.length - 1]
  const empty = { real: toV([]), imag: toV([]) }
  if (rank < n) return { K: null, characteristic: phi, closedLoop: empty, controllable: false, conditioning }
  // φ(A) = Aⁿ + α₁Aⁿ⁻¹ + … + αₙI by Horner's rule.
  let P = zerosM(n, n)
  for (const c of alpha) P = add(mul(P, A), mulScalarEye(n, c))
  // Row vector eₙᵀ 𝒞⁻¹: solve 𝒞ᵀ w = eₙ.
  const en = zerosM(n, 1)
  en.d[n - 1] = 1
  const w = solveM(tr(Cm), en)
  if (!w) return { K: null, characteristic: phi, closedLoop: empty, controllable: false, conditioning }
  const K = mul(tr(w), P)
  const cl = eig(toT(sub(A, mul(B, K))), { vectors: false })
  return {
    K: toT(K),
    characteristic: phi,
    closedLoop: { real: cl.real, imag: cl.imag },
    controllable: true,
    conditioning,
  }
}

function mulScalarEye(n: number, c: number): M {
  const I = eyeM(n)
  return { d: I.d.map((v) => v * c), r: n, c: n }
}
