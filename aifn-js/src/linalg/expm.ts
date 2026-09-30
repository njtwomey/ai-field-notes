/**
 * The matrix exponential e^A by Padé approximation with scaling and squaring (Higham, 2005), as used for linear
 * systems x′ = Ax (x(t) = e^{At} x₀) and the discretisation of continuous-time models.
 */

import { dense, fromData, toFlat, type Matrix, type MatrixLike, type Tensor } from 'aifn/tensor'
import { solve } from './lu'

type F64 = dense.F64
const { identity, matMul, norm1, toMatrixF64 } = dense

// Padé coefficients b_k of the [m/m] approximant to eˣ and the largest 1-norm θ_m for which degree m is accurate to
// double precision (Higham, 2005, "The scaling and squaring method for the matrix exponential revisited", SIAM J.
// Matrix Anal. Appl. 26(4), Table 2.3 and Algorithm 2.3).
const PADE: Record<number, number[]> = {
  3: [120, 60, 12, 1],
  5: [30240, 15120, 3360, 420, 30, 1],
  7: [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1],
  9: [17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1],
  13: [
    64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600,
    670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1,
  ],
}
const THETA: [number, number][] = [
  [3, 1.495585217958292e-2],
  [5, 2.53939833006323e-1],
  [7, 9.504178996162932e-1],
  [9, 2.097847961257068],
]
const THETA13 = 5.371920351148152

const addScaled = (out: F64, alpha: number, x: F64) => {
  for (let i = 0; i < out.length; i++) out[i] += alpha * x[i]
}

/** U and V of the [m/m] Padé approximant r_m(A) = (V − U)⁻¹(V + U), from the even powers A², A⁴, … . */
function padeUV(A: F64, n: number, m: number, powers: F64[]): { U: F64; V: F64 } {
  const b = PADE[m]
  const I = identity(n)
  if (m < 13) {
    // U = A Σ b_{2k+1} A^{2k}, V = Σ b_{2k} A^{2k}.
    const u = new Float64Array(n * n)
    const V = new Float64Array(n * n)
    addScaled(u, b[1], I)
    addScaled(V, b[0], I)
    for (let k = 1; 2 * k <= m; k++) {
      addScaled(u, b[2 * k + 1], powers[k - 1])
      addScaled(V, b[2 * k], powers[k - 1])
    }
    return { U: matMul(A, u, n, n, n), V }
  }
  const [A2, A4, A6] = powers
  const inner = new Float64Array(n * n)
  addScaled(inner, b[13], A6)
  addScaled(inner, b[11], A4)
  addScaled(inner, b[9], A2)
  const u = matMul(A6, inner, n, n, n)
  addScaled(u, b[7], A6)
  addScaled(u, b[5], A4)
  addScaled(u, b[3], A2)
  addScaled(u, b[1], I)
  const innerV = new Float64Array(n * n)
  addScaled(innerV, b[12], A6)
  addScaled(innerV, b[10], A4)
  addScaled(innerV, b[8], A2)
  const V = matMul(A6, innerV, n, n, n)
  addScaled(V, b[6], A6)
  addScaled(V, b[4], A4)
  addScaled(V, b[2], A2)
  addScaled(V, b[0], I)
  return { U: matMul(A, u, n, n, n), V }
}

/** The result of `expm`: e^A and how it was computed. */
export type MatrixExponential = {
  /** e^A (n×n). */
  value: Matrix
  /** The Padé degree used (3, 5, 7, 9 or 13). */
  degree: number
  /** The number of squarings s: e^A = (r_m(A/2^s))^{2^s}. */
  squarings: number
}

/**
 * The matrix exponential e^A = Σ A^k/k! of a real square matrix by the scaling and squaring method with Padé
 * approximants (Higham, 2005, Algorithm 2.3, as scipy's `expm`): pick the smallest Padé degree m whose accuracy bound
 * θ_m covers ‖A‖₁, or scale A by 2^{−s} until ‖A/2^s‖₁ ≤ θ₁₃ and use degree 13, solve (V − U) R = V + U, then square
 * R s times. Accurate to near machine precision relative to ‖e^A‖ for normal matrices.
 *
 * @example expm([[0, 1], [-1, 0]]).value // rotation by 1 radian: [[cos 1, sin 1], [−sin 1, cos 1]]
 */
export function expm(a: MatrixLike): MatrixExponential {
  const { data: A0, m: rows, n } = toMatrixF64(a, 'expm')
  if (rows !== n) throw new Error(`expm: expected a square matrix, got ${rows}×${n}`)
  for (let i = 0; i < A0.length; i++) if (!Number.isFinite(A0[i])) throw new Error('expm: the matrix must be finite')
  if (n === 0) return { value: fromData(new Float64Array(0), [0, 0]), degree: 3, squarings: 0 }
  const norm = norm1(A0, n)
  const A2 = matMul(A0, A0, n, n, n)
  for (const [m, theta] of THETA) {
    if (norm <= theta) {
      const powers = [A2]
      for (let k = 2; 2 * k <= m; k++) powers.push(matMul(powers[k - 2], A2, n, n, n))
      return { value: finish(A0, n, m, powers, 0), degree: m, squarings: 0 }
    }
  }
  const s = Math.max(0, Math.ceil(Math.log2(norm / THETA13)))
  const f = 2 ** -s
  const A = A0.map((v) => v * f)
  const B2 = A2.map((v) => v * f * f)
  const B4 = matMul(B2, B2, n, n, n)
  const B6 = matMul(B4, B2, n, n, n)
  return { value: finish(A, n, 13, [B2, B4, B6], s), degree: 13, squarings: s }
}

function finish(A: F64, n: number, m: number, powers: F64[], squarings: number): Matrix {
  const { U, V } = padeUV(A, n, m, powers)
  const P = new Float64Array(n * n)
  const Q = new Float64Array(n * n)
  for (let i = 0; i < n * n; i++) {
    P[i] = V[i] + U[i]
    Q[i] = V[i] - U[i]
  }
  let R = Float64Array.from(toFlat(solve(fromData(Q, [n, n]), fromData(P, [n, n])) as Tensor))
  for (let k = 0; k < squarings; k++) R = matMul(R, R, n, n, n)
  return fromData(R, [n, n])
}
