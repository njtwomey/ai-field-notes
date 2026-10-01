/**
 * New systems from old: discretisation (zero-order hold, forward Euler, Tustin), state feedback, and interconnection
 * (series, parallel, feedback).
 *
 * Sources: Van Loan (1978), "Computing integrals involving the matrix exponential", IEEE Trans. Automatic Control
 * 23(3) (the zero-order hold by one matrix exponential); Franklin, Powell & Workman (1998), "Digital Control of Dynamic
 * Systems", 3rd ed., §6.2 (Tustin); Ogata (2010), "Modern Control Engineering", §2-3 (block-diagram algebra).
 */

import { expm, solveDense } from 'aifn/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn/foundation/tensor'
import type { LtiSystem, MatrixLike, Scalar, VectorLike } from 'aifn/foundation/contracts'
import { DomainError, ShapeError } from 'aifn/foundation/errors'
import {
  mulCoefficients,
  rationalOf,
  stateSpace,
  stripLeading,
  toAscending,
  toStateSpace,
  withRepr,
  type LtiOf,
  type Rational,
  type StateSpaceForm,
  type TransferFunctionForm,
} from './system'

const { matMul, identity } = dense

const addM = (a: ArrayLike<number>, b: ArrayLike<number>, beta = 1) => Float64Array.from(a, (v, i) => v + beta * b[i])
const scaleM = (a: ArrayLike<number>, k: Scalar) => Float64Array.from(a, (v) => v * k)

/** Methods for `discretise`. */
export type DiscretisationMethod = 'zoh' | 'euler' | 'tustin'

/**
 * A continuous system sampled every `dt`, in state-space form (any representation is realised first):
 * - `zoh` (zero-order hold, exact for piecewise-constant inputs): exp([[A, B], [0, 0]]·dt) = [[A_d, B_d], [0, I]], so
 *   A_d = e^{A dt} and B_d = ∫₀^dt e^{As} ds B (Van Loan, 1978); C and D unchanged.
 * - `euler` (forward difference): A_d = I + A dt, B_d = B dt.
 * - `tustin` (bilinear, s ≈ (2/dt)(z − 1)/(z + 1)): with W = (I − A dt/2)⁻¹, A_d = W(I + A dt/2), B_d = W B dt,
 *   C_d = C W, D_d = D + C W B dt/2 (as scipy's `cont2discrete` with `bilinear`). Stability is preserved.
 * A delay τ (seconds) becomes τ/dt samples.
 */
export function discretise(sys: LtiSystem, dt: Scalar, method: DiscretisationMethod = 'zoh'): LtiOf<StateSpaceForm> {
  if (sys.domain !== 'continuous') throw new DomainError('discretise', 'discretise: the system is already discrete')
  if (!(dt > 0)) throw new DomainError('discretise', 'discretise: dt must be positive')
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  const p = r.C.shape[0]
  const A = dense.data(r.A)
  const B = dense.data(r.B)
  const C = dense.data(r.C)
  const D = dense.data(r.D)
  const options = { dt, delay: sys.delay / dt }
  const build = (Ad: ArrayLike<number>, Bd: ArrayLike<number>, Cd: ArrayLike<number>, Dd: ArrayLike<number>) =>
    stateSpace({
      A: fromData(Float64Array.from(Ad), [n, n]),
      B: fromData(Float64Array.from(Bd), [n, m]),
      C: fromData(Float64Array.from(Cd), [p, n]),
      D: fromData(Float64Array.from(Dd), [p, m]),
      ...options,
    })
  if (method === 'euler') return build(addM(identity(n), scaleM(A, dt)), scaleM(B, dt), C, D)
  if (method === 'tustin') {
    const lhs = addM(identity(n), scaleM(A, dt / 2), -1)
    const W = solveDense(lhs, identity(n), n)
    if (!W.x) throw new DomainError('discretise', 'discretise: I − A dt/2 is singular (an eigenvalue of A at 2/dt)')
    const Wm = W.x
    const Bd = scaleM(matMul(Wm, B, n, n, m), dt)
    return build(
      matMul(Wm, addM(identity(n), scaleM(A, dt / 2)), n, n, n),
      Bd,
      matMul(C, Wm, p, n, n),
      addM(D, scaleM(matMul(C, Bd, p, n, m), 0.5)),
    )
  }
  const size = n + m
  const big = new Float64Array(size * size)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) big[i * size + j] = A[i * n + j] * dt
    for (let j = 0; j < m; j++) big[i * size + n + j] = B[i * m + j] * dt
  }
  const E = dense.data(expm(fromData(big, [size, size])).value)
  const Ad = new Float64Array(n * n)
  const Bd = new Float64Array(n * m)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) Ad[i * n + j] = E[i * size + j]
    for (let j = 0; j < m; j++) Bd[i * m + j] = E[i * size + n + j]
  }
  return build(Ad, Bd, C, D)
}

/** The system with state feedback u = −Kx + v: A ← A − BK, C ← C − DK. K is m×n (a vector for one input). */
export function stateFeedback(sys: LtiSystem, K: MatrixLike | VectorLike): LtiOf<StateSpaceForm> {
  const r = toStateSpace(sys).repr
  const n = r.A.shape[0]
  const m = r.B.shape[1]
  const p = r.C.shape[0]
  const isMatrix =
    (K as { shape?: readonly number[] }).shape?.length === 2 || typeof (K as ArrayLike<unknown>)[0] === 'object'
  const k = isMatrix
    ? dense.toMatrixF64(K as MatrixLike, 'stateFeedback K').data
    : dense.toF64(K as VectorLike, 'stateFeedback K')
  if (k.length !== m * n) throw new ShapeError('stateFeedback', `stateFeedback: K must be ${m}×${n}`)
  const A = addM(dense.data(r.A), matMul(dense.data(r.B), k, n, m, n), -1)
  const C = addM(dense.data(r.C), matMul(dense.data(r.D), k, p, m, n), -1)
  return withRepr(sys, { ...r, A: fromData(A, [n, n]), C: fromData(C, [p, n]) })
}

function sameDomain(a: LtiSystem, b: LtiSystem, where: string) {
  if (a.domain !== b.domain || a.dt !== b.dt)
    throw new DomainError(where, `${where}: the systems must share a domain and sampling interval`)
}

function fromRational(template: LtiSystem, { num, den }: Rational, delay: Scalar): LtiOf<TransferFunctionForm> {
  const vec = (v: number[]): Tensor => fromData(Float64Array.from(v), [v.length])
  if (template.domain === 'continuous')
    return { ...withRepr(template, { form: 'tf', b: vec(stripLeading(num)), a: vec(stripLeading(den)) }), delay }
  const { b, a } = toAscending({ num, den })
  return { ...withRepr(template, { form: 'tf', b: vec(b), a: vec(a) }), delay }
}

const polyAdd = (a: readonly number[], b: readonly number[]) => {
  const n = Math.max(a.length, b.length)
  const pa = [...new Array(n - a.length).fill(0), ...a]
  const pb = [...new Array(n - b.length).fill(0), ...b]
  return pa.map((v, i) => v + pb[i])
}

/** The series connection G₂G₁ of two SISO systems (G₁ first; delays add), as a transfer function. */
export function series(g1: LtiSystem, g2: LtiSystem): LtiOf<TransferFunctionForm> {
  sameDomain(g1, g2, 'series')
  const a = rationalOf(g1)
  const b = rationalOf(g2)
  return fromRational(
    g1,
    { num: mulCoefficients(a.num, b.num), den: mulCoefficients(a.den, b.den) },
    g1.delay + g2.delay,
  )
}

/** The parallel connection G₁ + G₂ of two SISO systems with equal delays, as a transfer function. */
export function parallel(g1: LtiSystem, g2: LtiSystem): LtiOf<TransferFunctionForm> {
  sameDomain(g1, g2, 'parallel')
  if (g1.delay !== g2.delay) throw new DomainError('parallel', 'parallel: the delays must be equal')
  const a = rationalOf(g1)
  const b = rationalOf(g2)
  const num = polyAdd(mulCoefficients(a.num, b.den), mulCoefficients(b.num, a.den))
  return fromRational(g1, { num, den: mulCoefficients(a.den, b.den) }, g1.delay)
}

/**
 * The closed loop of L under negative feedback through H (default unity): L/(1 + LH), i.e.
 * num_L den_H / (den_L den_H + num_L num_H). A delay in the loop has no rational closed form: it throws.
 */
export function feedback(L: LtiSystem, H?: LtiSystem): LtiOf<TransferFunctionForm> {
  const h = H ?? withRepr(L, { form: 'tf', b: fromData(Float64Array.of(1), [1]), a: fromData(Float64Array.of(1), [1]) })
  sameDomain(L, h, 'feedback')
  if (L.delay || h.delay) throw new DomainError('feedback', 'feedback: a loop with delay is not rational; simulate it')
  const l = rationalOf(L)
  const k = rationalOf({ ...h, delay: 0 })
  const num = mulCoefficients(l.num, k.den)
  const den = polyAdd(mulCoefficients(l.den, k.den), mulCoefficients(l.num, k.num))
  return fromRational(L, { num, den }, 0)
}
