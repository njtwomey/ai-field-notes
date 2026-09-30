/**
 * Quasi-Newton methods: BFGS, which keeps a dense inverse-Hessian approximation, and L-BFGS, which keeps only the
 * last m curvature pairs (s, y) and applies the approximation by the two-loop recursion. Both take steps from a strong
 * Wolfe line search, which guarantees yᵀs > 0 and so keeps the approximation positive definite.
 */

import type { Matrix, Vector } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import type { StartOptions } from '../options'
import { strongWolfeSearch, type LineSearchResult, type StrongWolfeOptions } from 'aifn/optim/line-search'
import type { IterateState, ObjectiveFn, StoppingOptions } from 'aifn/foundation/contracts'
import { DEFAULT_DIVERGE, DEFAULT_TOLERANCE, divergedAt, evaluate } from '../options'
import { dense } from 'aifn/foundation/tensor'

const { axpy, data, dot, mat, matVec, norm, scale, sub, toF64, vec } = dense
type F64 = dense.F64

/** One curvature pair: the step s = x_{k+1} − x_k, the gradient change y = ∇f_{k+1} − ∇f_k, and ρ = 1/(yᵀs). */
export type CurvaturePair = { s: Vector; y: Vector; rho: number }

/** Options for `bfgs` and `lbfgs`. */
export type QuasiNewtonOptions = StoppingOptions & {
  /** Options for the strong Wolfe line search (default c₁ = 1e-4, c₂ = 0.9). */
  lineSearchOptions?: StrongWolfeOptions
}

/** The state of `bfgs`. */
export type BfgsState = IterateState & {
  grad: Vector
  gradNorm: number
  /** The inverse-Hessian approximation H_k (n×n, symmetric positive definite). */
  inverseHessian: Matrix
  /** The search direction p = −H∇f of the last step (zeros at t = 0). */
  direction: Vector
  /** The last curvature pair, or null at t = 0. */
  pair: CurvaturePair | null
  /** True when the last update was skipped because yᵀs was not safely positive. */
  skipped: boolean
  stepSize: number
  lineSearch: LineSearchResult | null
  /** True when the last line search could not lower f (x unchanged); the run stops. */
  stalled: boolean
}

/** The shared outer step: search along p, then report the pair. */
function searchAlong(f: ObjectiveFn, x: F64, value: number, g: F64, p: F64, options: StrongWolfeOptions | undefined) {
  const found = strongWolfeSearch(f, x, value, g, p, options)
  const s = sub(found.x, x)
  const y = sub(found.grad, g)
  return { found, s, y, sy: dot(s, y) }
}

/**
 * BFGS (Broyden, Fletcher, Goldfarb and Shanno, 1970; Nocedal & Wright, Algorithm 6.1): p = −H∇f, a strong Wolfe step,
 * then the inverse update H ← (I − ρsyᵀ)H(I − ρysᵀ) + ρssᵀ. Before the first update H₀ = I is rescaled to
 * (yᵀs / yᵀy)I (eq. 6.20). An update with yᵀs ≤ 10⁻¹⁰‖s‖‖y‖ is skipped and flagged. `init` takes `{ x0 }`.
 */
export function bfgs(f: ObjectiveFn, options: QuasiNewtonOptions = {}): Algorithm<StartOptions, BfgsState> {
  const { tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const name = 'bfgs'
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const n = x.length
      const { value, grad } = evaluate(f, x, name)
      const H = new Float64Array(n * n)
      for (let i = 0; i < n; i++) H[i * n + i] = 1
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        inverseHessian: mat(H, n, n),
        direction: vec(new Float64Array(n)),
        pair: null,
        skipped: false,
        stepSize: NaN,
        lineSearch: null,
        stalled: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (st) => {
      const n = st.x.shape[0]
      const x = data(st.x)
      const g = data(st.grad)
      let H = data(st.inverseHessian)
      const p = scale(-1, matVec(H, g, n, n))
      const { found, s, y, sy } = searchAlong(f, x, st.value, g, p, options.lineSearchOptions)
      const skipped = !(sy > 1e-10 * norm(s) * norm(y))
      if (!skipped) {
        if (st.t === 0) H = scale(sy / dot(y, y), H)
        const rho = 1 / sy
        const Hy = matVec(H, y, n, n)
        const yHy = dot(y, Hy)
        // Expanded form of (I − ρsyᵀ)H(I − ρysᵀ) + ρssᵀ, using the symmetry of H.
        const next = new Float64Array(n * n)
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++)
            next[i * n + j] = H[i * n + j] - rho * (s[i] * Hy[j] + Hy[i] * s[j]) + (rho * rho * yHy + rho) * s[i] * s[j]
        H = next
      }
      const gradNorm = norm(found.grad)
      return {
        t: st.t + 1,
        x: vec(found.x),
        value: found.value,
        grad: vec(found.grad),
        gradNorm,
        inverseHessian: mat(H === data(st.inverseHessian) ? Float64Array.from(H) : H, n, n),
        direction: vec(p),
        pair: { s: vec(s), y: vec(y), rho: 1 / sy },
        skipped,
        stepSize: found.result.alpha,
        lineSearch: found.result,
        stalled: found.result.alpha === 0,
        evaluations: st.evaluations + found.result.evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(found.value, found.x, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}

/** The state of `lbfgs`. */
export type LbfgsState = IterateState & {
  grad: Vector
  gradNorm: number
  /** The stored curvature pairs, oldest first (at most `memory`). */
  pairs: CurvaturePair[]
  /** The initial-Hessian scale γ = sᵀy / yᵀy of the newest pair (1 before any pair). */
  gamma: number
  /** The search direction of the last step (zeros at t = 0). */
  direction: Vector
  stepSize: number
  lineSearch: LineSearchResult | null
  /** True when the last pair was not stored because yᵀs was not safely positive. */
  skipped: boolean
  /** True when the last line search could not lower f (x unchanged); the run stops. */
  stalled: boolean
}

/** Options for `lbfgs`. */
export type LbfgsOptions = QuasiNewtonOptions & {
  /** Number of curvature pairs kept, m. Default 10. */
  memory?: number
}

/**
 * The two-loop recursion (Nocedal & Wright, Algorithm 7.4): returns H·q for the L-BFGS inverse Hessian built from the
 * pairs (oldest first) on the initial matrix γI.
 */
function twoLoop(q0: F64, pairs: { s: F64; y: F64; rho: number }[], gamma: number): F64 {
  let q = Float64Array.from(q0)
  const alphas = new Array<number>(pairs.length)
  for (let i = pairs.length - 1; i >= 0; i--) {
    const { s, y, rho } = pairs[i]
    alphas[i] = rho * dot(s, q)
    q = axpy(-alphas[i], y, q)
  }
  let r = scale(gamma, q)
  for (let i = 0; i < pairs.length; i++) {
    const { s, y, rho } = pairs[i]
    const beta = rho * dot(y, r)
    r = axpy(alphas[i] - beta, s, r)
  }
  return r
}

/**
 * Limited-memory BFGS (Liu & Nocedal, 1989; Nocedal & Wright, Algorithm 7.5): the direction −H∇f comes from the
 * two-loop recursion over the last m pairs with H₀ = γI, γ = sᵀy/yᵀy of the newest pair; steps satisfy the strong
 * Wolfe conditions. The first step, with no pairs, is scaled to length 1 along −∇f. `init` takes `{ x0 }`.
 */
export function lbfgs(f: ObjectiveFn, options: LbfgsOptions = {}): Algorithm<StartOptions, LbfgsState> {
  const { memory = 10, tolerance = DEFAULT_TOLERANCE, divergeAbove = DEFAULT_DIVERGE } = options
  const name = 'lbfgs'
  return {
    name,
    init: ({ x0 }) => {
      const x = toF64(x0, name)
      const { value, grad } = evaluate(f, x, name)
      const gradNorm = norm(grad)
      return {
        t: 0,
        x: vec(x),
        value,
        grad: vec(grad),
        gradNorm,
        pairs: [],
        gamma: 1,
        direction: vec(new Float64Array(x.length)),
        stepSize: NaN,
        lineSearch: null,
        skipped: false,
        stalled: false,
        evaluations: 1,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(value, x, divergeAbove),
      }
    },
    step: (st) => {
      const x = data(st.x)
      const g = data(st.grad)
      const pairs = st.pairs.map((q) => ({ s: data(q.s), y: data(q.y), rho: q.rho }))
      // Without curvature information, a unit-length steepest-descent step (as scipy's L-BFGS-B starts).
      const gamma = pairs.length ? st.gamma : 1 / Math.max(norm(g), 1e-300)
      const p = scale(-1, twoLoop(g, pairs, gamma))
      const { found, s, y, sy } = searchAlong(f, x, st.value, g, p, options.lineSearchOptions)
      const skipped = !(sy > 1e-10 * norm(s) * norm(y))
      let kept = st.pairs
      let nextGamma = st.gamma
      if (!skipped) {
        kept = [...st.pairs, { s: vec(s), y: vec(y), rho: 1 / sy }].slice(-memory)
        nextGamma = sy / dot(y, y)
      }
      const gradNorm = norm(found.grad)
      return {
        t: st.t + 1,
        x: vec(found.x),
        value: found.value,
        grad: vec(found.grad),
        gradNorm,
        pairs: kept,
        gamma: nextGamma,
        direction: vec(p),
        stepSize: found.result.alpha,
        lineSearch: found.result,
        skipped,
        stalled: found.result.alpha === 0,
        evaluations: st.evaluations + found.result.evaluations,
        converged: gradNorm <= tolerance,
        diverged: divergedAt(found.value, found.x, divergeAbove),
      }
    },
    done: (s) => s.converged || s.diverged || s.stalled,
  }
}
