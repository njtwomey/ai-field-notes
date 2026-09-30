/**
 * Discrete optimal transport between weighted point sets: cost matrices, the exact plan (the Hungarian algorithm for
 * equal numbers of equally weighted points, a linear program otherwise), and entropic transport by log-domain Sinkhorn
 * iterations (Cuturi, 2013, NeurIPS; Peyré and Cuturi, 2019, "Computational Optimal Transport", §4.4), traceable.
 */

import { hungarian, linprog } from 'aifn/programming'
import { copy, fromData, isTensor, type Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { run } from 'aifn/trace'

/** Points: an n × d tensor, rows of coordinates, or a 1-D array (n points on a line). */
export type PointsInput = Tensor | readonly (readonly number[])[] | readonly number[]
/** Weights or a vector: a rank-1 tensor or an array. */
export type WeightsInput = Tensor | ArrayLike<number>
/** A cost matrix: a rank-2 tensor or rows. */
export type CostInput = Tensor | readonly (readonly number[])[]

export function readPoints(x: PointsInput, what: string): { v: Float64Array; n: number; d: number } {
  if (isTensor(x)) {
    const v = copy(x, 'float64').data as Float64Array
    if (x.shape.length === 1) return { v, n: x.shape[0], d: 1 }
    if (x.shape.length === 2) return { v, n: x.shape[0], d: x.shape[1] }
    throw new Error(`${what}: points must be rank 1 or 2`)
  }
  if (x.length === 0) return { v: new Float64Array(0), n: 0, d: 1 }
  if (typeof x[0] === 'number') return { v: Float64Array.from(x as readonly number[]), n: x.length, d: 1 }
  const rows = x as readonly (readonly number[])[]
  const d = rows[0].length
  const v = new Float64Array(rows.length * d)
  rows.forEach((r, i) => v.set(r, i * d))
  return { v, n: rows.length, d }
}

export function readVector(x: WeightsInput, what: string): Float64Array {
  if (isTensor(x)) {
    if (x.shape.length !== 1) throw new Error(`${what}: expected a vector`)
    return copy(x, 'float64').data as Float64Array
  }
  return Float64Array.from(x)
}

export function readCost(c: CostInput, n: number, m: number, what: string): Float64Array {
  let v: Float64Array
  if (isTensor(c)) {
    if (c.shape.length !== 2 || c.shape[0] !== n || c.shape[1] !== m)
      throw new Error(`${what}: cost must be ${n} × ${m}`)
    v = copy(c, 'float64').data as Float64Array
  } else {
    if (c.length !== n) throw new Error(`${what}: cost must have ${n} rows`)
    v = new Float64Array(n * m)
    c.forEach((r, i) => {
      if (r.length !== m) throw new Error(`${what}: cost row ${i} must have ${m} entries`)
      v.set(r, i * m)
    })
  }
  return v
}

/** Uniform weights 1/n. */
export function uniformWeights(n: number): Tensor {
  return fromData(new Float64Array(n).fill(1 / n))
}

/**
 * The cost matrix C[i][j] = ‖x_i − y_j‖^p between two point sets (Euclidean norm; p = 2 by default, the squared
 * distance), or `metric: 'cityblock'` for the L1 norm raised to p.
 */
export function costMatrix(
  x: PointsInput,
  y: PointsInput,
  { p = 2, metric = 'euclidean' }: { p?: number; metric?: 'euclidean' | 'cityblock' } = {},
): Tensor {
  const X = readPoints(x, 'costMatrix')
  const Y = readPoints(y, 'costMatrix')
  if (X.d !== Y.d) throw new Error(`costMatrix: points of dimension ${X.d} and ${Y.d}`)
  const out = new Float64Array(X.n * Y.n)
  for (let i = 0; i < X.n; i++)
    for (let j = 0; j < Y.n; j++) {
      let s = 0
      for (let k = 0; k < X.d; k++) {
        const diff = X.v[i * X.d + k] - Y.v[j * Y.d + k]
        s += metric === 'euclidean' ? diff * diff : Math.abs(diff)
      }
      const dist = metric === 'euclidean' ? Math.sqrt(s) : s
      out[i * Y.n + j] = p === 2 && metric === 'euclidean' ? s : dist ** p
    }
  return fromData(out, [X.n, Y.n])
}

/** An exact optimal transport plan. */
export interface TransportPlan {
  /** The plan P (n × m): P[i][j] is the mass moved from x_i to y_j. */
  plan: Tensor
  /** ⟨C, P⟩, the optimal cost. */
  cost: number
  /** Dual potentials f (n) and g (m) with f_i + g_j ≤ C_ij and equality on the support; null for the Hungarian path. */
  f: Tensor | null
  g: Tensor | null
  method: 'hungarian' | 'simplex'
  /** For `hungarian`, the column matched to each row (int32). */
  assignment: Tensor | null
}

function isUniform(w: Float64Array, n: number): boolean {
  return w.every((v) => Math.abs(v - 1 / n) < 1e-12)
}

/**
 * The exact optimal transport plan minimising ⟨C, P⟩ over couplings P ≥ 0 with row sums a and column sums b
 * (Kantorovich's problem). With n = m and uniform weights an optimal plan is a permutation (Birkhoff–von Neumann), found
 * by the Hungarian algorithm; otherwise the linear program is solved by the simplex method. Weights must have equal
 * totals.
 */
export function exactTransport(a: WeightsInput, b: WeightsInput, cost: CostInput): TransportPlan {
  const av = readVector(a, 'exactTransport a')
  const bv = readVector(b, 'exactTransport b')
  const n = av.length
  const m = bv.length
  const C = readCost(cost, n, m, 'exactTransport')
  const ta = av.reduce((s, v) => s + v, 0)
  const tb = bv.reduce((s, v) => s + v, 0)
  if (Math.abs(ta - tb) > 1e-9 * Math.max(1, ta)) throw new RangeError(`exactTransport: masses ${ta} and ${tb} differ`)
  if (n === m && isUniform(av, n) && isUniform(bv, m)) {
    const r = hungarian(fromData(C, [n, m]))
    const plan = new Float64Array(n * m)
    const assign = r.assignment.data
    for (let i = 0; i < n; i++) plan[i * m + assign[i]] = 1 / n
    return {
      plan: fromData(plan, [n, m]),
      cost: r.cost / n,
      f: null,
      g: null,
      method: 'hungarian',
      assignment: r.assignment,
    }
  }
  // Variables P_ij in row-major order; one equality per row sum and per column sum.
  const Aeq: number[][] = []
  for (let i = 0; i < n; i++) Aeq.push(Array.from({ length: n * m }, (_, k) => (Math.floor(k / m) === i ? 1 : 0)))
  for (let j = 0; j < m; j++) Aeq.push(Array.from({ length: n * m }, (_, k) => (k % m === j ? 1 : 0)))
  const r = linprog({ c: Array.from(C), A_eq: Aeq, b_eq: [...av, ...bv] })
  if (r.status !== 'optimal') throw new Error(`exactTransport: the linear program ended ${r.status}`)
  const eq = r.report ? (copy(r.report.duals.eq, 'float64').data as Float64Array) : null
  return {
    plan: fromData(copy(r.x, 'float64').data as Float64Array, [n, m]),
    cost: r.objective,
    f: eq ? fromData(eq.slice(0, n)) : null,
    g: eq ? fromData(eq.slice(n)) : null,
    method: 'simplex',
    assignment: null,
  }
}

// ── Sinkhorn ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** Options for `sinkhornSteps`. */
export interface SinkhornOptions {
  a: WeightsInput
  b: WeightsInput
  cost: CostInput
  /** Entropic regularisation ε > 0 (in cost units). */
  epsilon: number
  /** Stop when the L1 error of the row marginal is below this. Default 1e-9. */
  tolerance?: number
}

/** One state of log-domain Sinkhorn. */
export interface SinkhornState {
  /** Dual potentials f (n) and g (m). */
  f: Tensor
  g: Tensor
  /** The plan P_ij = a_i b_j exp((f_i + g_j − C_ij)/ε), n × m. */
  plan: Tensor
  iteration: number
  /** L1 error of the row sums after the column update (the column sums are exact). */
  marginalError: number
  /** ⟨C, P⟩. */
  transportCost: number
  /** The entropic objective ⟨C, P⟩ + ε KL(P ‖ a bᵀ). */
  value: number
  /** The dual objective aᵀf + bᵀg (increases monotonically). */
  dual: number
  converged: boolean
  /** A potential became non-finite (ε too small for the cost scale is the usual cause). */
  diverged: boolean
  epsilon: number
  tolerance: number
  a: Tensor
  b: Tensor
  cost: Tensor
}

/**
 * −ε log Σ_k exp((pot_k − cost_k)/ε + logw_k), the soft minimum, computed stably by shifting by the maximum exponent.
 */
function softMin(
  pot: Float64Array,
  cost: (k: number) => number,
  logw: Float64Array,
  eps: number,
  buf: Float64Array,
): number {
  let mx = -Infinity
  for (let k = 0; k < pot.length; k++) {
    const v = (pot[k] - cost(k)) / eps + logw[k]
    buf[k] = v
    if (v > mx) mx = v
  }
  if (!Number.isFinite(mx)) return -eps * mx
  let s = 0
  for (let k = 0; k < pot.length; k++) s += Math.exp(buf[k] - mx)
  return -eps * (mx + Math.log(s))
}

function summarise(
  base: Omit<SinkhornState, 'plan' | 'marginalError' | 'transportCost' | 'value' | 'dual' | 'converged' | 'diverged'>,
): SinkhornState {
  const a = base.a.data
  const b = base.b.data
  const C = base.cost.data
  const f = base.f.data
  const g = base.g.data
  const n = a.length
  const m = b.length
  const eps = base.epsilon
  const P = new Float64Array(n * m)
  let cost = 0
  let kl = 0
  let err = 0
  for (let i = 0; i < n; i++) {
    let row = 0
    for (let j = 0; j < m; j++) {
      const ab = a[i] * b[j]
      const p = ab * Math.exp((f[i] + g[j] - C[i * m + j]) / eps)
      P[i * m + j] = p
      row += p
      cost += p * C[i * m + j]
      if (p > 0) kl += p * Math.log(p / ab) - p + ab
    }
    err += Math.abs(row - a[i])
  }
  let dual = 0
  for (let i = 0; i < n; i++) dual += a[i] * f[i]
  for (let j = 0; j < m; j++) dual += b[j] * g[j]
  const diverged = !Number.isFinite(dual) || !Number.isFinite(err)
  return {
    ...base,
    plan: fromData(P, [n, m]),
    marginalError: err,
    transportCost: cost,
    value: cost + eps * kl,
    dual,
    converged: err < base.tolerance,
    diverged,
  }
}

/**
 * Sinkhorn's algorithm in the log domain as a traceable algorithm (Schmitzer, 2019, SIAM J. Sci. Comput. 41(3)):
 * each step sets f to the soft c-transform of g, then g to that of f, so the column marginal is exact and the row
 * marginal carries the error. The plan is relative to the product measure a bᵀ, so ε is the weight of KL(P ‖ a bᵀ).
 * As ε → 0 the plan approaches an exact optimal plan; large ε blurs it towards a bᵀ.
 */
export const sinkhornSteps: Algorithm<SinkhornOptions, SinkhornState> = {
  name: 'sinkhorn',
  init({ a, b, cost, epsilon, tolerance = 1e-9 }) {
    if (!(epsilon > 0)) throw new RangeError('sinkhorn: epsilon must be positive')
    const av = readVector(a, 'sinkhorn a')
    const bv = readVector(b, 'sinkhorn b')
    const C = readCost(cost, av.length, bv.length, 'sinkhorn')
    return summarise({
      f: fromData(new Float64Array(av.length)),
      g: fromData(new Float64Array(bv.length)),
      iteration: 0,
      epsilon,
      tolerance,
      a: fromData(av),
      b: fromData(bv),
      cost: fromData(C, [av.length, bv.length]),
    })
  },
  step(s) {
    const a = s.a.data as Float64Array
    const b = s.b.data as Float64Array
    const C = s.cost.data
    const n = a.length
    const m = b.length
    const la = a.map(Math.log)
    const lb = b.map(Math.log)
    const f = new Float64Array(n)
    const g = new Float64Array(m)
    const gPrev = s.g.data as Float64Array
    const bufN = new Float64Array(n)
    const bufM = new Float64Array(m)
    for (let i = 0; i < n; i++) f[i] = softMin(gPrev, (j) => C[i * m + j], lb, s.epsilon, bufM)
    for (let j = 0; j < m; j++) g[j] = softMin(f, (i) => C[i * m + j], la, s.epsilon, bufN)
    return summarise({ ...s, f: fromData(f), g: fromData(g), iteration: s.iteration + 1 })
  },
  done: (s) => s.converged,
}

/** Entropic optimal transport by Sinkhorn iterations run to `tolerance` or `maxIterations` (default 1000). */
export function sinkhorn(
  a: WeightsInput,
  b: WeightsInput,
  cost: CostInput,
  { epsilon, tolerance, maxIterations = 1000 }: { epsilon: number; tolerance?: number; maxIterations?: number },
): SinkhornState {
  return run(sinkhornSteps, { a, b, cost, epsilon, tolerance }, maxIterations)
}
