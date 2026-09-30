/**
 * Convex quadratic programming: minimise ½xᵀQx + cᵀx subject to Ax ≤ b and Ex = e, by the primal active-set method
 * (Nocedal and Wright, 2006, "Numerical Optimization", Algorithm 16.3) and a primal–dual interior-point method with
 * Mehrotra's predictor–corrector (ibid., §16.6); and box-constrained QP by projected gradient with subspace Newton
 * steps (Moré and Toraldo, 1991, "On the solution of large quadratic programming problems with bound constraints",
 * SIAM J. Optimization 1(1)). Multipliers follow the Lagrangian L = ½xᵀQx + cᵀx + λᵀ(Ax − b) + νᵀ(Ex − e), λ ≥ 0.
 */

import type { Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { run } from 'aifn/trace'
import {
  checkFinite,
  dot,
  intVector,
  matTVec,
  matVec,
  matrix,
  norm2,
  normInf,
  readMatrix,
  readVector,
  solveSystem,
  vector,
  type Mat,
  type MatrixInput,
  type VectorInput,
} from './dense'
import { simplexSolve } from './simplex'

/** A convex quadratic program: minimise ½xᵀQx + cᵀx subject to A x ≤ b and E x = e (variables otherwise free). */
export interface QuadraticProgram {
  /** Symmetric positive semi-definite Hessian, n × n. */
  Q: MatrixInput
  /** Linear term, length n. */
  c: VectorInput
  /** Inequality constraints A x ≤ b: A is m × n. */
  A?: MatrixInput
  b?: VectorInput
  /** Equality constraints E x = e: E is p × n. */
  E?: MatrixInput
  e?: VectorInput
}

/** A quadratic program read into dense arrays. */
export interface ParsedQP {
  n: number
  Q: Mat
  c: Float64Array
  A: Mat
  b: Float64Array
  E: Mat
  e: Float64Array
}

/** Read and validate a quadratic program. */
export function parseQP(problem: QuadraticProgram): ParsedQP {
  const c = readVector(problem.c, 'quadratic program: c')
  const n = c.length
  const Q = readMatrix(problem.Q, 'quadratic program: Q', n)
  if (Q.m !== n) throw new Error(`quadratic program: Q must be ${n}×${n}`)
  const A = readMatrix(problem.A, 'quadratic program: A', n)
  const b = readVector(problem.b, 'quadratic program: b', A.m)
  const E = readMatrix(problem.E, 'quadratic program: E', n)
  const e = readVector(problem.e, 'quadratic program: e', E.m)
  for (const [v, name] of [
    [Q.a, 'Q'],
    [c, 'c'],
    [A.a, 'A'],
    [b, 'b'],
    [E.a, 'E'],
    [e, 'e'],
  ] as const)
    checkFinite(v, `quadratic program: ${name}`)
  return { n, Q, c, A, b, E, e }
}

const objectiveOf = (qp: ParsedQP, x: ArrayLike<number>) => 0.5 * dot(x, matVec(qp.Q, x)) + dot(qp.c, x)

/** The Karush–Kuhn–Tucker residuals of a point and multipliers of a quadratic program. */
export interface KKTReport {
  x: Tensor
  /** ½xᵀQx + cᵀx. */
  objective: number
  /** Multipliers λ of A x ≤ b (length m; ≥ 0 at an optimum). */
  lambda: Tensor
  /** Multipliers ν of E x = e (length p). */
  nu: Tensor
  /** Slacks b − A x (length m). */
  slack: Tensor
  /** ‖Qx + c + Aᵀλ + Eᵀν‖∞: stationarity of the Lagrangian. */
  stationarity: number
  /** Largest violation of A x ≤ b or E x = e. */
  primalInfeasibility: number
  /** Largest negative multiplier, as a positive number (0 when λ ≥ 0). */
  dualInfeasibility: number
  /** max |λᵢ · slackᵢ|: complementary slackness. */
  complementarity: number
  /** 1 for each inequality with slack at most `tol`, else 0; int32. */
  active: Tensor
}

/**
 * The KKT report of a point `x` with multipliers `lambda` (inequalities) and `nu` (equalities): stationarity, primal
 * and dual feasibility and complementary slackness (Nocedal and Wright, 2006, Theorem 12.1). All four are zero at an
 * optimum of a convex QP.
 */
export function kktReport(
  problem: QuadraticProgram | ParsedQP,
  x: VectorInput,
  lambda: VectorInput,
  nu: VectorInput,
  tol = 1e-7,
): KKTReport {
  const qp = 'Q' in problem && 'n' in problem ? (problem as ParsedQP) : parseQP(problem as QuadraticProgram)
  const xs = readVector(x, 'kktReport: x', qp.n)
  const l = readVector(lambda, 'kktReport: lambda', qp.A.m)
  const v = readVector(nu, 'kktReport: nu', qp.E.m)
  const g = matVec(qp.Q, xs)
  const Al = matTVec(qp.A, l)
  const Ev = matTVec(qp.E, v)
  for (let j = 0; j < qp.n; j++) g[j] += qp.c[j] + Al[j] + Ev[j]
  const Ax = matVec(qp.A, xs)
  const slack = qp.b.map((bi, i) => bi - Ax[i])
  const Ex = matVec(qp.E, xs)
  let primal = 0
  let dual = 0
  let comp = 0
  for (let i = 0; i < qp.A.m; i++) {
    primal = Math.max(primal, -slack[i])
    dual = Math.max(dual, -l[i])
    comp = Math.max(comp, Math.abs(l[i] * slack[i]))
  }
  for (let i = 0; i < qp.E.m; i++) primal = Math.max(primal, Math.abs(Ex[i] - qp.e[i]))
  return {
    x: vector(xs),
    objective: objectiveOf(qp, xs),
    lambda: vector(l),
    nu: vector(v),
    slack: vector(slack),
    stationarity: normInf(g),
    primalInfeasibility: primal,
    dualInfeasibility: dual,
    complementarity: comp,
    active: intVector(Array.from(slack, (s) => (Math.abs(s) <= tol ? 1 : 0))),
  }
}

/** Outcome of a QP solver. */
export type QuadraticProgramStatus = 'running' | 'optimal' | 'infeasible' | 'singular' | 'diverged'

// ---------------------------------------------------------------------------------------------------------------------
// Primal active-set method.

/** Options for `activeSet`. */
export interface ActiveSetOptions {
  problem: QuadraticProgram
  /** A feasible starting point. Default: a vertex of the feasible set found by the simplex method (phase 1). */
  x0?: VectorInput
  /** Tolerance for zero steps, multipliers and active constraints (default 1e-10). */
  tol?: number
}

/** What an active-set step did. */
export type ActiveSetEvent = 'start' | 'step' | 'add' | 'drop' | 'optimal'

/** One iterate of the primal active-set method. */
export interface ActiveSetState {
  /** The current feasible point, length n. */
  x: Tensor
  /** The working set: indices of inequalities held as equalities, int32, ascending. */
  working: Tensor
  /** The step direction p computed by the last step (zero when x minimises over the working set). */
  p: Tensor
  /** The step length taken along p. */
  alpha: number
  /** Multipliers of the inequalities (length m; zero outside the working set) from the last equality-QP solve. */
  lambda: Tensor
  /** Multipliers of the equalities (length p). */
  nu: Tensor
  /** The constraint added (blocking) or dropped (most negative multiplier) by the last step, or −1. */
  changed: number
  event: ActiveSetEvent
  objective: number
  iteration: number
  status: QuadraticProgramStatus
  problem: ParsedQP
  tol: number
}

/** Rows of [E; A_W] as a dense matrix. */
function workingRows(qp: ParsedQP, working: ArrayLike<number>): Mat {
  const k = qp.E.m + working.length
  const a = new Float64Array(k * qp.n)
  a.set(qp.E.a)
  for (let r = 0; r < working.length; r++)
    a.set(qp.A.a.subarray(working[r] * qp.n, (working[r] + 1) * qp.n), (qp.E.m + r) * qp.n)
  return { m: k, n: qp.n, a }
}

/** Solve the equality-constrained QP min ½pᵀQp + gᵀp s.t. M p = 0 through its KKT system; μ are M's multipliers. */
function equalityQP(qp: ParsedQP, M: Mat, g: Float64Array) {
  const n = qp.n
  const size = n + M.m
  const K = new Float64Array(size * size)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * size + j] = qp.Q.a[i * n + j]
  for (let r = 0; r < M.m; r++)
    for (let j = 0; j < n; j++) {
      K[(n + r) * size + j] = M.a[r * n + j]
      K[j * size + n + r] = M.a[r * n + j]
    }
  const rhs = new Float64Array(size)
  for (let j = 0; j < n; j++) rhs[j] = -g[j]
  const { x: sol, singular } = solveSystem(K, size, rhs)
  return { p: sol.subarray(0, n), mu: sol.subarray(n), singular }
}

/** Greedily choose inequalities active at x whose rows are independent of E's and each other's. */
function initialWorkingSet(qp: ParsedQP, x: Float64Array, tol: number): number[] {
  const Ax = matVec(qp.A, x)
  const chosen: number[] = []
  const basis: Float64Array[] = []
  const add = (row: Float64Array) => {
    const v = row.slice()
    for (let pass = 0; pass < 2; pass++)
      for (const q of basis) {
        const d = dot(v, q)
        for (let k = 0; k < v.length; k++) v[k] -= d * q[k]
      }
    const r = norm2(v)
    if (r <= 1e-9 * Math.max(norm2(row), 1e-300)) return false
    basis.push(v.map((a) => a / r))
    return true
  }
  for (let i = 0; i < qp.E.m; i++) add(qp.E.a.slice(i * qp.n, (i + 1) * qp.n))
  for (let i = 0; i < qp.A.m; i++)
    if (Math.abs(qp.b[i] - Ax[i]) <= tol * (1 + Math.abs(qp.b[i])) && add(qp.A.a.slice(i * qp.n, (i + 1) * qp.n)))
      chosen.push(i)
  return chosen
}

/** A feasible point of {A x ≤ b, E x = e} from the simplex method, or null when there is none. */
function feasiblePoint(qp: ParsedQP): Float64Array | null {
  const rows = (M: Mat) => Array.from({ length: M.m }, (_, i) => Array.from(M.a.subarray(i * M.n, (i + 1) * M.n)))
  const r = simplexSolve({
    c: new Array<number>(qp.n).fill(0),
    A_ub: rows(qp.A),
    b_ub: Array.from(qp.b),
    A_eq: rows(qp.E),
    b_eq: Array.from(qp.e),
    bounds: [null, null],
  })
  return r.status === 'optimal' ? Float64Array.from(r.x.data) : null
}

/**
 * The primal active-set method for convex QP (Nocedal and Wright, 2006, Algorithm 16.3) as a traceable algorithm.
 * Options: `{ problem, x0?, tol? }`. Each step solves the equality-constrained QP on the working set; it then either
 * moves to its minimiser (`step`), stops at a blocking constraint and adds it (`add`), drops the inequality with the
 * most negative multiplier (`drop`), or certifies optimality (`optimal`, all multipliers ≥ 0). It needs Q positive
 * definite on the null space of each working set (e.g. Q positive definite); otherwise it stops `singular`.
 */
export const activeSet: Algorithm<ActiveSetOptions, ActiveSetState> = {
  name: 'active-set-qp',
  init: (opts) => {
    const qp = parseQP(opts.problem)
    const tol = opts.tol ?? 1e-10
    const x0 = opts.x0 !== undefined ? readVector(opts.x0, 'activeSet: x0', qp.n) : feasiblePoint(qp)
    const base = {
      problem: qp,
      tol,
      p: vector(new Float64Array(qp.n)),
      alpha: 0,
      lambda: vector(new Float64Array(qp.A.m)),
      nu: vector(new Float64Array(qp.E.m)),
      changed: -1,
      event: 'start' as const,
      iteration: 0,
    }
    if (!x0) {
      const x = new Float64Array(qp.n).fill(NaN)
      return { ...base, x: vector(x), working: intVector([]), objective: NaN, status: 'infeasible' }
    }
    return {
      ...base,
      x: vector(x0),
      working: intVector(initialWorkingSet(qp, x0, Math.max(tol, 1e-9))),
      objective: objectiveOf(qp, x0),
      status: 'running',
    }
  },
  step: (s) => {
    if (s.status !== 'running') return s
    const qp = s.problem
    const x = Float64Array.from(s.x.data)
    const working = Array.from(s.working.data)
    const g = matVec(qp.Q, x)
    for (let j = 0; j < qp.n; j++) g[j] += qp.c[j]
    const { p, mu, singular } = equalityQP(qp, workingRows(qp, working), g)
    const next = { ...s, iteration: s.iteration + 1 }
    if (singular) return { ...next, status: 'singular' as const }
    const lambda = new Float64Array(qp.A.m)
    working.forEach((i, r) => (lambda[i] = mu[qp.E.m + r]))
    const nu = Float64Array.from(mu.subarray(0, qp.E.m))
    const scale = 1 + normInf(x)
    if (normInf(p) <= s.tol * scale * 1e3) {
      // x minimises over the working set: optimal if every working multiplier is non-negative, else drop the most
      // negative one (its constraint is pushing the wrong way).
      let worst = -1
      for (const i of working) if (lambda[i] < -s.tol * 1e3 && (worst < 0 || lambda[i] < lambda[worst])) worst = i
      const common = { p: vector(p), alpha: 0, lambda: vector(lambda), nu: vector(nu) }
      if (worst < 0) return { ...next, ...common, changed: -1, event: 'optimal' as const, status: 'optimal' as const }
      return {
        ...next,
        ...common,
        working: intVector(working.filter((i) => i !== worst)),
        changed: worst,
        event: 'drop' as const,
      }
    }
    // Step towards the working-set minimiser, stopping at the first inequality outside the working set it would cross.
    let alpha = 1
    let blocking = -1
    const inW = new Set(working)
    const Ap = matVec(qp.A, p)
    const Ax = matVec(qp.A, x)
    for (let i = 0; i < qp.A.m; i++) {
      if (inW.has(i) || Ap[i] <= s.tol) continue
      const t = Math.max(0, qp.b[i] - Ax[i]) / Ap[i]
      if (t < alpha) {
        alpha = t
        blocking = i
      }
    }
    for (let j = 0; j < qp.n; j++) x[j] += alpha * p[j]
    const common = {
      x: vector(x),
      p: vector(p),
      alpha,
      lambda: vector(lambda),
      nu: vector(nu),
      objective: objectiveOf(qp, x),
    }
    if (blocking < 0) return { ...next, ...common, changed: -1, event: 'step' as const }
    return {
      ...next,
      ...common,
      working: intVector([...working, blocking].sort((a, b) => a - b)),
      changed: blocking,
      event: 'add' as const,
    }
  },
  done: (s) => s.status !== 'running',
}

// ---------------------------------------------------------------------------------------------------------------------
// Interior-point method.

/** Options for `qpInteriorPoint`. */
export interface QPInteriorPointOptions {
  problem: QuadraticProgram
  /** Stop when residuals and the complementarity measure μ are below this, relative to the data (default 1e-9). */
  tol?: number
  /** Fraction of the step to the boundary taken (default 0.99). */
  stepFraction?: number
}

/** One iterate of the QP interior-point method. */
export interface QPInteriorPointState {
  x: Tensor
  /** Slacks s = b − A x of the inequalities, length m (strictly positive). */
  s: Tensor
  /** Multipliers λ of the inequalities, length m (strictly positive). */
  lambda: Tensor
  /** Multipliers ν of the equalities, length p. */
  nu: Tensor
  /** Complementarity measure μ = sᵀλ / m. */
  mu: number
  sigma: number
  alpha: number
  /** ‖Qx + c + Aᵀλ + Eᵀν‖∞. */
  stationarity: number
  /** max(‖Ax + s − b‖∞, ‖Ex − e‖∞). */
  primalResidual: number
  objective: number
  iteration: number
  converged: boolean
  diverged: boolean
  status: QuadraticProgramStatus
  problem: ParsedQP
  tol: number
  stepFraction: number
}

function qpResiduals(qp: ParsedQP, x: Float64Array, s: Float64Array, l: Float64Array, v: Float64Array) {
  const rd = matVec(qp.Q, x)
  const Al = matTVec(qp.A, l)
  const Ev = matTVec(qp.E, v)
  for (let j = 0; j < qp.n; j++) rd[j] += qp.c[j] + Al[j] + Ev[j]
  const rp = matVec(qp.A, x)
  for (let i = 0; i < qp.A.m; i++) rp[i] += s[i] - qp.b[i]
  const re = matVec(qp.E, x)
  for (let i = 0; i < qp.E.m; i++) re[i] -= qp.e[i]
  return { rd, rp, re }
}

/**
 * Solve the QP Newton system by eliminating Δs and Δλ:
 * (Q + Aᵀ S⁻¹Λ A) Δx + EᵀΔν = −r_d − Aᵀ S⁻¹(−r_sl + Λ r_p), E Δx = −r_e, then Δs = −r_p − AΔx and
 * Δλ = S⁻¹(−r_sl − Λ Δs).
 */
function qpNewton(
  qp: ParsedQP,
  s: Float64Array,
  l: Float64Array,
  rd: Float64Array,
  rp: Float64Array,
  re: Float64Array,
  rsl: Float64Array,
) {
  const { n } = qp
  const m = qp.A.m
  const p = qp.E.m
  const size = n + p
  const K = new Float64Array(size * size)
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * size + j] = qp.Q.a[i * n + j]
  const t = new Float64Array(m)
  for (let r = 0; r < m; r++) {
    const w = l[r] / s[r]
    for (let i = 0; i < n; i++) {
      const ai = qp.A.a[r * n + i]
      if (ai === 0) continue
      for (let j = 0; j < n; j++) K[i * size + j] += w * ai * qp.A.a[r * n + j]
    }
    t[r] = (-rsl[r] + l[r] * rp[r]) / s[r]
  }
  for (let r = 0; r < p; r++)
    for (let j = 0; j < n; j++) {
      K[(n + r) * size + j] = qp.E.a[r * n + j]
      K[j * size + n + r] = qp.E.a[r * n + j]
    }
  const At = matTVec(qp.A, t)
  const rhs = new Float64Array(size)
  for (let j = 0; j < n; j++) rhs[j] = -rd[j] - At[j]
  for (let r = 0; r < p; r++) rhs[n + r] = -re[r]
  const { x: sol, singular } = solveSystem(K, size, rhs)
  const dx = sol.slice(0, n)
  const dv = sol.slice(n)
  const Adx = matVec(qp.A, dx)
  const ds = new Float64Array(m)
  const dl = new Float64Array(m)
  for (let r = 0; r < m; r++) {
    ds[r] = -rp[r] - Adx[r]
    dl[r] = (-rsl[r] - l[r] * ds[r]) / s[r]
  }
  return { dx, dv, ds, dl, singular }
}

function maxStep(v: Float64Array, dv: Float64Array): number {
  let a = 1
  for (let j = 0; j < v.length; j++) if (dv[j] < 0) a = Math.min(a, -v[j] / dv[j])
  return a
}

function qpState(
  base: Pick<QPInteriorPointState, 'problem' | 'tol' | 'stepFraction'>,
  x: Float64Array,
  s: Float64Array,
  l: Float64Array,
  v: Float64Array,
  extra: { sigma: number; alpha: number; iteration: number; singular: boolean },
): QPInteriorPointState {
  const qp = base.problem
  const { rd, rp, re } = qpResiduals(qp, x, s, l, v)
  const m = qp.A.m
  const mu = m ? dot(s, l) / m : 0
  const stationarity = normInf(rd)
  const primalResidual = Math.max(normInf(rp), normInf(re))
  const scale = 1 + Math.max(normInf(qp.c), normInf(qp.b), normInf(qp.e), normInf(qp.Q.a))
  const converged = stationarity < base.tol * scale && primalResidual < base.tol * scale && mu < base.tol * scale
  const diverged =
    !converged &&
    (extra.singular || !(Math.max(normInf(x), normInf(l), normInf(v)) < 1e12) || !Number.isFinite(stationarity))
  return {
    ...base,
    ...extra,
    x: vector(x),
    s: vector(s),
    lambda: vector(l),
    nu: vector(v),
    mu,
    stationarity,
    primalResidual,
    objective: objectiveOf(qp, x),
    converged,
    diverged,
    status: converged ? 'optimal' : extra.singular ? 'singular' : diverged ? 'diverged' : 'running',
  }
}

/**
 * A primal–dual interior-point method for convex QP with Mehrotra's predictor–corrector (Nocedal and Wright, 2006,
 * §16.6) as a traceable algorithm. Options: `{ problem, tol?, stepFraction? }`. It starts from x = 0 with slacks and
 * multipliers of 1 (it need not be feasible) and each step is one predictor–corrector iteration; the iterates x trace a
 * path through the interior towards the optimum. The run is done when `converged`; it stops `diverged` when the
 * iterates grow without bound (an infeasible problem) or the Newton system is singular.
 */
export const qpInteriorPoint: Algorithm<QPInteriorPointOptions, QPInteriorPointState> = {
  name: 'interior-point-qp',
  init: (opts) => {
    const qp = parseQP(opts.problem)
    const base = { problem: qp, tol: opts.tol ?? 1e-9, stepFraction: opts.stepFraction ?? 0.99 }
    const x = new Float64Array(qp.n)
    // Slacks start at max(b − Ax, 1) so that they are positive; multipliers at 1.
    const s = qp.b.map((b) => Math.max(b, 1))
    const l = new Float64Array(qp.A.m).fill(1)
    const v = new Float64Array(qp.E.m)
    return qpState(base, x, s, l, v, { sigma: NaN, alpha: 0, iteration: 0, singular: false })
  },
  step: (st) => {
    if (st.status !== 'running') return st
    const qp = st.problem
    const x = Float64Array.from(st.x.data)
    const s = Float64Array.from(st.s.data)
    const l = Float64Array.from(st.lambda.data)
    const v = Float64Array.from(st.nu.data)
    const m = qp.A.m
    const { rd, rp, re } = qpResiduals(qp, x, s, l, v)
    const mu = m ? dot(s, l) / m : 0
    const rsl = s.map((si, r) => si * l[r])
    const aff = qpNewton(qp, s, l, rd, rp, re, rsl)
    const aAff = Math.min(maxStep(s, aff.ds), maxStep(l, aff.dl))
    let muAff = 0
    for (let r = 0; r < m; r++) muAff += (s[r] + aAff * aff.ds[r]) * (l[r] + aAff * aff.dl[r])
    muAff = m ? muAff / m : 0
    const sigma = mu > 0 ? Math.min(1, (muAff / mu) ** 3) : 0
    for (let r = 0; r < m; r++) rsl[r] = s[r] * l[r] + aff.ds[r] * aff.dl[r] - sigma * mu
    const dir = qpNewton(qp, s, l, rd, rp, re, rsl)
    const alpha = Math.min(1, st.stepFraction * Math.min(maxStep(s, dir.ds), maxStep(l, dir.dl)))
    for (let j = 0; j < qp.n; j++) x[j] += alpha * dir.dx[j]
    for (let r = 0; r < m; r++) {
      s[r] += alpha * dir.ds[r]
      l[r] += alpha * dir.dl[r]
    }
    for (let r = 0; r < qp.E.m; r++) v[r] += alpha * dir.dv[r]
    return qpState(st, x, s, l, v, {
      sigma,
      alpha,
      iteration: st.iteration + 1,
      singular: aff.singular || dir.singular,
    })
  },
  done: (s) => s.status !== 'running',
}

/** The result of `quadprog`. */
export interface QuadraticProgramResult {
  status: Exclude<QuadraticProgramStatus, 'running'> | 'limit'
  x: Tensor
  objective: number
  iterations: number
  method: 'active-set' | 'interior-point'
  /** The KKT residuals and multipliers at the returned point. */
  report: KKTReport
}

/**
 * Solve a convex quadratic program min ½xᵀQx + cᵀx s.t. Ax ≤ b, Ex = e by the primal active-set method (default) or
 * the interior-point method. The result carries the multipliers and KKT residuals.
 */
export function quadprog(
  problem: QuadraticProgram,
  options: { method?: 'active-set' | 'interior-point'; tol?: number; maxIterations?: number; x0?: VectorInput } = {},
): QuadraticProgramResult {
  const qp = parseQP(problem)
  if (options.method === 'interior-point') {
    const s = run(qpInteriorPoint, { problem, tol: options.tol }, options.maxIterations ?? 200)
    return {
      status: s.status === 'running' ? 'limit' : s.status,
      x: s.x,
      objective: s.objective,
      iterations: s.iteration,
      method: 'interior-point',
      report: kktReport(qp, s.x, s.lambda, s.nu),
    }
  }
  const s = run(activeSet, { problem, tol: options.tol, x0: options.x0 }, options.maxIterations ?? 10_000)
  return {
    status: s.status === 'running' ? 'limit' : s.status,
    x: s.x,
    objective: s.objective,
    iterations: s.iteration,
    method: 'active-set',
    report: kktReport(qp, s.x, s.lambda, s.nu),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Box-constrained QP.

/** A box-constrained QP: minimise ½xᵀQx + cᵀx subject to lower ≤ x ≤ upper (±Infinity for no bound). */
export interface BoxQuadraticProgram {
  Q: MatrixInput
  c: VectorInput
  lower: VectorInput
  upper: VectorInput
}

/** Options for `boxQP`. */
export interface BoxQPOptions {
  problem: BoxQuadraticProgram
  /** Starting point (projected onto the box). Default: the projection of 0. */
  x0?: VectorInput
  /** Stop when ‖x − P(x − ∇f)‖∞ is at most this (default 1e-10). */
  tol?: number
}

/** One iterate of the box-QP solver. */
export interface BoxQPState {
  x: Tensor
  /** ∇f = Qx + c. */
  gradient: Tensor
  /** ‖x − P(x − ∇f)‖∞: zero exactly at a KKT point. */
  projectedGradient: number
  /** Where each variable sits: −1 at its lower bound, 1 at its upper bound, 0 free; int32. */
  bounds: Tensor
  /** The point after the projected-gradient (Cauchy) half of the last step. */
  cauchy: Tensor
  /** Whether the last step's subspace Newton half was taken. */
  newton: boolean
  objective: number
  iteration: number
  converged: boolean
  lower: Tensor
  upper: Tensor
  Q: Tensor
  c: Tensor
  tol: number
}

const project = (x: Float64Array, lo: ArrayLike<number>, hi: ArrayLike<number>) =>
  x.map((v, j) => Math.min(hi[j], Math.max(lo[j], v)))

function boxState(
  base: Pick<BoxQPState, 'Q' | 'c' | 'lower' | 'upper' | 'tol'>,
  x: Float64Array,
  extra: { cauchy: Float64Array; newton: boolean; iteration: number },
): BoxQPState {
  const n = x.length
  const Q: Mat = { m: n, n, a: base.Q.data as Float64Array }
  const c = base.c.data
  const lo = base.lower.data
  const hi = base.upper.data
  const g = matVec(Q, x)
  for (let j = 0; j < n; j++) g[j] += c[j]
  let pg = 0
  const at = new Int32Array(n)
  for (let j = 0; j < n; j++) {
    pg = Math.max(pg, Math.abs(x[j] - Math.min(hi[j], Math.max(lo[j], x[j] - g[j]))))
    at[j] = x[j] <= lo[j] ? -1 : x[j] >= hi[j] ? 1 : 0
  }
  return {
    ...base,
    x: vector(x),
    gradient: vector(g),
    projectedGradient: pg,
    bounds: intVector(at),
    cauchy: vector(extra.cauchy),
    newton: extra.newton,
    objective: 0.5 * dot(x, matVec(Q, x)) + dot(c, x),
    iteration: extra.iteration,
    converged: pg <= base.tol * (1 + normInf(g)),
  }
}

/**
 * Box-constrained convex QP by projected gradient with subspace Newton steps (after Moré and Toraldo, 1991) as a
 * traceable algorithm. Options: `{ problem, x0?, tol? }`. Each step (1) takes a projected-gradient step along the
 * projection arc P(x − t∇f) with an Armijo backtracking search, which identifies the bounds that are active, and
 * (2) solves Q_FF d = −∇f_F on the free variables F and searches along P(x + t d). With Q positive definite it finds
 * the active set in finitely many steps and then converges in one Newton step. Bound multipliers at the end are
 * max(∇f, 0) at lower bounds and max(−∇f, 0) at upper bounds.
 */
export const boxQP: Algorithm<BoxQPOptions, BoxQPState> = {
  name: 'box-qp',
  init: (opts) => {
    const c = readVector(opts.problem.c, 'boxQP: c')
    const n = c.length
    const Q = readMatrix(opts.problem.Q, 'boxQP: Q', n)
    const lower = readVector(opts.problem.lower, 'boxQP: lower', n)
    const upper = readVector(opts.problem.upper, 'boxQP: upper', n)
    for (let j = 0; j < n; j++) if (!(lower[j] <= upper[j])) throw new Error(`boxQP: empty box for x${j + 1}`)
    const x0 = opts.x0 !== undefined ? readVector(opts.x0, 'boxQP: x0', n) : new Float64Array(n)
    const x = project(x0, lower, upper)
    const base = {
      Q: matrix(Q.a, n, n),
      c: vector(c),
      lower: vector(lower),
      upper: vector(upper),
      tol: opts.tol ?? 1e-10,
    }
    return boxState(base, x, { cauchy: x, newton: false, iteration: 0 })
  },
  step: (s) => {
    if (s.converged) return s
    const n = s.x.shape[0]
    const Q: Mat = { m: n, n, a: s.Q.data as Float64Array }
    const c = s.c.data
    const lo = s.lower.data
    const hi = s.upper.data
    const f = (x: Float64Array) => 0.5 * dot(x, matVec(Q, x)) + dot(c, x)
    const grad = (x: Float64Array) => {
      const g = matVec(Q, x)
      for (let j = 0; j < n; j++) g[j] += c[j]
      return g
    }
    const x = Float64Array.from(s.x.data)
    const g = grad(x)
    const fx = f(x)
    // (1) Cauchy step: backtrack along the projection arc from the exact minimiser of the unconstrained line.
    const gQg = dot(g, matVec(Q, g))
    let t = gQg > 0 ? dot(g, g) / gQg : 1
    let cauchy = x
    for (let k = 0; k < 60; k++) {
      const trial = project(
        x.map((v, j) => v - t * g[j]),
        lo,
        hi,
      )
      const decrease = dot(
        g,
        trial.map((v, j) => v - x[j]),
      )
      if (f(trial) <= fx + 1e-4 * decrease) {
        cauchy = trial
        break
      }
      t /= 2
    }
    // (2) Newton step on the variables strictly inside their bounds after the Cauchy step.
    const free: number[] = []
    for (let j = 0; j < n; j++) if (cauchy[j] > lo[j] && cauchy[j] < hi[j]) free.push(j)
    let next = cauchy
    let newton = false
    if (free.length > 0) {
      const gc = grad(cauchy)
      const k = free.length
      const H = new Float64Array(k * k)
      free.forEach((i, a) => free.forEach((j, b) => (H[a * k + b] = Q.a[i * n + j])))
      const { x: d, singular } = solveSystem(
        H,
        k,
        Float64Array.from(free, (i) => -gc[i]),
      )
      if (!singular) {
        const fc = f(cauchy)
        let step = 1
        for (let it = 0; it < 60; it++) {
          const trial = cauchy.slice()
          free.forEach((i, a) => (trial[i] += step * d[a]))
          const projected = project(trial, lo, hi)
          if (f(projected) <= fc) {
            next = projected
            newton = true
            break
          }
          step /= 2
        }
      }
    }
    return boxState(s, next, { cauchy, newton, iteration: s.iteration + 1 })
  },
  done: (s) => s.converged,
}

/** Solve a box-constrained QP (a `run` of `boxQP`); returns the final state, whose `converged` says whether it met `tol`. */
export function boxQuadprog(
  problem: BoxQuadraticProgram,
  options: { x0?: VectorInput; tol?: number; maxIterations?: number } = {},
): BoxQPState {
  return run(boxQP, { problem, x0: options.x0, tol: options.tol }, options.maxIterations ?? 1000)
}
