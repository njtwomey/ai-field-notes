/**
 * Fixed points of x′ = f(x) and their linearisation: Newton's method (`aifn/solve`) from a grid of seeds, the Jacobian
 * by autodiff, eigenvalues (`aifn/ode`'s general eigensolver), the Hartman–Grobman classification, the stable and
 * unstable manifolds of saddles, and Lyapunov function checks (Strogatz, 2015, §5–6; Perko, 2001, "Differential
 * Equations and Dynamical Systems", §2.6–2.9; Khalil, 2002, "Nonlinear Systems", §4.1).
 */

import { det } from 'aifn/linalg'
import { eig, type Eigen } from 'aifn/ode'
import { solveSystem } from 'aifn/solve'
import { fromData, toFlat, type Matrix, type Vector } from 'aifn/tensor'
import { gradientAt, jacobianAt } from './calculus'
import { streamline, type Box } from './flow'
import { point2, sampleScalar, scalarValue, type Grid2, type ScalarField, type VectorField } from './grid'
import { toF64, toMatrixF64, type MatrixLike, type VectorLike } from './vector'

/** The kinds of fixed point. The planar names follow the trace–determinant plane. */
export type FixedPointKind =
  | 'stable node'
  | 'unstable node'
  | 'stable spiral'
  | 'unstable spiral'
  | 'saddle'
  | 'centre'
  | 'stable degenerate node'
  | 'unstable degenerate node'
  | 'star'
  | 'non-hyperbolic'

/** The linear stability of a fixed point. */
export type Classification = {
  kind: FixedPointKind
  /** `stable` when every eigenvalue has negative real part, `unstable` when one is positive, `marginal` otherwise. */
  stability: 'stable' | 'unstable' | 'marginal'
  /** True when no eigenvalue lies on the imaginary axis: then the linearisation decides the local phase portrait. */
  hyperbolic: boolean
  eigen: Eigen
  trace: number
  determinant: number
}

/**
 * Classifies the linearisation x′ = Jx of a fixed point from the eigenvalues of J (n × n). Hyperbolic points (no
 * eigenvalue with |Re λ| ≤ tol·(1 + ‖J‖)) are stable, unstable or saddles, named node or spiral by whether complex
 * eigenvalues occur; a planar point with a repeated eigenvalue is a star (J = λI) or degenerate node; a planar point
 * with purely imaginary eigenvalues is a centre of the linearisation (the nonlinear point may be a weak spiral).
 */
export function classifyLinear(J: MatrixLike, { tol = 1e-9 }: { tol?: number } = {}): Classification {
  const { data, m: n } = toMatrixF64(J, 'classifyLinear')
  if (data.length !== n * n) throw new Error('classifyLinear: expected a square matrix')
  const e = eig(fromData(Float64Array.from(data), [n, n]))
  const re = toFlat(e.real)
  const im = toFlat(e.imag)
  let scale = 0
  for (const v of data) scale = Math.max(scale, Math.abs(v))
  const eps = tol * (1 + scale)
  let traceJ = 0
  for (let i = 0; i < n; i++) traceJ += data[i * n + i]
  const determinant = det(fromData(Float64Array.from(data), [n, n]))
  const positive = re.filter((r) => r > eps).length
  const negative = re.filter((r) => r < -eps).length
  const hyperbolic = positive + negative === n
  const complex = im.some((v) => Math.abs(v) > eps)
  let kind: FixedPointKind
  let stability: Classification['stability']
  if (!hyperbolic) {
    stability = positive > 0 ? 'unstable' : 'marginal'
    kind = n === 2 && complex && positive === 0 ? 'centre' : 'non-hyperbolic'
  } else if (positive > 0 && negative > 0) {
    stability = 'unstable'
    kind = 'saddle'
  } else {
    const stable = negative === n
    stability = stable ? 'stable' : 'unstable'
    const repeated = n === 2 && !complex && Math.abs(re[0] - re[1]) <= Math.sqrt(eps) * (1 + Math.abs(re[0]))
    if (complex) kind = stable ? 'stable spiral' : 'unstable spiral'
    else if (repeated) {
      const offDiagonal = Math.abs(data[1]) + Math.abs(data[2]) + Math.abs(data[0] - data[3])
      kind = offDiagonal <= eps ? 'star' : stable ? 'stable degenerate node' : 'unstable degenerate node'
    } else kind = stable ? 'stable node' : 'unstable node'
  }
  return { kind, stability, hyperbolic, eigen: e, trace: traceJ, determinant }
}

/** A fixed point with its linearisation. */
export type FixedPoint = Classification & { point: Vector; jacobian: Matrix }

/** Linearises f at a fixed point x*: the Jacobian by autodiff and its classification. */
export function linearise(f: VectorField, point: VectorLike): FixedPoint {
  const J = jacobianAt(f, point)
  const p = toF64(point, 'linearise')
  return { ...classifyLinear(J), point: fromData(p, [p.length]), jacobian: J }
}

/** Options for `fixedPoints`. */
export type FixedPointOptions = {
  /** Newton starts per axis on the box (seeds on a regular grid). Default 7. */
  seeds?: number
  /** Extra starting points (k × n). */
  starts?: MatrixLike
  /** Points closer than this are the same fixed point. Default 1e-6 × the box's size. */
  mergeTol?: number
}

/**
 * The fixed points f(x*) = 0 inside a box, each linearised and classified. Damped Newton (`aifn/solve`, Jacobian by
 * autodiff) runs from a regular grid of seeds (`seeds` per axis; 7ⁿ starts) and any given `starts`; converged points
 * inside the box are merged within `mergeTol`. A fixed point whose basin of attraction for Newton misses every seed
 * is not found, so raise `seeds` for fields with many fixed points. Sorted by the first coordinate, then the second.
 */
export function fixedPoints(f: VectorField, box: Box, options: FixedPointOptions = {}): FixedPoint[] {
  const n = box.length
  const { seeds = 7 } = options
  const size = Math.max(...box.map(([lo, hi]) => hi - lo))
  const mergeTol = options.mergeTol ?? 1e-6 * size
  const starts: number[][] = []
  const total = seeds ** n
  for (let k = 0; k < total; k++) {
    let r = k
    const x = box.map(([lo, hi]) => {
      const i = r % seeds
      r = Math.floor(r / seeds)
      return lo + ((hi - lo) * (i + 0.5)) / seeds
    })
    starts.push(x)
  }
  if (options.starts) {
    const { data, m } = toMatrixF64(options.starts, 'fixedPoints', undefined, n)
    for (let i = 0; i < m; i++) starts.push(Array.from(data.subarray(i * n, (i + 1) * n)))
  }
  const F = (x: Vector) => ({ value: toF64(f(x), 'fixedPoints'), jacobian: jacobianAt(f, x) })
  const found: number[][] = []
  for (const x0 of starts) {
    let r
    try {
      r = solveSystem(F, x0, { method: 'damped-newton', maxSteps: 50, ftol: 1e-12 })
    } catch {
      continue
    }
    if (!r.converged) continue
    const x = toFlat(r.x)
    const inside = x.every((v, i) => v >= box[i][0] - mergeTol && v <= box[i][1] + mergeTol)
    if (!inside || found.some((y) => Math.hypot(...y.map((v, i) => v - x[i])) <= mergeTol)) continue
    found.push(x)
  }
  found.sort((a, b) => a[0] - b[0] || (a[1] ?? 0) - (b[1] ?? 0))
  return found.map((x) => linearise(f, x))
}

/** The branches of a fixed point's invariant manifolds, each a curve (m × n) leaving the point. */
export type Manifolds = { stable: Matrix[]; unstable: Matrix[] }

/**
 * The one-dimensional stable and unstable manifolds of a hyperbolic fixed point (both branches of each), traced from
 * x* ± ε v along each real eigenvector v, forwards in time for an unstable direction (λ > 0) and backwards for a
 * stable one (λ < 0). For a saddle in the plane these are the separatrices. Each branch stops on leaving `bounds`.
 */
export function invariantManifolds(
  f: VectorField,
  fixed: FixedPoint,
  { eps = 1e-5, t = 20, steps = 400, bounds }: { eps?: number; t?: number; steps?: number; bounds?: Box } = {},
): Manifolds {
  const re = toFlat(fixed.eigen.real)
  const im = toFlat(fixed.eigen.imag)
  const V = toFlat(fixed.eigen.vectorsReal)
  const n = re.length
  const x = toFlat(fixed.point)
  const out: Manifolds = { stable: [], unstable: [] }
  re.forEach((lambda, k) => {
    if (im[k] !== 0 || lambda === 0) return
    const v = Array.from({ length: n }, (_, i) => V[i * n + k])
    for (const s of [1, -1]) {
      const start = x.map((xi, i) => xi + s * eps * v[i])
      const curve = streamline(f, start, {
        t,
        steps,
        direction: lambda > 0 ? 'forward' : 'backward',
        bounds,
        minSpeed: 0,
      })
      // Prepend the fixed point so the branch visibly leaves it; backward branches come reversed from streamline.
      const pts = toFlat(curve)
      const m = curve.shape[0]
      const ordered =
        lambda > 0 ? pts : Array.from({ length: m }, (_, r) => pts.slice((m - 1 - r) * n, (m - r) * n)).flat()
      const data = Float64Array.from([...x, ...ordered])
      ;(lambda > 0 ? out.unstable : out.stable).push(fromData(data, [m + 1, n]))
    }
  })
  return out
}

/** The orbital derivative V̇(x) = ∇V(x)·f(x) of a candidate Lyapunov function along the flow. */
export function lyapunovDerivative(V: ScalarField, f: VectorField, x: VectorLike): number {
  const g = toFlat(gradientAt(V, x))
  const p = toF64(x, 'lyapunovDerivative')
  const fx = toF64(f(fromData(p, [p.length])), 'lyapunovDerivative')
  return g.reduce((s, v, i) => s + v * fx[i], 0)
}

/** The result of `lyapunovCheck`. */
export type LyapunovCheck = {
  x: Vector
  y: Vector
  /** V on the grid (ny × nx). */
  values: Matrix
  /** V̇ = ∇V·f on the grid (ny × nx). */
  derivative: Matrix
  /** V(x) > V(x*) at every grid point other than x* (V is positive definite about x* on the grid). */
  positiveDefinite: boolean
  /** V̇ ≤ tol at every grid point (Lyapunov stability on the region). */
  nonIncreasing: boolean
  /** V̇ < 0 at every grid point farther than one cell from x* (asymptotic stability on the region). */
  decreasing: boolean
  /** The largest V̇ on the grid away from x*, and where it occurs. */
  worst: { value: number; at: Vector }
}

/**
 * Checks Lyapunov's conditions for a candidate V about an equilibrium x* of a planar flow on a grid: V positive
 * definite (V(x) > V(x*) for x ≠ x*) and V̇ = ∇V·f ≤ 0 (stable) or < 0 away from x* (asymptotically stable). A grid
 * check is evidence on the sampled region, not a proof.
 */
export function lyapunovCheck(
  V: ScalarField,
  f: VectorField,
  equilibrium: VectorLike,
  grid: Grid2,
  { tol = 1e-12 }: { tol?: number } = {},
): LyapunovCheck {
  const eq = toF64(equilibrium, 'lyapunovCheck')
  const values = sampleScalar(V, grid)
  const xs = toFlat(values.x)
  const ys = toFlat(values.y)
  const v0 = scalarValue(V(point2(eq[0], eq[1])), 'lyapunovCheck')
  const cell = Math.hypot(
    (grid.x[1] - grid.x[0]) / Math.max(1, grid.nx - 1),
    (grid.y[1] - grid.y[0]) / Math.max(1, grid.ny - 1),
  )
  const vv = toFlat(values.values)
  const dv = new Float64Array(vv.length)
  let positiveDefinite = true
  let nonIncreasing = true
  let decreasing = true
  let worst = -Infinity
  let worstAt = [NaN, NaN]
  for (let i = 0; i < ys.length; i++)
    for (let j = 0; j < xs.length; j++) {
      const k = i * xs.length + j
      const d = lyapunovDerivative(V, f, [xs[j], ys[i]])
      dv[k] = d
      const r = Math.hypot(xs[j] - eq[0], ys[i] - eq[1])
      if (r > 1e-12 && !(vv[k] > v0)) positiveDefinite = false
      if (d > tol) nonIncreasing = false
      if (r > cell) {
        if (!(d < 0)) decreasing = false
        if (d > worst) [worst, worstAt] = [d, [xs[j], ys[i]]]
      }
    }
  return {
    x: values.x,
    y: values.y,
    values: values.values,
    derivative: fromData(dv, [ys.length, xs.length]),
    positiveDefinite,
    nonIncreasing,
    decreasing,
    worst: { value: worst, at: fromData(Float64Array.from(worstAt), [2]) },
  }
}
