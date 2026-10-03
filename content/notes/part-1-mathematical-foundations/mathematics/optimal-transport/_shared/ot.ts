import {
  sinkhorn as aifnSinkhorn,
  monotonePlan as aifnMonotonePlan,
  wasserstein1d as aifnWasserstein1d,
} from 'aifn/transport'
import { fromData, toFlat, toRows } from 'aifn/foundation/tensor'

export type Pt = [number, number]

/** Cost matrix C[i][j] = |x_i - y_j|^p for points in the plane (Euclidean norm). */
export function costMatrix(xs: Pt[], ys: Pt[], p = 2): number[][] {
  return xs.map(([a, b]) => ys.map(([c, d]) => Math.hypot(a - c, b - d) ** p))
}

/**
 * Exact assignment for a square cost matrix (Hungarian algorithm with potentials, O(n³)). Returns col[i], the column
 * assigned to row i. With uniform weights on equal numbers of points this is an optimal transport plan.
 */
export function hungarian(C: number[][]): number[] {
  const n = C.length
  const u = new Array(n + 1).fill(0)
  const v = new Array(n + 1).fill(0)
  const p = new Array(n + 1).fill(0)
  const way = new Array(n + 1).fill(0)
  for (let i = 1; i <= n; i++) {
    p[0] = i
    let j0 = 0
    const minv = new Array(n + 1).fill(Infinity)
    const used = new Array(n + 1).fill(false)
    do {
      used[j0] = true
      const i0 = p[j0]
      let delta = Infinity
      let j1 = 0
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue
        const cur = C[i0 - 1][j - 1] - u[i0] - v[j]
        if (cur < minv[j]) {
          minv[j] = cur
          way[j] = j0
        }
        if (minv[j] < delta) {
          delta = minv[j]
          j1 = j
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta
          v[j] -= delta
        } else minv[j] -= delta
      }
      j0 = j1
    } while (p[j0] !== 0)
    do {
      const j1 = way[j0]
      p[j0] = p[j1]
      j0 = j1
    } while (j0 !== 0)
  }
  const col = new Array(n).fill(0)
  for (let j = 1; j <= n; j++) if (p[j] > 0) col[p[j] - 1] = j - 1
  return col
}

export type SinkhornResult = {
  /** The plan P = diag(e^{f/ε}) K diag(e^{g/ε}). */
  P: number[][]
  f: number[]
  g: number[]
  /** Transport cost ⟨C, P⟩. */
  cost: number
  /** Entropic objective ⟨C, P⟩ + ε KL(P | a bᵀ), the regularised OT value. */
  value: number
  /** Dual value aᵀf + bᵀg. Column sums are exact after the last update, so this is the dual objective D(f, g). */
  dual: number
  /** L1 error of the row marginal after the last (column) update. */
  marginalError: number
}

/**
 * Sinkhorn iterations in the log domain, backed by aifn/transport.
 */
export function sinkhorn(a: number[], b: number[], C: number[][], eps: number, iters: number, tol = 0): SinkhornResult {
  const n = a.length
  const m = b.length
  const xC = fromData(Float64Array.from(C.flat()), [n, m])
  const res = aifnSinkhorn(Float64Array.from(a), Float64Array.from(b), xC, {
    epsilon: eps,
    maxSteps: iters,
    tolerance: tol > 0 ? tol : 1e-9,
  })
  return {
    P: toRows(res.plan) as number[][],
    f: Array.from(toFlat(res.f)),
    g: Array.from(toFlat(res.g)),
    cost: res.transportCost,
    value: res.value,
    dual: res.dual,
    marginalError: res.marginalError,
  }
}

/**
 * Exact optimal plan between two histograms on the same sorted 1-D grid, backed by aifn/transport.
 */
export function monotonePlan(a: number[], b: number[]): { i: number; j: number; mass: number }[] {
  const plan = aifnMonotonePlan(Float64Array.from(a), Float64Array.from(b))
  const is = Array.from(toFlat(plan.i))
  const js = Array.from(toFlat(plan.j))
  const masses = Array.from(toFlat(plan.mass))
  return is.map((i, k) => ({ i, j: js[k], mass: masses[k] }))
}

/** Empirical W_p^p between two equal-size 1-D samples, backed by aifn/transport. */
export function wass1dSamples(x: number[], y: number[], p = 2): number {
  return aifnWasserstein1d(Float64Array.from(x), Float64Array.from(y), { p }) ** p
}

export const normalise = (w: number[]) => {
  const s = w.reduce((t, x) => t + x, 0)
  return w.map((x) => x / s)
}

/** KL(p ‖ q) in nats; Infinity when p puts mass where q has none. */
export function klDiv(p: number[], q: number[]): number {
  let s = 0
  for (let i = 0; i < p.length; i++) {
    if (p[i] <= 0) continue
    if (q[i] <= 0) return Infinity
    s += p[i] * Math.log(p[i] / q[i])
  }
  return s
}

export function jsDiv(p: number[], q: number[]): number {
  const m = p.map((x, i) => (x + q[i]) / 2)
  return 0.5 * klDiv(p, m) + 0.5 * klDiv(q, m)
}

export function tvDist(p: number[], q: number[]): number {
  return 0.5 * p.reduce((s, x, i) => s + Math.abs(x - q[i]), 0)
}
