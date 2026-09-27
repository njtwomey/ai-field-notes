/** Small optimal-transport solvers for the widgets in this category. Sizes stay small (n ≤ ~60). */

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

/** -ε log Σ_k exp((pot_k - cost_k) / ε + logw_k), computed stably; `buf` is scratch space of the same length. */
function softMin(pot: Float64Array, cost: (k: number) => number, logw: Float64Array, eps: number, buf: Float64Array) {
  let m = -Infinity
  for (let k = 0; k < pot.length; k++) {
    const v = (pot[k] - cost(k)) / eps + logw[k]
    buf[k] = v
    if (v > m) m = v
  }
  if (!Number.isFinite(m)) return -eps * m
  let s = 0
  for (let k = 0; k < pot.length; k++) s += Math.exp(buf[k] - m)
  return -eps * (m + Math.log(s))
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
 * Sinkhorn iterations in the log domain: f and g are soft c-transforms of each other. Each iteration updates f, then
 * g, so column sums are exact and row sums carry the remaining error.
 */
export function sinkhorn(a: number[], b: number[], C: number[][], eps: number, iters: number, tol = 0): SinkhornResult {
  const n = a.length
  const m = b.length
  const la = Float64Array.from(a, Math.log)
  const lb = Float64Array.from(b, Math.log)
  const f = new Float64Array(n)
  const g = new Float64Array(m)
  const bufN = new Float64Array(n)
  const bufM = new Float64Array(m)
  for (let t = 0; t < iters; t++) {
    for (let i = 0; i < n; i++) f[i] = softMin(g, (j) => C[i][j], lb, eps, bufM)
    for (let j = 0; j < m; j++) g[j] = softMin(f, (i) => C[i][j], la, eps, bufN)
    // Stop early once the row sums (the only inexact ones) are within tol.
    if (tol > 0 && t % 10 === 9) {
      let err = 0
      for (let i = 0; i < n; i++) {
        let r = 0
        for (let j = 0; j < m; j++) r += a[i] * b[j] * Math.exp((f[i] + g[j] - C[i][j]) / eps)
        err += Math.abs(r - a[i])
      }
      if (err < tol) break
    }
  }
  // Plan relative to the product measure a bᵀ, as in the dual form with KL(P | a bᵀ).
  const P = a.map((ai, i) => b.map((bj, j) => ai * bj * Math.exp((f[i] + g[j] - C[i][j]) / eps)))
  let cost = 0
  let kl = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      const pij = P[i][j]
      cost += pij * C[i][j]
      if (pij > 0) kl += pij * Math.log(pij / (a[i] * b[j])) - pij + a[i] * b[j]
    }
  let marginalError = 0
  for (let i = 0; i < n; i++) marginalError += Math.abs(P[i].reduce((s, x) => s + x, 0) - a[i])
  const dual = a.reduce((s, ai, i) => s + ai * f[i], 0) + b.reduce((s, bj, j) => s + bj * g[j], 0)
  return { P, f: Array.from(f), g: Array.from(g), cost, value: cost + eps * kl, dual, marginalError }
}

/**
 * Exact optimal plan between two histograms on the same sorted 1-D grid, for any convex cost of |x - y|: the monotone
 * (north-west corner) coupling that matches quantiles. Returns the nonzero entries.
 */
export function monotonePlan(a: number[], b: number[]): { i: number; j: number; mass: number }[] {
  const ra = [...a]
  const rb = [...b]
  const out: { i: number; j: number; mass: number }[] = []
  let i = 0
  let j = 0
  while (i < ra.length && j < rb.length) {
    const mass = Math.min(ra[i], rb[j])
    if (mass > 1e-12) out.push({ i, j, mass })
    ra[i] -= mass
    rb[j] -= mass
    if (ra[i] <= 1e-12) i++
    else j++
  }
  return out
}

/** Empirical W_p^p between two equal-size 1-D samples: sort both and pair in order. */
export function wass1dSamples(x: number[], y: number[], p = 2): number {
  const xs = [...x].sort((s, t) => s - t)
  const ys = [...y].sort((s, t) => s - t)
  let s = 0
  for (let i = 0; i < xs.length; i++) s += Math.abs(xs[i] - ys[i]) ** p
  return s / xs.length
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
