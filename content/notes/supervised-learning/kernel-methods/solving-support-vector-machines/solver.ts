/**
 * The linear soft-margin SVM of the worked example, solved two ways for the figures in this note: SMO with a recorded
 * trace (Platt's update, maximal-violating-pair selection) and an exact active-set solve for a chosen guess.
 */

export type Point = [number, number]
export type Status = 'zero' | 'free' | 'bound'

/** The note's dataset. Keep in step with the table in index.mdx. */
export const X: Point[] = [
  [1, 1],
  [2, 1],
  [1, 2],
  [2, 2],
  [4, 3],
  [4, 1],
]
export const Y: number[] = [-1, -1, -1, 1, 1, 1]
export const C0 = 0.5

const dot = (a: Point, b: Point) => a[0] * b[0] + a[1] * b[1]

export function gram(x: Point[]): number[][] {
  return x.map((a) => x.map((b) => dot(a, b)))
}

export function weights(x: Point[], y: number[], alpha: number[]): Point {
  const w: Point = [0, 0]
  alpha.forEach((a, i) => {
    w[0] += a * y[i] * x[i][0]
    w[1] += a * y[i] * x[i][1]
  })
  return w
}

/** Dual objective Σα − ½ αᵀQα, to be maximised. */
export function dualObjective(K: number[][], y: number[], alpha: number[]): number {
  let s = 0
  let q = 0
  for (let i = 0; i < alpha.length; i++) {
    s += alpha[i]
    for (let j = 0; j < alpha.length; j++) q += alpha[i] * alpha[j] * y[i] * y[j] * K[i][j]
  }
  return s - q / 2
}

export type SmoStep = {
  /** Indices of the pair: α_i is the first multiplier, α_j the second (the one updated by the closed form). */
  i: number
  j: number
  /** Errors E_t = f(x_t) − y_t before the step, for every point. */
  E: number[]
  eta: number
  /** α_j before the step, its unclipped optimum along the line, the box limits and the clipped value. */
  ajOld: number
  ajUnclipped: number
  L: number
  H: number
  aj: number
  aiOld: number
  ai: number
  /** Bias after the step, by Platt's rule. */
  b: number
  /** α after the step. */
  alpha: number[]
  /** Dual objective after the step. */
  objective: number
  /** KKT gap max_low E − min_up E before the step. */
  gap: number
}

export type SmoTrace = { steps: SmoStep[]; alpha: number[]; b: number; gap: number }

/**
 * SMO from α = 0, b = 0. Each step takes the maximal violating pair: i minimises E over the points whose α can move
 * to raise yᵀf (the "up" set), j maximises E over the points whose α can move the other way (the "low" set). Ties go
 * to the lower index. Stops when the gap E_j − E_i is at most `tol`.
 */
export function smo(x: Point[], y: number[], C: number, tol = 1e-9, maxIter = 500): SmoTrace {
  const n = x.length
  const K = gram(x)
  const alpha = new Array<number>(n).fill(0)
  let b = 0
  const steps: SmoStep[] = []
  let gap = Infinity
  const eps = 1e-12
  for (let it = 0; it < maxIter; it++) {
    const E = x.map((_, t) => {
      let f = b
      for (let s = 0; s < n; s++) f += alpha[s] * y[s] * K[s][t]
      return f - y[t]
    })
    let i = -1
    let j = -1
    for (let t = 0; t < n; t++) {
      const up = y[t] > 0 ? alpha[t] < C - eps : alpha[t] > eps
      const low = y[t] > 0 ? alpha[t] > eps : alpha[t] < C - eps
      if (up && (i < 0 || E[t] < E[i])) i = t
      if (low && (j < 0 || E[t] > E[j])) j = t
    }
    if (i < 0 || j < 0) break
    gap = E[j] - E[i]
    if (gap <= tol) break
    const eta = Math.max(K[i][i] + K[j][j] - 2 * K[i][j], 1e-12)
    const aiOld = alpha[i]
    const ajOld = alpha[j]
    const ajUnclipped = ajOld + (y[j] * (E[i] - E[j])) / eta
    const [L, H] =
      y[i] !== y[j]
        ? [Math.max(0, ajOld - aiOld), Math.min(C, C + ajOld - aiOld)]
        : [Math.max(0, aiOld + ajOld - C), Math.min(C, aiOld + ajOld)]
    const aj = Math.min(Math.max(ajUnclipped, L), H)
    const ai = aiOld + y[i] * y[j] * (ajOld - aj)
    const di = ai - aiOld
    const dj = aj - ajOld
    const b1 = b - E[i] - y[i] * di * K[i][i] - y[j] * dj * K[i][j]
    const b2 = b - E[j] - y[i] * di * K[i][j] - y[j] * dj * K[j][j]
    const free = (a: number) => a > eps && a < C - eps
    b = free(ai) ? b1 : free(aj) ? b2 : (b1 + b2) / 2
    alpha[i] = ai
    alpha[j] = aj
    steps.push({
      i,
      j,
      E,
      eta,
      ajOld,
      ajUnclipped,
      L,
      H,
      aj,
      aiOld,
      ai,
      b,
      alpha: [...alpha],
      objective: dualObjective(K, y, alpha),
      gap,
    })
  }
  return { steps, alpha, b, gap }
}

/**
 * The bias implied by α: the average of y_t − wᵀx_t over the free support vectors, or, with none, the middle of the
 * interval of b values that satisfy every KKT condition.
 */
export function bias(x: Point[], y: number[], alpha: number[], C: number): number {
  const w = weights(x, y, alpha)
  const eps = 1e-7
  let sum = 0
  let count = 0
  let lo = -Infinity
  let hi = Infinity
  x.forEach((p, t) => {
    const r = y[t] - dot(w, p)
    if (alpha[t] > eps && alpha[t] < C - eps) {
      sum += r
      count++
    }
    // α_t = 0 needs y_t f ≥ 1; α_t = C needs y_t f ≤ 1. Each bounds b on one side.
    const atZero = alpha[t] <= eps
    if (atZero === y[t] > 0) lo = Math.max(lo, r)
    else hi = Math.min(hi, r)
  })
  if (count > 0) return sum / count
  if (Number.isFinite(lo) && Number.isFinite(hi)) return (lo + hi) / 2
  return Number.isFinite(lo) ? lo : Number.isFinite(hi) ? hi : 0
}

/** Solves the dual to high accuracy with SMO and returns α, w and b. */
export function solve(x: Point[], y: number[], C: number) {
  const { alpha, steps } = smo(x, y, C, 1e-10, 5000)
  const w = weights(x, y, alpha)
  const b = bias(x, y, alpha, C)
  return { alpha, w, b, iterations: steps.length }
}

export type ActiveSetResult =
  | { ok: false; reason: string }
  | {
      ok: true
      alpha: number[]
      b: number
      w: Point
      /** y_t f(x_t) for every point. */
      margin: number[]
      /** Per point: whether its KKT condition holds, and the violated requirement if not. */
      checks: { holds: boolean; requirement: string }[]
    }

/**
 * Fixes α_t = 0 or C for points guessed at a bound, and solves the linear KKT system for the free α_t and b:
 * y_t f(x_t) = 1 for every free t, and Σ α_t y_t = 0. Then checks every KKT condition.
 */
export function activeSet(x: Point[], y: number[], C: number, status: Status[]): ActiveSetResult {
  const n = x.length
  const K = gram(x)
  const F = status.flatMap((s, t) => (s === 'free' ? [t] : []))
  const fixed = status.map((s) => (s === 'bound' ? C : 0))
  // Unknowns: α_F then b. Rows: one per free point, then the equality constraint.
  const m = F.length + 1
  const A = Array.from({ length: m }, () => new Array<number>(m + 1).fill(0))
  F.forEach((t, r) => {
    // y_t f(x_t) = Σ_s α_s y_t y_s K_ts + y_t b = 1.
    F.forEach((s, c) => (A[r][c] = y[t] * y[s] * K[t][s]))
    A[r][m - 1] = y[t]
    let rhs = 1
    for (let s = 0; s < n; s++) if (status[s] !== 'free') rhs -= fixed[s] * y[t] * y[s] * K[t][s]
    A[r][m] = rhs
  })
  F.forEach((s, c) => (A[m - 1][c] = y[s]))
  let rhs = 0
  for (let s = 0; s < n; s++) if (status[s] !== 'free') rhs -= fixed[s] * y[s]
  A[m - 1][m] = rhs
  if (F.length === 0) {
    return Math.abs(rhs) < 1e-9
      ? { ok: false, reason: 'No point is free, so b is not fixed by any equation. Mark one point free.' }
      : { ok: false, reason: `Σ αᵢyᵢ = ${fmt(-rhs)} ≠ 0 with no free α to correct it, so the guess is infeasible.` }
  }
  const sol = gauss(A)
  if (!sol) return { ok: false, reason: 'The KKT system is singular for this guess (for example, identical points).' }
  const alpha = [...fixed]
  F.forEach((t, c) => (alpha[t] = sol[c]))
  const b = sol[m - 1]
  const w = weights(x, y, alpha)
  const margin = x.map((p, t) => y[t] * (dot(w, p) + b))
  const tol = 1e-9
  const checks = status.map((s, t) => {
    if (s === 'zero') return { holds: margin[t] >= 1 - tol, requirement: 'y f ≥ 1' }
    if (s === 'bound') return { holds: margin[t] <= 1 + tol, requirement: 'y f ≤ 1' }
    return { holds: alpha[t] >= -tol && alpha[t] <= C + tol, requirement: '0 ≤ α ≤ C' }
  })
  return { ok: true, alpha, b, w, margin, checks }
}

function fmt(v: number) {
  return Number(v.toFixed(3)).toString()
}

/** Gaussian elimination with partial pivoting on an augmented matrix; null if singular. */
function gauss(M: number[][]): number[] | null {
  const A = M.map((r) => [...r])
  const n = A.length
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r
    if (Math.abs(A[p][c]) < 1e-10) return null
    ;[A[c], A[p]] = [A[p], A[c]]
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = A[r][c] / A[c][c]
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k]
    }
  }
  return A.map((r, i) => r[n] / r[i])
}

/** Where a point lies relative to the margin, from y f(x). */
export function position(m: number): string {
  const tol = 1e-6
  if (m > 1 + tol) return 'beyond the margin'
  if (m >= 1 - tol) return 'on the margin'
  if (m > tol) return 'inside the margin'
  if (m >= -tol) return 'on the boundary'
  return 'misclassified'
}
