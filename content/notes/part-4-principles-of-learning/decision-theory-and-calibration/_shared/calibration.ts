/**
 * Small numeric helpers shared by the decision-making figures: ROC points and their convex hull, pool adjacent
 * violators, and a Newton-fitted logistic regression used for logistic (Platt) and beta calibration maps.
 */

export const logit = (p: number) => Math.log(p / (1 - p))
export const expit = (z: number) => 1 / (1 + Math.exp(-z))
export const clampProb = (p: number, eps = 1e-6) => Math.min(1 - eps, Math.max(eps, p))

export type Point = [number, number]

/**
 * Empirical ROC points (FPR, TPR) from scores and labels, one point per distinct score, from (0, 0) to (1, 1). Tied
 * scores move the curve diagonally in one step.
 */
export function rocPoints(scores: number[], labels: boolean[]): Point[] {
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a])
  const pos = labels.filter(Boolean).length
  const neg = labels.length - pos
  const out: Point[] = [[0, 0]]
  let tp = 0
  let fp = 0
  for (let k = 0; k < order.length; k++) {
    const i = order[k]
    if (labels[i]) tp++
    else fp++
    const next = order[k + 1]
    if (next === undefined || scores[next] !== scores[i]) out.push([fp / neg, tp / pos])
  }
  return out
}

/** Upper convex hull of points in ROC space, from (0, 0) to (1, 1), left to right (monotone chain). */
export function rocHull(points: Point[]): Point[] {
  const pts = [...points, [0, 0] as Point, [1, 1] as Point].sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const hull: Point[] = []
  const cross = (o: Point, a: Point, b: Point) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  for (const p of pts) {
    while (hull.length >= 2 && cross(hull[hull.length - 2], hull[hull.length - 1], p) >= 0) hull.pop()
    hull.push(p)
  }
  return hull
}

/**
 * Pool adjacent violators: the non-decreasing sequence closest to `y` in weighted least squares. `y` must already be
 * ordered by score. Returns the fitted value for each position.
 */
export function pav(y: number[], w: number[] = y.map(() => 1)): number[] {
  const blocks: { mean: number; weight: number; size: number }[] = []
  for (let i = 0; i < y.length; i++) {
    blocks.push({ mean: y[i], weight: w[i], size: 1 })
    while (blocks.length >= 2 && blocks[blocks.length - 2].mean > blocks[blocks.length - 1].mean) {
      const b = blocks.pop()!
      const a = blocks.pop()!
      const weight = a.weight + b.weight
      blocks.push({ mean: (a.mean * a.weight + b.mean * b.weight) / weight, weight, size: a.size + b.size })
    }
  }
  return blocks.flatMap((b) => new Array<number>(b.size).fill(b.mean))
}

/**
 * Isotonic calibration map fitted on (score, label) pairs: returns a step function from score to probability. Scores
 * between fitted points take the value of the nearest fitted score below (or the first one).
 */
export function fitIsotonic(scores: number[], labels: boolean[]): (s: number) => number {
  const order = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b])
  const xs = order.map((i) => scores[i])
  const fitted = pav(order.map((i) => (labels[i] ? 1 : 0)))
  return (s: number) => {
    let lo = 0
    let hi = xs.length - 1
    if (s <= xs[0]) return fitted[0]
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (xs[mid] <= s) lo = mid
      else hi = mid - 1
    }
    return fitted[lo]
  }
}

/**
 * Logistic regression by Newton's method with a small ridge penalty, for one to three features plus an intercept.
 * Returns the weights, intercept last.
 */
export function fitLogistic(features: number[][], labels: boolean[], ridge = 1e-4, iterations = 50): number[] {
  const d = features[0].length + 1
  const w = new Array<number>(d).fill(0)
  for (let it = 0; it < iterations; it++) {
    const g = new Array<number>(d).fill(0)
    const h = Array.from({ length: d }, () => new Array<number>(d).fill(0))
    for (let i = 0; i < features.length; i++) {
      const x = [...features[i], 1]
      const p = expit(x.reduce((acc, xi, j) => acc + xi * w[j], 0))
      const r = p - (labels[i] ? 1 : 0)
      const v = p * (1 - p)
      for (let j = 0; j < d; j++) {
        g[j] += r * x[j]
        for (let k = 0; k < d; k++) h[j][k] += v * x[j] * x[k]
      }
    }
    for (let j = 0; j < d - 1; j++) {
      g[j] += ridge * w[j]
      h[j][j] += ridge
    }
    const step = solve(h, g)
    let size = 0
    for (let j = 0; j < d; j++) {
      w[j] -= step[j]
      size = Math.max(size, Math.abs(step[j]))
    }
    if (size < 1e-9) break
  }
  return w
}

/** Gaussian elimination with partial pivoting for a small dense system. */
function solve(a: number[][], b: number[]): number[] {
  const n = b.length
  const m = a.map((row, i) => [...row, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r
    ;[m[c], m[p]] = [m[p], m[c]]
    const pivot = m[c][c] || 1e-12
    for (let r = c + 1; r < n; r++) {
      const f = m[r][c] / pivot
      for (let k = c; k <= n; k++) m[r][k] -= f * m[c][k]
    }
  }
  const x = new Array<number>(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r][n]
    for (let k = r + 1; k < n; k++) s -= m[r][k] * x[k]
    x[r] = s / (m[r][r] || 1e-12)
  }
  return x
}

/** Logistic calibration (Platt scaling applied to a probability score s): σ(w s + b). */
export function fitPlatt(scores: number[], labels: boolean[]): (s: number) => number {
  const [w, b] = fitLogistic(
    scores.map((s) => [s]),
    labels,
  )
  return (s: number) => expit(w * s + b)
}

/** Beta calibration: σ(a ln s − b ln(1 − s) + c), fitted as logistic regression on (ln s, −ln(1 − s)). */
export function fitBeta(scores: number[], labels: boolean[]): (s: number) => number {
  const feats = (s: number) => {
    const p = clampProb(s)
    return [Math.log(p), -Math.log(1 - p)]
  }
  const [a, b, c] = fitLogistic(scores.map(feats), labels)
  return (s: number) => {
    const [u, v] = feats(s)
    return expit(a * u + b * v + c)
  }
}

/** Mean log loss of probabilities `p` for binary labels. */
export function logLoss(p: number[], labels: boolean[]): number {
  let total = 0
  for (let i = 0; i < p.length; i++) {
    const q = clampProb(p[i], 1e-12)
    total -= labels[i] ? Math.log(q) : Math.log(1 - q)
  }
  return total / p.length
}

/** Mean squared error of probabilities `p` against binary labels (the binary Brier score). */
export function brier(p: number[], labels: boolean[]): number {
  let total = 0
  for (let i = 0; i < p.length; i++) total += (p[i] - (labels[i] ? 1 : 0)) ** 2
  return total / p.length
}
