/**
 * The linear-Gaussian reviewer calibration model: score = μ + quality[paper] + bias[reviewer] + noise, with Gaussian
 * priors on qualities and biases. The posterior is Gaussian, and its mean solves one small linear system.
 */

export type Review = { paper: number; reviewer: number; score: number }

export type CalibrationPrior = {
  /** Global mean score μ, treated as known. */
  mu: number
  qualityVar: number
  biasVar: number
  noiseVar: number
}

/** Solve A x = b for a small dense symmetric positive-definite A by Gaussian elimination with partial pivoting. */
export function solve(A: number[][], b: number[]): number[] {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    ;[M[c], M[p]] = [M[p], M[c]]
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c]
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  const x = Array(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n]
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]
    x[r] = s / M[r][r]
  }
  return x
}

/**
 * Posterior means of every paper's quality and every reviewer's bias. The unknowns are θ = (q_1..q_P, b_1..b_R); the
 * normal equations are (AᵀA/σ² + diag(1/α)) θ = Aᵀ(y − μ)/σ², where each row of A has a 1 for the paper and one for
 * the reviewer. This is ridge regression, and the ridge is the prior.
 */
export function calibrate(
  reviews: Review[],
  nPapers: number,
  nReviewers: number,
  p: CalibrationPrior,
): { quality: number[]; bias: number[] } {
  const n = nPapers + nReviewers
  const A = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 / (i < nPapers ? p.qualityVar : p.biasVar) : 0)),
  )
  const rhs = Array(n).fill(0)
  for (const r of reviews) {
    const i = r.paper
    const j = nPapers + r.reviewer
    A[i][i] += 1 / p.noiseVar
    A[j][j] += 1 / p.noiseVar
    A[i][j] += 1 / p.noiseVar
    A[j][i] += 1 / p.noiseVar
    rhs[i] += (r.score - p.mu) / p.noiseVar
    rhs[j] += (r.score - p.mu) / p.noiseVar
  }
  const x = solve(A, rhs)
  return { quality: x.slice(0, nPapers).map((q) => p.mu + q), bias: x.slice(nPapers) }
}

/** Ranks from 0 (smallest); ties do not occur with continuous scores. */
export function ranks(xs: number[]): number[] {
  const order = xs.map((x, i) => [x, i] as const).sort((a, b) => a[0] - b[0])
  const r = Array(xs.length).fill(0)
  order.forEach(([, i], k) => (r[i] = k))
  return r
}

/** Spearman rank correlation. */
export function spearman(a: number[], b: number[]): number {
  const ra = ranks(a)
  const rb = ranks(b)
  const n = a.length
  const d2 = ra.reduce((s, r, i) => s + (r - rb[i]) ** 2, 0)
  return 1 - (6 * d2) / (n * (n * n - 1))
}

/** Indices of the k largest values. */
export function topK(xs: number[], k: number): Set<number> {
  return new Set(
    xs
      .map((x, i) => [x, i] as const)
      .sort((a, b) => b[0] - a[0])
      .slice(0, k)
      .map(([, i]) => i),
  )
}

/** Reviewers of each paper in the figure's design: reviewers 0–2 mostly see papers 0–5, reviewers 3–5 papers 6–11. */
export const ASSIGNMENT: number[][] = [
  [0, 1, 2],
  [0, 1, 3],
  [0, 2, 4],
  [1, 2, 5],
  [0, 1, 2],
  [0, 2, 3],
  [3, 4, 5],
  [3, 4, 0],
  [3, 5, 1],
  [4, 5, 2],
  [3, 4, 5],
  [5, 4, 1],
]

/**
 * A synthetic conference: `nPapers` true qualities drawn from N(0, 1) and sorted from best to worst, plus one noise
 * draw per review. Scores are then μ + quality + bias + noise for any choice of reviewer biases.
 */
export function makeConference(normal: () => number, noiseSd: number) {
  const quality = ASSIGNMENT.map(() => normal()).sort((a, b) => b - a)
  const noise = ASSIGNMENT.map((rs) => rs.map(() => noiseSd * normal()))
  return { quality, noise }
}

export function scoreReviews(conf: { quality: number[]; noise: number[][] }, bias: number[], mu: number): Review[] {
  return ASSIGNMENT.flatMap((rs, p) =>
    rs.map((r, j) => ({ paper: p, reviewer: r, score: mu + conf.quality[p] + bias[r] + conf.noise[p][j] })),
  )
}
