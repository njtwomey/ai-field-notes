/**
 * Crowd label aggregation: majority vote, Bayesian classifier combination (one confusion matrix per worker) and a
 * community model (workers share their community's confusion matrix), both fitted by mean-field variational Bayes.
 */

export type CrowdLabel = { item: number; worker: number; label: number }

type Rand = () => number

/** Digamma ψ(x) for x > 0: shift above 6 with ψ(x) = ψ(x + 1) − 1/x, then the asymptotic series. */
export function digamma(x: number): number {
  let r = 0
  while (x < 6) {
    r -= 1 / x
    x += 1
  }
  const f = 1 / (x * x)
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))
}

const sample = (p: number[], rand: Rand) => {
  let u = rand()
  for (let i = 0; i < p.length; i++) {
    u -= p[i]
    if (u <= 0) return i
  }
  return p.length - 1
}

/** E[log π] for each row of a matrix of Dirichlet parameters. */
const expectedLog = (alpha: number[][]) =>
  alpha.map((row) => {
    const s = digamma(row.reduce((a, b) => a + b, 0))
    return row.map((a) => digamma(a) - s)
  })

const logSumExp = (logp: number[]) => {
  const m = Math.max(...logp)
  return m + Math.log(logp.reduce((s, l) => s + Math.exp(l - m), 0))
}

const normaliseLog = (logp: number[]) => {
  const z = logSumExp(logp)
  return logp.map((l) => Math.exp(l - z))
}

export type CrowdData = { labels: CrowdLabel[]; truth: number[]; community: number[] }

/**
 * Simulate `nItems` items with uniformly random true classes. Workers are listed community by community; each item is
 * labelled by `labelsPerItem` distinct workers drawn uniformly, and each worker answers from its community's confusion
 * matrix (row = true class, column = label).
 */
export function simulateCrowd(
  rand: Rand,
  nItems: number,
  confusions: number[][][],
  workersPerCommunity: number[],
  labelsPerItem: number,
): CrowdData {
  const C = confusions[0].length
  const community = workersPerCommunity.flatMap((n, m) => Array(n).fill(m))
  const K = community.length
  const truth = Array.from({ length: nItems }, () => Math.floor(rand() * C))
  const labels: CrowdLabel[] = []
  for (let i = 0; i < nItems; i++) {
    const chosen = new Set<number>()
    while (chosen.size < Math.min(labelsPerItem, K)) chosen.add(Math.floor(rand() * K))
    for (const k of chosen) labels.push({ item: i, worker: k, label: sample(confusions[community[k]][truth[i]], rand) })
  }
  return { labels, truth, community }
}

/** Majority vote; ties go to the lowest class index. */
export function majorityVote(labels: CrowdLabel[], nItems: number, C: number): number[] {
  const counts = Array.from({ length: nItems }, () => Array(C).fill(0))
  for (const l of labels) counts[l.item][l.label]++
  return counts.map((c) => c.indexOf(Math.max(...c)))
}

/** Prior pseudo-counts for one confusion matrix: `diag` on the diagonal, `off` elsewhere. */
const priorMatrix = (C: number, diag: number, off: number) =>
  Array.from({ length: C }, (_, c) => Array.from({ length: C }, (_, l) => (c === l ? diag : off)))

const itemPosteriors = (
  labels: CrowdLabel[],
  nItems: number,
  C: number,
  logRow: (l: CrowdLabel, c: number) => number,
  logP: number[],
) => {
  const logq = Array.from({ length: nItems }, () => logP.slice())
  for (const l of labels) for (let c = 0; c < C; c++) logq[l.item][c] += logRow(l, c)
  return logq.map(normaliseLog)
}

const softVotes = (labels: CrowdLabel[], nItems: number, C: number) => {
  const q = Array.from({ length: nItems }, () => Array(C).fill(1e-3))
  for (const l of labels) q[l.item][l.label] += 1
  return q.map((r) => {
    const s = r.reduce((a, b) => a + b, 0)
    return r.map((v) => v / s)
  })
}

export type BccFit = { items: number[][]; confusion: number[][][] }

/** Bayesian classifier combination: one Dirichlet-distributed confusion matrix per worker. */
export function fitBcc(
  labels: CrowdLabel[],
  nItems: number,
  nWorkers: number,
  C: number,
  iters = 30,
  prior = { diag: 4, off: 1 },
): BccFit {
  let q = softVotes(labels, nItems, C)
  let alpha: number[][][] = []
  for (let it = 0; it < iters; it++) {
    alpha = Array.from({ length: nWorkers }, () => priorMatrix(C, prior.diag, prior.off))
    const classCounts = Array(C).fill(1)
    for (const l of labels) for (let c = 0; c < C; c++) alpha[l.worker][c][l.label] += q[l.item][c]
    q.forEach((r) => r.forEach((v, c) => (classCounts[c] += v)))
    const elog = alpha.map(expectedLog)
    const logP = expectedLog([classCounts])[0]
    q = itemPosteriors(labels, nItems, C, (l, c) => elog[l.worker][c][l.label], logP)
  }
  return { items: q, confusion: alpha.map(rowNormalise) }
}

const rowNormalise = (m: number[][]) =>
  m.map((row) => {
    const s = row.reduce((a, b) => a + b, 0)
    return row.map((v) => v / s)
  })

export type CommunityFit = {
  items: number[][]
  membership: number[][]
  confusion: number[][][]
  /** Sum of the log normalisers of the item and membership updates: a fit score for choosing between restarts. */
  score: number
}

/**
 * Community model: M communities, each with one confusion matrix; every worker belongs to one community. Community
 * memberships start at random (seeded) soft assignments, which breaks the symmetry between communities.
 */
export function fitCommunities(
  labels: CrowdLabel[],
  nItems: number,
  nWorkers: number,
  C: number,
  M: number,
  rand: Rand,
  iters = 40,
  prior = { diag: 4, off: 1 },
): CommunityFit {
  let q = softVotes(labels, nItems, C)
  let r = Array.from({ length: nWorkers }, () => normaliseLog(Array.from({ length: M }, () => rand())))
  let alpha: number[][][] = []
  let score = 0
  for (let it = 0; it < iters; it++) {
    alpha = Array.from({ length: M }, () => priorMatrix(C, prior.diag, prior.off))
    for (const l of labels)
      for (let m = 0; m < M; m++) for (let c = 0; c < C; c++) alpha[m][c][l.label] += r[l.worker][m] * q[l.item][c]
    const elog = alpha.map(expectedLog)
    const sizes = Array(M).fill(1)
    r.forEach((row) => row.forEach((v, m) => (sizes[m] += v)))
    const logH = expectedLog([sizes])[0]
    // Memberships: each worker's labels scored under every community's matrix.
    const logR = Array.from({ length: nWorkers }, () => logH.slice())
    for (const l of labels)
      for (let m = 0; m < M; m++) for (let c = 0; c < C; c++) logR[l.worker][m] += q[l.item][c] * elog[m][c][l.label]
    r = logR.map(normaliseLog)
    score = logR.reduce((acc, row) => acc + logSumExp(row), 0)
    const classCounts = Array(C).fill(1)
    q.forEach((row) => row.forEach((v, c) => (classCounts[c] += v)))
    const logP = expectedLog([classCounts])[0]
    const logq = Array.from({ length: nItems }, () => logP.slice())
    for (const l of labels)
      for (let c = 0; c < C; c++) logq[l.item][c] += r[l.worker].reduce((s, w, m) => s + w * elog[m][c][l.label], 0)
    q = logq.map(normaliseLog)
    score += logq.reduce((acc, row) => acc + logSumExp(row), 0)
  }
  return { items: q, membership: r, confusion: alpha.map(rowNormalise), score }
}

export const argmax = (xs: number[]) => xs.indexOf(Math.max(...xs))

export const accuracy = (pred: number[], truth: number[]) =>
  pred.reduce((s, p, i) => s + (p === truth[i] ? 1 : 0), 0) / truth.length

/** The best of several fits from different random initial memberships, judged by `score`. */
export function fitCommunitiesRestarts(
  labels: CrowdLabel[],
  nItems: number,
  nWorkers: number,
  C: number,
  M: number,
  rand: Rand,
  restarts = 4,
): CommunityFit {
  let best: CommunityFit | null = null
  for (let s = 0; s < restarts; s++) {
    const fit = fitCommunities(labels, nItems, nWorkers, C, M, rand)
    if (!best || fit.score > best.score) best = fit
  }
  return best!
}

/** All permutations of 0..n-1. */
export function permutations(n: number): number[][] {
  if (n === 1) return [[0]]
  return permutations(n - 1).flatMap((p) =>
    Array.from({ length: n }, (_, i) => [...p.slice(0, i), n - 1, ...p.slice(i)]),
  )
}

/**
 * The relabelling of fitted communities that best matches the true ones: perm[m] is the fitted community shown as true
 * community m, chosen to maximise the summed membership probability of workers in their true community.
 */
export function matchCommunities(membership: number[][], community: number[], M: number): number[] {
  let best: number[] = []
  let bestScore = -Infinity
  for (const perm of permutations(M)) {
    const s = community.reduce((acc, m, k) => acc + membership[k][perm[m]], 0)
    if (s > bestScore) {
      bestScore = s
      best = perm
    }
  }
  return best
}
