import { normal, stream, uniform } from 'aifn-compute/foundation/random'

/** A small synthetic ratings problem: U users, M items, ratings 1–5 from a rank-2 taste model plus noise. */
export type Ratings = {
  users: number
  items: number
  /** Full matrix of "true" ratings, including entries no one observed. */
  truth: number[][]
  /** Observed entries used for training, as [u, i, r]. */
  train: [number, number, number][]
  /** Observed entries held out for testing. */
  test: [number, number, number][]
  mean: number
}

export function makeRatings(users = 12, items = 16, seed = 7, observed = 0.45): Ratings {
  const r = stream(seed)
  const w = Array.from({ length: users }, () => [normal(r), normal(r)])
  const v = Array.from({ length: items }, () => [normal(r), normal(r)])
  const truth = w.map((wu) => v.map((vi) => clamp(3 + 0.9 * (wu[0] * vi[0] + wu[1] * vi[1]) + 0.3 * normal(r))))
  const train: [number, number, number][] = []
  const test: [number, number, number][] = []
  for (let u = 0; u < users; u++)
    for (let i = 0; i < items; i++) {
      const p = uniform(r)
      if (p < observed * 0.75) train.push([u, i, truth[u][i]])
      else if (p < observed) test.push([u, i, truth[u][i]])
    }
  const mean = train.reduce((s, t) => s + t[2], 0) / train.length
  return { users, items, truth, train, test, mean }
}

const clamp = (x: number) => Math.round(Math.min(5, Math.max(1, x)))

/** Solve (A) x = b for a small symmetric positive-definite A by Gaussian elimination. */
function solve(A: number[][], b: number[]): number[] {
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
  const x = new Array(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n]
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]
    x[r] = s / M[r][r]
  }
  return x
}

export type Factors = { W: number[][]; V: number[][] }

export function initFactors(data: Ratings, d: number, seed = 3): Factors {
  const r = stream(seed)
  const f = () => Array.from({ length: d }, () => 0.3 * normal(r))
  return { W: Array.from({ length: data.users }, f), V: Array.from({ length: data.items }, f) }
}

/**
 * One ALS sweep: solve every user's ridge regression with V fixed, then every item's with W fixed. Ratings are centred
 * on the training mean, so the model is r̂ = mean + w_u · v_i.
 */
export function alsSweep(data: Ratings, { W, V }: Factors, lambda: number): Factors {
  const d = W[0].length
  const byUser: [number, number][][] = Array.from({ length: data.users }, () => [])
  const byItem: [number, number][][] = Array.from({ length: data.items }, () => [])
  for (const [u, i, r] of data.train) {
    byUser[u].push([i, r - data.mean])
    byItem[i].push([u, r - data.mean])
  }
  const update = (rows: [number, number][], other: number[][]) => {
    const A = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => (a === b ? lambda : 0)))
    const b = new Array(d).fill(0)
    for (const [j, y] of rows) {
      const x = other[j]
      for (let a = 0; a < d; a++) {
        b[a] += y * x[a]
        for (let c = 0; c < d; c++) A[a][c] += x[a] * x[c]
      }
    }
    return lambda === 0 && rows.length === 0 ? new Array(d).fill(0) : solve(A, b)
  }
  const W2 = byUser.map((rows) => update(rows, V))
  const V2 = byItem.map((rows) => update(rows, W2))
  return { W: W2, V: V2 }
}

/**
 * One SGD epoch over the training entries in a fixed shuffled order. The penalty λ‖w_u‖² is spread over user u's n_u
 * ratings (and likewise for items), so an epoch follows the gradient of the same objective that ALS minimises.
 */
export function sgdEpoch(data: Ratings, { W, V }: Factors, lambda: number, lr: number, order: number[]): Factors {
  const W2 = W.map((w) => [...w])
  const V2 = V.map((v) => [...v])
  const nu = new Array(data.users).fill(0)
  const ni = new Array(data.items).fill(0)
  for (const [u, i] of data.train) {
    nu[u]++
    ni[i]++
  }
  for (const k of order) {
    const [u, i, r] = data.train[k]
    const e = r - predict(data, W2[u], V2[i])
    for (let a = 0; a < W2[u].length; a++) {
      const wu = W2[u][a]
      W2[u][a] += lr * (e * V2[i][a] - (lambda / nu[u]) * wu)
      V2[i][a] += lr * (e * wu - (lambda / ni[i]) * V2[i][a])
    }
  }
  return { W: W2, V: V2 }
}

export const predict = (data: Ratings, w: number[], v: number[]) => data.mean + w.reduce((s, x, a) => s + x * v[a], 0)

export function rmse(data: Ratings, f: Factors, set: [number, number, number][]): number {
  const s = set.reduce((acc, [u, i, r]) => acc + (r - predict(data, f.W[u], f.V[i])) ** 2, 0)
  return Math.sqrt(s / set.length)
}

/** Run `steps` sweeps and return the factors after each (index 0 = initial). */
export function history(
  data: Ratings,
  d: number,
  lambda: number,
  steps: number,
  method: 'als' | 'sgd' = 'als',
  lr = 0.03,
): Factors[] {
  const out = [initFactors(data, d)]
  const r = stream(11)
  const order = data.train.map((_, k) => k)
  for (let k = order.length - 1; k > 0; k--) {
    const j = Math.floor(uniform(r) * (k + 1))
    ;[order[k], order[j]] = [order[j], order[k]]
  }
  for (let t = 0; t < steps; t++) {
    const prev = out[out.length - 1]
    out.push(method === 'als' ? alsSweep(data, prev, lambda) : sgdEpoch(data, prev, lambda, lr, order))
  }
  return out
}
