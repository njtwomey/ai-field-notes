import { normal, stream } from 'aifn/foundation/random'
/**
 * Multiclass classification on 2-D toy data from linear logistic classifiers: one-versus-rest, one-versus-one, an
 * exhaustive error-correcting output code, and multinomial logistic regression. Everything here is small enough to
 * refit on every drag: a few hundred points, a handful of 3-parameter Newton fits and one 3K-parameter Newton fit.
 */

export type Vec2 = [number, number]

/** Linear weights [bias, w1, w2]; the score at x is bias + w1 x1 + w2 x2. */
export type Weights = [number, number, number]

export type Dataset = { x: Vec2[]; y: number[]; k: number }

export type Method = 'ovr-argmax' | 'ovr-claimed' | 'ovo' | 'ecoc' | 'ecoc-loss' | 'multinomial'

/** Class −1: no single class (a tie, or claimed by none); −2: claimed by several (one-versus-rest only). */
export const UNASSIGNED = -1
export const CONTESTED = -2

/** Ridge penalty on every weight, so separable problems still have a finite fit. */
const LAMBDA = 0.5
const NEWTON_STEPS = 12

/**
 * K Gaussian clusters with standard deviation `spread` around `centres`. The standard normal draws depend only on the
 * seed, so moving a centre translates its cluster without redrawing it.
 */
export function makeData(seed: number, centres: Vec2[], perClass: number, spread: number): Dataset {
  const g = stream(seed)
  const x: Vec2[] = []
  const y: number[] = []
  centres.forEach(([cx, cy], k) => {
    for (let i = 0; i < perClass; i++) {
      x.push([cx + spread * normal(g), cy + spread * normal(g)])
      y.push(k)
    }
  })
  return { x, y, k: centres.length }
}

const score = (w: Weights, [a, b]: Vec2) => w[0] + w[1] * a + w[2] * b

/** Solve A v = r for a small dense system by Gaussian elimination with partial pivoting. */
function solve(a: number[][], r: number[]): number[] {
  const n = r.length
  const m = a.map((row, i) => [...row, r[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let i = c + 1; i < n; i++) if (Math.abs(m[i][c]) > Math.abs(m[p][c])) p = i
    ;[m[c], m[p]] = [m[p], m[c]]
    for (let i = c + 1; i < n; i++) {
      const f = m[i][c] / m[c][c]
      for (let j = c; j <= n; j++) m[i][j] -= f * m[c][j]
    }
  }
  const v = new Array<number>(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let s = m[i][n]
    for (let j = i + 1; j < n; j++) s -= m[i][j] * v[j]
    v[i] = s / m[i][i]
  }
  return v
}

/** Ridge-penalised binary logistic regression (targets 0/1) fitted by Newton's method. */
export function fitBinary(x: Vec2[], t: number[]): Weights {
  const w: number[] = [0, 0, 0]
  for (let step = 0; step < NEWTON_STEPS; step++) {
    const grad = w.map((v) => LAMBDA * v)
    const hess = [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? LAMBDA : 0)))
    x.forEach((p, n) => {
      const f = [1, p[0], p[1]]
      const s = 1 / (1 + Math.exp(-(w[0] + w[1] * p[0] + w[2] * p[1])))
      for (let i = 0; i < 3; i++) {
        grad[i] += (s - t[n]) * f[i]
        for (let j = 0; j < 3; j++) hess[i][j] += s * (1 - s) * f[i] * f[j]
      }
    })
    const d = solve(hess, grad)
    for (let i = 0; i < 3; i++) w[i] -= d[i]
  }
  return [w[0], w[1], w[2]]
}

/** Ridge-penalised multinomial logistic regression (one weight vector per class) fitted by Newton's method. */
export function fitMultinomial(d: Dataset): Weights[] {
  const K = d.k
  const P = 3 * K
  const w = new Array<number>(P).fill(0)
  for (let step = 0; step < NEWTON_STEPS; step++) {
    const grad = w.map((v) => LAMBDA * v)
    const hess = Array.from({ length: P }, (_, i) => Array.from({ length: P }, (__, j) => (i === j ? LAMBDA : 0)))
    d.x.forEach((p, n) => {
      const f = [1, p[0], p[1]]
      const s = Array.from({ length: K }, (_, k) => w[3 * k] + w[3 * k + 1] * p[0] + w[3 * k + 2] * p[1])
      const top = Math.max(...s)
      const e = s.map((v) => Math.exp(v - top))
      const z = e.reduce((a, b) => a + b, 0)
      const q = e.map((v) => v / z)
      for (let k = 0; k < K; k++) {
        const r = q[k] - (d.y[n] === k ? 1 : 0)
        for (let i = 0; i < 3; i++) grad[3 * k + i] += r * f[i]
        for (let l = 0; l < K; l++) {
          const c = q[k] * ((k === l ? 1 : 0) - q[l])
          for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) hess[3 * k + i][3 * l + j] += c * f[i] * f[j]
        }
      }
    })
    const delta = solve(hess, grad)
    for (let i = 0; i < P; i++) w[i] -= delta[i]
  }
  return Array.from({ length: K }, (_, k): Weights => [w[3 * k], w[3 * k + 1], w[3 * k + 2]])
}

/**
 * The exhaustive code for K classes: every split of the classes into two non-empty groups, once each, with class 0
 * always coded 1. It has 2^(K−1) − 1 columns, and any two rows differ in 2^(K−2) of them.
 */
export function exhaustiveCode(K: number): number[][] {
  const cols = 2 ** (K - 1) - 1
  return Array.from({ length: K }, (_, k) =>
    Array.from({ length: cols }, (_, c) => (k === 0 ? 1 : ((c + 1) >> (K - 1 - k)) & 1 ? 0 : 1)),
  )
}

export type Fitted = {
  method: Method
  /** The binary (or, for multinomial, per-class) linear scorers. */
  weights: Weights[]
  /** For one-versus-one: the class pair [j, k] of each scorer (positive means j). */
  pairs: Vec2[]
  /** For the output code: the K × L code matrix of 0/1 bits. */
  code: number[][]
  /** Number of binary problems and the number of training examples in each. */
  problems: number
  problemSize: number
}

/** Fit every scorer a method needs. */
export function fit(d: Dataset, method: Method): Fitted {
  const K = d.k
  const base = { method, pairs: [] as Vec2[], code: [] as number[][] }
  if (method === 'multinomial') return { ...base, weights: fitMultinomial(d), problems: 0, problemSize: d.x.length }
  if (method === 'ovo') {
    const pairs: Vec2[] = []
    const weights: Weights[] = []
    let size = 0
    for (let j = 0; j < K; j++)
      for (let k = j + 1; k < K; k++) {
        const idx = d.y.flatMap((c, i) => (c === j || c === k ? [i] : []))
        weights.push(
          fitBinary(
            idx.map((i) => d.x[i]),
            idx.map((i) => (d.y[i] === j ? 1 : 0)),
          ),
        )
        pairs.push([j, k])
        size = idx.length
      }
    return { ...base, weights, pairs, problems: pairs.length, problemSize: size }
  }
  if (method === 'ecoc' || method === 'ecoc-loss') {
    const code = exhaustiveCode(K)
    const weights = code[0].map((_, c) =>
      fitBinary(
        d.x,
        d.y.map((k) => code[k][c]),
      ),
    )
    return { ...base, weights, code, problems: weights.length, problemSize: d.x.length }
  }
  const weights = Array.from({ length: K }, (_, k) =>
    fitBinary(
      d.x,
      d.y.map((c) => (c === k ? 1 : 0)),
    ),
  )
  return { ...base, weights, problems: K, problemSize: d.x.length }
}

/** The classes a method cannot choose between at x (one class when it decides), and the cell value to draw. */
export function decide(f: Fitted, p: Vec2, K: number): { classes: number[]; value: number } {
  const all = Array.from({ length: K }, (_, k) => k)
  const winners = (v: number[], best: (a: number, b: number) => boolean) => {
    let top = v[0]
    for (const s of v) if (best(s, top)) top = s
    return all.filter((k) => v[k] === top)
  }
  const single = (classes: number[]) => ({ classes, value: classes.length === 1 ? classes[0] : UNASSIGNED })
  const s = f.weights.map((w) => score(w, p))
  switch (f.method) {
    case 'multinomial':
    case 'ovr-argmax':
      return single(winners(s, (a, b) => a > b))
    case 'ovr-claimed': {
      const claim = all.filter((k) => s[k] > 0)
      if (claim.length === 1) return { classes: claim, value: claim[0] }
      if (claim.length === 0) return { classes: all, value: UNASSIGNED }
      return { classes: claim, value: CONTESTED }
    }
    case 'ovo': {
      const votes = new Array<number>(K).fill(0)
      f.pairs.forEach(([j, k], i) => (votes[s[i] > 0 ? j : k] += 1))
      return single(winners(votes, (a, b) => a > b))
    }
    case 'ecoc': {
      const dist = f.code.map((row) => row.reduce((a, bit, c) => a + (bit !== (s[c] > 0 ? 1 : 0) ? 1 : 0), 0))
      return single(winners(dist, (a, b) => a < b))
    }
    case 'ecoc-loss': {
      // Loss-based decoding: the logistic loss of each score against the codeword's bit, as a label ±1.
      const loss = f.code.map((row) => row.reduce((a, bit, c) => a + Math.log1p(Math.exp(-(2 * bit - 1) * s[c])), 0))
      return single(winners(loss, (a, b) => a < b))
    }
  }
}

/** Expected training accuracy when every tie is broken uniformly at random among the classes left in it. */
export function expectedAccuracy(f: Fitted, d: Dataset): number {
  const total = d.x.reduce((a, p, i) => {
    const { classes } = decide(f, p, d.k)
    return a + (classes.includes(d.y[i]) ? 1 / classes.length : 0)
  }, 0)
  return total / d.x.length
}
