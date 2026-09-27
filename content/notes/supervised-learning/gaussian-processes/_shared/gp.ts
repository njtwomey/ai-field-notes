/**
 * Gaussian-process regression on one-dimensional inputs, for the figures in this category. Matrices are small (at
 * most a few hundred rows), so plain loops and a dense Cholesky factorisation are enough.
 */

export type Matrix = number[][]
export type Kernel = (a: number, b: number) => number

export type KernelName = 'se' | 'matern12' | 'matern32' | 'matern52' | 'rq' | 'periodic' | 'linear'

export type KernelParams = {
  /** Length-scale ℓ. */
  ell: number
  /** Signal standard deviation σ_f; the kernel's variance at zero distance is σ_f². */
  sf: number
  /** Period p of the periodic kernel. */
  period?: number
  /** Shape α of the rational quadratic kernel. */
  alpha?: number
}

export const KERNEL_OPTIONS = [
  { value: 'se' as const, label: 'squared exponential' },
  { value: 'matern32' as const, label: 'Matérn 3/2' },
  { value: 'matern12' as const, label: 'Matérn 1/2' },
  { value: 'periodic' as const, label: 'periodic' },
]

/** A covariance function on the real line. */
export function makeKernel(name: KernelName, { ell, sf, period = 1, alpha = 1 }: KernelParams): Kernel {
  const s2 = sf * sf
  switch (name) {
    case 'se':
      return (a, b) => s2 * Math.exp(-((a - b) ** 2) / (2 * ell * ell))
    case 'matern12':
      return (a, b) => s2 * Math.exp(-Math.abs(a - b) / ell)
    case 'matern32':
      return (a, b) => {
        const r = (Math.sqrt(3) * Math.abs(a - b)) / ell
        return s2 * (1 + r) * Math.exp(-r)
      }
    case 'matern52':
      return (a, b) => {
        const r = (Math.sqrt(5) * Math.abs(a - b)) / ell
        return s2 * (1 + r + (r * r) / 3) * Math.exp(-r)
      }
    case 'rq':
      return (a, b) => s2 * (1 + (a - b) ** 2 / (2 * alpha * ell * ell)) ** -alpha
    case 'periodic':
      return (a, b) => s2 * Math.exp((-2 * Math.sin((Math.PI * Math.abs(a - b)) / period) ** 2) / (ell * ell))
    case 'linear':
      return (a, b) => s2 * a * b
  }
}

export const gram = (k: Kernel, a: number[], b: number[]): Matrix => a.map((ai) => b.map((bj) => k(ai, bj)))

/** Lower-triangular L with A = L Lᵀ. A tiny floor on the pivots keeps near-singular prior covariances usable. */
export function cholesky(a: Matrix): Matrix {
  const n = a.length
  const l: Matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j]
      for (let k = 0; k < j; k++) s -= l[i][k] * l[j][k]
      l[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / l[j][j]
    }
  }
  return l
}

/** Solves L z = b for lower-triangular L. */
export function forward(l: Matrix, b: number[]): number[] {
  const n = l.length
  const z = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= l[i][k] * z[k]
    z[i] = s / l[i][i]
  }
  return z
}

/** Solves Lᵀ x = z for lower-triangular L. */
export function backward(l: Matrix, z: number[]): number[] {
  const n = l.length
  const x = new Array<number>(n)
  for (let i = n - 1; i >= 0; i--) {
    let s = z[i]
    for (let k = i + 1; k < n; k++) s -= l[k][i] * x[k]
    x[i] = s / l[i][i]
  }
  return x
}

/** Solves A x = b given the Cholesky factor L of A. */
export const cholSolve = (l: Matrix, b: number[]) => backward(l, forward(l, b))

/** log |A| from the Cholesky factor of A. */
export const logDet = (l: Matrix) => 2 * l.reduce((s, row, i) => s + Math.log(row[i]), 0)

export const addDiagonal = (a: Matrix, d: number | number[]): Matrix =>
  a.map((row, i) => row.map((v, j) => (i === j ? v + (typeof d === 'number' ? d : d[i]) : v)))

export type Posterior = {
  mean: number[]
  /** Posterior variance of the latent function f at each test input (noise excluded). */
  variance: number[]
  /** Full posterior covariance of f over the test inputs, when requested. */
  covariance?: Matrix
}

/**
 * The GP posterior at test inputs xs given noisy targets y at inputs x:
 * mean k*ᵀ(K + σ²I)⁻¹y and covariance K** − k*ᵀ(K + σ²I)⁻¹k*.
 */
export function posterior(
  k: Kernel,
  x: number[],
  y: number[],
  noiseVar: number,
  xs: number[],
  full = false,
): Posterior {
  if (x.length === 0) {
    const covariance = full ? gram(k, xs, xs) : undefined
    return { mean: xs.map(() => 0), variance: xs.map((v) => k(v, v)), covariance }
  }
  const l = cholesky(addDiagonal(gram(k, x, x), noiseVar))
  const alpha = cholSolve(l, y)
  // Column j of V is L⁻¹ k(x, xs_j), so k*ᵀ(K + σ²I)⁻¹k* = Vᵀ V.
  const v = xs.map((s) =>
    forward(
      l,
      x.map((xi) => k(xi, s)),
    ),
  )
  const mean = xs.map((s) => x.reduce((acc, xi, i) => acc + alpha[i] * k(xi, s), 0))
  const variance = xs.map((s, j) => Math.max(k(s, s) - v[j].reduce((acc, u) => acc + u * u, 0), 0))
  if (!full) return { mean, variance }
  const covariance = xs.map((a, i) => xs.map((b, j) => k(a, b) - v[i].reduce((acc, u, m) => acc + u * v[j][m], 0)))
  return { mean, variance, covariance }
}

/**
 * Draws from N(mean, cov) given a fixed matrix of standard normals, one row per sample, so that samples change
 * smoothly as the parameters change.
 */
export function samples(mean: number[], cov: Matrix, normals: number[][], jitter = 1e-6): number[][] {
  const l = cholesky(addDiagonal(cov, jitter))
  return normals.map((z) => mean.map((m, i) => m + l[i].reduce((s, lij, j) => (j <= i ? s + lij * z[j] : s), 0)))
}

export type LogMarginal = {
  /** ln p(y | X, θ). */
  value: number
  /** −½ yᵀ(K + σ²I)⁻¹y. */
  dataFit: number
  /** −½ ln |K + σ²I|. */
  complexity: number
  /** −(N/2) ln 2π. */
  constant: number
}

/** The log marginal likelihood of a zero-mean GP with Gaussian noise, split into its three terms. */
export function logMarginal(k: Kernel, x: number[], y: number[], noiseVar: number): LogMarginal {
  const l = cholesky(addDiagonal(gram(k, x, x), noiseVar))
  const z = forward(l, y)
  const dataFit = -0.5 * z.reduce((s, v) => s + v * v, 0)
  const complexity = -0.5 * logDet(l)
  const constant = -0.5 * x.length * Math.log(2 * Math.PI)
  return { value: dataFit + complexity + constant, dataFit, complexity, constant }
}

/** The largest entry of a row-major grid and its row i and column j. */
export function gridMaximum(z: number[][]): { value: number; i: number; j: number } {
  let best = { value: -Infinity, i: 0, j: 0 }
  z.forEach((row, i) =>
    row.forEach((value, j) => {
      if (value > best.value) best = { value, i, j }
    }),
  )
  return best
}
