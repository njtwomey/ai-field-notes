/**
 * Private helpers: small dense matrices as rows of numbers (the state-space models here have a handful of states, and
 * the filters run inside likelihood optimisations, so plain arrays keep the code close to the equations and fast),
 * plus the conversions to and from the public tensor surface.
 */

import { fromData, isTensor, toFlat, toRows, type Tensor } from 'aifn/tensor'

export type Mat = number[][]
export type Vec = number[]

/** A vector argument: a rank-1 tensor or a plain array of numbers. */
export type VectorLike = Tensor | ArrayLike<number>
/** A matrix argument: a rank-2 tensor or rows of numbers. */
export type MatrixLike = Tensor | readonly ArrayLike<number>[]

export function toVec(v: VectorLike, where: string): Vec {
  if (isTensor(v)) {
    if (v.shape.length > 1) throw new Error(`${where}: expected a vector, got shape [${v.shape.join(', ')}]`)
    return toFlat(v)
  }
  return Array.from(v as ArrayLike<number>)
}

export function toMat(a: MatrixLike | number, where: string): Mat {
  if (typeof a === 'number') return [[a]]
  if (isTensor(a)) {
    if (a.shape.length === 0) return [[toFlat(a)[0]]]
    if (a.shape.length !== 2) throw new Error(`${where}: expected a matrix, got shape [${a.shape.join(', ')}]`)
    return toRows(a)
  }
  const rows = (a as readonly ArrayLike<number>[]).map((r) => Array.from(r))
  if (rows.some((r) => r.length !== rows[0].length)) throw new Error(`${where}: ragged matrix rows`)
  return rows
}

/** Observations as T rows of m numbers: a vector is T scalar observations, a matrix is T×m. */
export function toSeries(y: VectorLike | MatrixLike, where: string): Mat {
  if (isTensor(y)) return y.shape.length === 1 ? toFlat(y).map((v) => [v]) : toMat(y, where)
  const list = y as ArrayLike<number> | readonly ArrayLike<number>[]
  if (list.length > 0 && typeof list[0] === 'number') return Array.from(list as ArrayLike<number>, (v) => [v])
  return toMat(list as readonly ArrayLike<number>[], where)
}

export const vecT = (v: Vec): Tensor => fromData(Float64Array.from(v), [v.length])
export const matT = (a: Mat): Tensor => fromData(Float64Array.from(a.flat()), [a.length, a[0]?.length ?? 0])
/** Stack T vectors of length n into a [T, n] tensor. */
export const stackVecs = (vs: Vec[], n: number): Tensor => fromData(Float64Array.from(vs.flat()), [vs.length, n])
/** Stack T n×m matrices into a [T, n, m] tensor. */
export const stackMats = (ms: Mat[], n: number, m: number): Tensor =>
  fromData(Float64Array.from(ms.flatMap((a) => a.flat())), [ms.length, n, m])

export const zerosM = (n: number, m = n): Mat => Array.from({ length: n }, () => new Array<number>(m).fill(0))
export const eyeM = (n: number): Mat =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)))
export const transpose = (a: Mat): Mat => (a.length ? a[0].map((_, j) => a.map((r) => r[j])) : [])
export const matmul = (a: Mat, b: Mat): Mat =>
  a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
export const matvec = (a: Mat, x: Vec): Vec => a.map((row) => row.reduce((s, v, k) => s + v * x[k], 0))
export const addM = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((v, j) => v + b[i][j]))
export const subM = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((v, j) => v - b[i][j]))
export const scaleM = (a: Mat, s: number): Mat => a.map((r) => r.map((v) => v * s))
export const addV = (a: Vec, b: Vec): Vec => a.map((v, i) => v + b[i])
export const subV = (a: Vec, b: Vec): Vec => a.map((v, i) => v - b[i])
export const outer = (a: Vec, b: Vec): Mat => a.map((u) => b.map((v) => u * v))
/** (A + Aᵀ)/2, to remove the asymmetry rounding leaves in a covariance. */
export const symmetrise = (a: Mat): Mat => a.map((r, i) => r.map((v, j) => (v + a[j][i]) / 2))
/** A B Aᵀ */
export const sandwich = (a: Mat, b: Mat): Mat => matmul(matmul(a, b), transpose(a))

/**
 * Solve A X = B (A n×n, B n×k) by Gaussian elimination with partial pivoting, and log|det A|. `singular` is true when
 * a pivot is at most n·ε·max|A| (then X is null): the caller reports it rather than dividing by zero.
 */
export function solveM(a: Mat, b: Mat): { x: Mat | null; logDet: number; singular: boolean } {
  const n = a.length
  const m = a.map((r) => [...r])
  const x = b.map((r) => [...r])
  let big = 0
  for (const r of a) for (const v of r) big = Math.max(big, Math.abs(v))
  const tol = n * 2.220446049250313e-16 * big
  let logDet = 0
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r
    if (!(Math.abs(m[p][c]) > tol)) return { x: null, logDet: -Infinity, singular: true }
    ;[m[c], m[p]] = [m[p], m[c]]
    ;[x[c], x[p]] = [x[p], x[c]]
    logDet += Math.log(Math.abs(m[c][c]))
    for (let r = c + 1; r < n; r++) {
      const f = m[r][c] / m[c][c]
      if (f === 0) continue
      for (let j = c; j < n; j++) m[r][j] -= f * m[c][j]
      for (let j = 0; j < x[r].length; j++) x[r][j] -= f * x[c][j]
    }
  }
  for (let r = n - 1; r >= 0; r--)
    for (let j = 0; j < x[r].length; j++) {
      let s = x[r][j]
      for (let k = r + 1; k < n; k++) s -= m[r][k] * x[k][j]
      x[r][j] = s / m[r][r]
    }
  return { x, logDet, singular: false }
}

/** A⁻¹, or null when A is singular to working precision. */
export const inverseM = (a: Mat): Mat | null => solveM(a, eyeM(a.length)).x

/**
 * A symmetric square root S with S Sᵀ = A for a symmetric positive semi-definite A, by Jacobi eigendecomposition:
 * S = V diag(√max(λ, 0)). Unlike a Cholesky factor it exists for singular A (a noise covariance with a deterministic
 * component, e.g. a zero diagonal entry), so sampling such noise gives exact zeros rather than NaN. `negative` is the
 * most negative eigenvalue found (0 for a valid covariance); callers report it.
 */
export function sqrtPsd(a: Mat): { S: Mat; negative: number } {
  const n = a.length
  const m = symmetrise(a)
  const V = eyeM(n)
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += m[i][j] ** 2
    if (off < 1e-30) break
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(m[p][q]) < 1e-300) continue
        // Rotation annihilating m[p][q] (Golub & Van Loan, 2013, §8.5.2).
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q])
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const mkp = m[k][p]
          const mkq = m[k][q]
          m[k][p] = c * mkp - s * mkq
          m[k][q] = s * mkp + c * mkq
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p][k]
          const mqk = m[q][k]
          m[p][k] = c * mpk - s * mqk
          m[q][k] = s * mpk + c * mqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p]
          const vkq = V[k][q]
          V[k][p] = c * vkp - s * vkq
          V[k][q] = s * vkp + c * vkq
        }
      }
  }
  let negative = 0
  const roots = m.map((r, i) => {
    negative = Math.min(negative, r[i])
    return Math.sqrt(Math.max(r[i], 0))
  })
  return { S: V.map((row) => row.map((v, j) => v * roots[j])), negative }
}

export const allFinite = (xs: Iterable<number>): boolean => {
  for (const v of xs) if (!Number.isFinite(v)) return false
  return true
}

export const meanOf = (x: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i]
  return s / x.length
}
