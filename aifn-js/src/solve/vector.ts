/**
 * Private helpers (a copy of `optim`'s, as modules share code only through public surfaces): small dense vector and matrix arithmetic on Float64Array, and the conversions between the public
 * tensor surface and the working arrays. Every helper returns a new array; inputs are never mutated.
 */

import { fromData, isTensor, toFlat, type Tensor } from 'aifn/tensor'

/** A vector argument: a rank-1 tensor (any strides) or a plain array of numbers. Always copied on the way in. */
export type VectorLike = Tensor | ArrayLike<number>

/** A matrix argument: a rank-2 tensor or rows of numbers. Always copied on the way in. */
export type MatrixLike = Tensor | readonly ArrayLike<number>[]

export type F64 = Float64Array<ArrayBuffer>

/** A copy of a vector argument as a Float64Array (reads strided tensors correctly). */
export function toF64(v: VectorLike, where: string): F64 {
  if (isTensor(v)) {
    if (v.shape.length > 1) throw new Error(`${where}: expected a vector, got shape [${v.shape.join(', ')}]`)
    return Float64Array.from(toFlat(v))
  }
  return Float64Array.from(v as ArrayLike<number>)
}

/** A copy of a matrix argument (m×n) as a row-major Float64Array, checking its shape when `m`, `n` are given. */
export function toMatrixF64(a: MatrixLike, where: string, m?: number, n?: number): { data: F64; m: number; n: number } {
  let rows: number
  let cols: number
  let data: F64
  if (isTensor(a)) {
    if (a.shape.length !== 2) throw new Error(`${where}: expected a matrix, got shape [${a.shape.join(', ')}]`)
    ;[rows, cols] = a.shape
    data = Float64Array.from(toFlat(a))
  } else {
    const list = a as readonly ArrayLike<number>[]
    rows = list.length
    cols = rows > 0 ? list[0].length : 0
    data = new Float64Array(rows * cols)
    for (let i = 0; i < rows; i++) {
      if (list[i].length !== cols) throw new Error(`${where}: ragged matrix rows`)
      for (let j = 0; j < cols; j++) data[i * cols + j] = list[i][j]
    }
  }
  if ((m !== undefined && rows !== m) || (n !== undefined && cols !== n))
    throw new Error(`${where}: expected a ${m ?? '?'}×${n ?? '?'} matrix, got ${rows}×${cols}`)
  return { data, m: rows, n: cols }
}

/** Wraps a working array as a rank-1 tensor (no copy: the array must not be written afterwards). */
export const vec = (a: F64): Tensor => fromData(a, [a.length])

/** Wraps a row-major working array as an m×n tensor (no copy). */
export const mat = (a: F64, m: number, n: number): Tensor => fromData(a, [m, n])

/**
 * The elements of a tensor this module created (contiguous float64 at offset 0) without copying, or a copy of any
 * other tensor. Callers must not write to the result.
 */
export function data(t: Tensor): F64 {
  if (t.dtype === 'float64' && t.offset === 0 && t.data.length === t.shape.reduce((a, b) => a * b, 1)) {
    let contiguous = true
    let stride = 1
    for (let d = t.shape.length - 1; d >= 0; d--) {
      if (t.shape[d] !== 1 && t.strides[d] !== stride) contiguous = false
      stride *= t.shape[d]
    }
    if (contiguous) return t.data as F64
  }
  return Float64Array.from(toFlat(t))
}

export function dot(a: F64, b: F64): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export const norm = (a: F64): number => {
  // Scaled to avoid overflow for huge components (a diverging iterate must read as large, not Infinity).
  let scale = 0
  for (let i = 0; i < a.length; i++) scale = Math.max(scale, Math.abs(a[i]))
  if (scale === 0 || !Number.isFinite(scale)) return scale
  let s = 0
  for (let i = 0; i < a.length; i++) s += (a[i] / scale) ** 2
  return scale * Math.sqrt(s)
}

/** y + αx */
export function axpy(alpha: number, x: F64, y: F64): F64 {
  const out = new Float64Array(y.length)
  for (let i = 0; i < y.length; i++) out[i] = y[i] + alpha * x[i]
  return out
}

export function scale(alpha: number, x: F64): F64 {
  const out = new Float64Array(x.length)
  for (let i = 0; i < x.length; i++) out[i] = alpha * x[i]
  return out
}

export function sub(a: F64, b: F64): F64 {
  const out = new Float64Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i]
  return out
}

export function add(a: F64, b: F64): F64 {
  const out = new Float64Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]
  return out
}

/** A·x for a row-major m×n matrix. */
export function matVec(a: F64, x: F64, m: number, n: number): F64 {
  const out = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let s = 0
    for (let j = 0; j < n; j++) s += a[i * n + j] * x[j]
    out[i] = s
  }
  return out
}

/** Aᵀ·x for a row-major m×n matrix. */
export function matTVec(a: F64, x: F64, m: number, n: number): F64 {
  const out = new Float64Array(n)
  for (let i = 0; i < m; i++) {
    const xi = x[i]
    for (let j = 0; j < n; j++) out[j] += a[i * n + j] * xi
  }
  return out
}

/** AᵀA (n×n) for a row-major m×n matrix. */
export function gram(a: F64, m: number, n: number): F64 {
  const out = new Float64Array(n * n)
  for (let k = 0; k < m; k++)
    for (let i = 0; i < n; i++) {
      const aki = a[k * n + i]
      if (aki === 0) continue
      for (let j = 0; j < n; j++) out[i * n + j] += aki * a[k * n + j]
    }
  return out
}

export const allFinite = (a: ArrayLike<number>): boolean => {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false
  return true
}
