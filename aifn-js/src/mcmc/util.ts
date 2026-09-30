/**
 * Private helpers: conversions between the public tensor surface and Float64Array working arrays, small vector
 * arithmetic, and evaluation of a target's log-density and gradient. Every helper returns new arrays.
 */

import { valueAndGrad } from 'aifn/autodiff'
import { normals, type Stream } from 'aifn/random'
import { fromData, isTensor, item, toFlat, unwrap, type Tensor, type Value } from 'aifn/tensor'
import type { Target, VectorLike } from './types'

export type F64 = Float64Array<ArrayBuffer>

/** A copy of a vector argument (a number is a vector of length 1). */
export function toF64(v: VectorLike | number, where: string): F64 {
  if (typeof v === 'number') return Float64Array.of(v)
  if (isTensor(v)) {
    if (v.shape.length > 1) throw new Error(`${where}: expected a vector, got shape [${v.shape.join(', ')}]`)
    return Float64Array.from(toFlat(v))
  }
  return Float64Array.from(v as ArrayLike<number>)
}

/** Wraps a working array as a rank-1 tensor (no copy: the array must not be written afterwards). */
export const vec = (a: F64): Tensor => fromData(a, [a.length])

/** Wraps a row-major working array as an m×n tensor (no copy). */
export const mat = (a: F64, m: number, n: number): Tensor => fromData(a, [m, n])

/** The elements of a tensor as a Float64Array (a copy unless it is a contiguous float64 tensor at offset 0). */
export function data(t: Tensor): F64 {
  if (t.dtype === 'float64' && t.offset === 0 && t.data.length === t.shape.reduce((a, b) => a * b, 1)) {
    let stride = 1
    let contiguous = true
    for (let d = t.shape.length - 1; d >= 0; d--) {
      if (t.shape[d] !== 1 && t.strides[d] !== stride) contiguous = false
      stride *= t.shape[d]
    }
    if (contiguous) return t.data as F64
  }
  return Float64Array.from(toFlat(t))
}

/** A number from a number, a one-element tensor or a traced value. */
export function toNumber(v: Value): number {
  const raw = unwrap(v)
  return typeof raw === 'number' ? raw : item(raw)
}

/** The target's log-density at x as a number (−Infinity outside the support; NaN is reported as NaN). */
export function logDensityAt(target: Target, x: F64): number {
  return toNumber(target.logDensity(vec(x)))
}

/**
 * The log-density and its gradient at x: `target.grad` when given, otherwise reverse-mode autodiff of
 * `target.logDensity` (which must then be written with aifn primitives).
 */
export function logDensityAndGrad(target: Target, x: F64): { value: number; grad: F64 } {
  if (target.grad) {
    const value = logDensityAt(target, x)
    const g = toF64(target.grad(vec(x)), 'mcmc: target.grad')
    if (g.length !== x.length)
      throw new Error(`mcmc: target.grad returned ${g.length} values for dimension ${x.length}`)
    return { value, grad: g }
  }
  const { value, grad } = valueAndGrad((theta: Tensor) => target.logDensity(theta) as Value)(vec(x))
  return { value: toNumber(value as Value), grad: toF64(grad as Tensor, 'mcmc: autodiff gradient') }
}

/** n standard normal draws as a working array. */
export const standardNormals = (s: Stream, n: number): F64 => data(normals(s, n))

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

/** a + αb. */
export function axpy(a: F64, alpha: number, b: ArrayLike<number>): F64 {
  const out = new Float64Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] + alpha * b[i]
  return out
}

export const allFinite = (a: ArrayLike<number>): boolean => {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false
  return true
}

/** Per-coordinate values from a number or an array of length n. */
export function perCoordinate(v: number | ArrayLike<number>, n: number, where: string): F64 {
  if (typeof v === 'number') return new Float64Array(n).fill(v)
  if (v.length !== n) throw new Error(`${where}: expected ${n} values, got ${v.length}`)
  return Float64Array.from(v)
}

/** log Σ exp(aᵢ), stable, with −Infinity for an empty or all −Infinity input. */
export function logSumExp(a: ArrayLike<number>): number {
  let m = -Infinity
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]
  if (m === -Infinity || !Number.isFinite(m)) return m
  let s = 0
  for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - m)
  return m + Math.log(s)
}

/** Stack equal-length rows into an m×n tensor. */
export function stackRows(rows: readonly ArrayLike<number>[], n: number): Tensor {
  const out = new Float64Array(rows.length * n)
  rows.forEach((r, i) => out.set(r, i * n))
  return mat(out, rows.length, n)
}
