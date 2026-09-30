/**
 * Complex values for LTI systems until tensors gain the `complex128` dtype (phase 3). A complex vector is a float64
 * tensor of shape [k, 2] holding (re, im) rows: the memory layout of interleaved `complex128`, so the switch will be a
 * change of dtype, not of data. Poles, zeros and frequency responses use this one shape. The scalar arithmetic
 * (`complex.add`, `complex.mul`, …) is the textbook one (Press et al., 2007, "Numerical Recipes", 3rd ed., §5.5), with
 * Smith's (1962) scaling in division.
 */

import { dense, fromData, isTensor, type Tensor } from 'aifn/foundation/tensor'
import type { Scalar, VectorLike } from 'aifn/foundation/contracts'
import { ShapeError } from 'aifn/foundation/errors'

/** A complex number. */
export type Complex = { readonly re: Scalar; readonly im: Scalar }

/** A complex input: a [k, 2] tensor of (re, im) rows, a list of complex numbers, or real values (imaginary part 0). */
export type ComplexLike = Tensor | readonly Complex[] | VectorLike

/** The complex number re + i·im. */
export const of = (re: Scalar, im: Scalar = 0): Complex => ({ re, im })
/** a + b. */
export const add = (a: Complex, b: Complex): Complex => of(a.re + b.re, a.im + b.im)
/** a − b. */
export const sub = (a: Complex, b: Complex): Complex => of(a.re - b.re, a.im - b.im)
/** a·b. */
export const mul = (a: Complex, b: Complex): Complex => of(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
/** k·a for a real k. */
export const scale = (a: Complex, k: Scalar): Complex => of(a.re * k, a.im * k)
/** a / b, scaled to avoid overflow (Smith, 1962, "Algorithm 116: Complex division", CACM 5(8)). */
export function div(a: Complex, b: Complex): Complex {
  if (Math.abs(b.re) >= Math.abs(b.im)) {
    const r = b.im / b.re
    const d = b.re + b.im * r
    return of((a.re + a.im * r) / d, (a.im - a.re * r) / d)
  }
  const r = b.re / b.im
  const d = b.re * r + b.im
  return of((a.re * r + a.im) / d, (a.im * r - a.re) / d)
}
/** The principal square root (non-negative real part; the sign of im follows a's). */
export function sqrt(a: Complex): Complex {
  const r = Math.hypot(a.re, a.im)
  const re = Math.sqrt((r + a.re) / 2)
  const im = Math.sqrt(Math.max(0, (r - a.re) / 2))
  return of(re, a.im < 0 ? -im : im)
}
/** e^a. */
export const exp = (a: Complex): Complex => scale(of(Math.cos(a.im), Math.sin(a.im)), Math.exp(a.re))
/** |a|. */
export const abs = (a: Complex): Scalar => Math.hypot(a.re, a.im)
/** The product of a list (1 when empty). */
export const product = (xs: readonly Complex[]): Complex => xs.reduce(mul, of(1))

/**
 * The coefficients, highest power first, of Π (x − r_k), in complex arithmetic (numpy's `poly`). Real when the roots
 * come in conjugate pairs; `real` returns the real parts.
 */
export function polyFromRoots(roots: readonly Complex[]): Complex[] {
  let p: Complex[] = [of(1)]
  for (const r of roots) {
    const next: Complex[] = [...p, of(0)]
    for (let i = 0; i < p.length; i++) next[i + 1] = sub(next[i + 1], mul(r, p[i]))
    p = next
  }
  return p
}

/** p(x) by Horner's rule for real coefficients, highest power first, at a complex x. */
export function horner(c: ArrayLike<number>, x: Complex): Complex {
  let pr = 0
  let pi = 0
  for (let k = 0; k < c.length; k++) {
    const r = pr * x.re - pi * x.im + c[k]
    pi = pr * x.im + pi * x.re
    pr = r
  }
  return of(pr, pi)
}

/** Complex numbers as a [k, 2] tensor of (re, im) rows. */
export function toPairs(xs: readonly Complex[]): Tensor {
  const d = new Float64Array(2 * xs.length)
  xs.forEach((z, k) => {
    d[2 * k] = z.re
    d[2 * k + 1] = z.im
  })
  return fromData(d, [xs.length, 2])
}

/** A [k, 2] tensor of (re, im) rows from real and imaginary parts of equal length. */
export function pairsOf(re: ArrayLike<number>, im: ArrayLike<number>): Tensor {
  if (re.length !== im.length) throw new ShapeError('complex', 'complex: real and imaginary parts differ in length')
  return toPairs(Array.from(re, (r, k) => of(r, im[k])))
}

/**
 * Complex numbers from a `ComplexLike`: a [k, 2] tensor is read as (re, im) rows, a rank-1 tensor or a numeric array
 * as real values, and a list of `{ re, im }` as given.
 */
export function read(x: ComplexLike, where: string): Complex[] {
  if (isTensor(x)) {
    if (x.shape.length === 2 && x.shape[1] === 2) {
      const d = dense.data(x)
      return Array.from({ length: x.shape[0] }, (_, k) => of(d[2 * k], d[2 * k + 1]))
    }
    return Array.from(dense.toF64(x, where), (r) => of(r))
  }
  const list = x as ArrayLike<unknown>
  if (list.length > 0 && typeof list[0] === 'object')
    return Array.from(list as ArrayLike<Complex>, (z) => of(z.re, z.im))
  return Array.from(dense.toF64(x as VectorLike, where), (r) => of(r))
}

/** The real and imaginary parts of a [k, 2] complex tensor, as rank-1 tensors. */
export function parts(z: Tensor): { re: Tensor; im: Tensor } {
  const xs = read(z, 'complex.parts')
  return {
    re: fromData(
      Float64Array.from(xs, (v) => v.re),
      [xs.length],
    ),
    im: fromData(
      Float64Array.from(xs, (v) => v.im),
      [xs.length],
    ),
  }
}
