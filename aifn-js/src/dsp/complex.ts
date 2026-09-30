/**
 * Complex arrays as pairs of real tensors, and the private readers every dsp function uses. aifn has no complex dtype:
 * a complex signal is `{ re, im }`, two float64 tensors of the same shape.
 */

import { copy, fromData, isTensor, type Tensor } from 'aifn/tensor'

/** A complex array: real and imaginary parts, float64 tensors of the same shape. */
export interface ComplexTensor {
  re: Tensor
  im: Tensor
}

/** A real signal: a rank-1 tensor or an array of numbers. */
export type Signal = Tensor | ArrayLike<number>

export function isComplex(x: unknown): x is ComplexTensor {
  return typeof x === 'object' && x !== null && 're' in x && 'im' in x && isTensor((x as ComplexTensor).re)
}

/** A real signal's values as a fresh Float64Array (rank 1 required for tensors). */
export function readSignal(x: Signal, what: string): Float64Array {
  if (isTensor(x)) {
    if (x.shape.length !== 1) throw new Error(`${what}: expected a rank-1 signal, got shape [${x.shape.join(', ')}]`)
    return copy(x, 'float64').data as Float64Array
  }
  return Float64Array.from(x)
}

/** Any tensor's values (row-major) as a fresh Float64Array. */
export function readValues(x: Tensor | ArrayLike<number>): Float64Array {
  return isTensor(x) ? (copy(x, 'float64').data as Float64Array) : Float64Array.from(x)
}

/** Real and imaginary parts of a real or complex rank-1 input. */
export function readComplex(x: Signal | ComplexTensor, what: string): { re: Float64Array; im: Float64Array } {
  if (isComplex(x)) {
    const re = readSignal(x.re, what)
    const im = readSignal(x.im, what)
    if (re.length !== im.length) throw new Error(`${what}: real and imaginary parts differ in length`)
    return { re, im }
  }
  const re = readSignal(x, what)
  return { re, im: new Float64Array(re.length) }
}

/** A float64 vector tensor over `a` (no copy). */
export function vec(a: Float64Array): Tensor {
  return fromData(a)
}

export function complexOf(re: Float64Array, im: Float64Array, shape?: readonly number[]): ComplexTensor {
  return { re: fromData(re, shape ?? [re.length]), im: fromData(im, shape ?? [im.length]) }
}

/** |z| elementwise. */
export function magnitude(z: ComplexTensor): Tensor {
  const re = readValues(z.re)
  const im = readValues(z.im)
  return fromData(
    re.map((r, i) => Math.hypot(r, im[i])),
    z.re.shape,
  )
}

/** arg z in (−π, π] elementwise. */
export function phase(z: ComplexTensor): Tensor {
  const re = readValues(z.re)
  const im = readValues(z.im)
  return fromData(
    re.map((r, i) => Math.atan2(im[i], r)),
    z.re.shape,
  )
}

/** |z|² elementwise. */
export function power(z: ComplexTensor): Tensor {
  const re = readValues(z.re)
  const im = readValues(z.im)
  return fromData(
    re.map((r, i) => r * r + im[i] * im[i]),
    z.re.shape,
  )
}

/**
 * Decibels: 10 log₁₀(x / reference) for powers (`power: true`, the default) or 20 log₁₀(x / reference) for
 * amplitudes. Zero maps to −∞; there is no floor (clip for display in the figure, where the choice is visible).
 */
export function decibels(
  x: Tensor | ArrayLike<number>,
  { power: isPower = true, reference = 1 }: { power?: boolean; reference?: number } = {},
): Tensor {
  const v = readValues(x)
  const k = isPower ? 10 : 20
  const out = v.map((u) => k * Math.log10(u / reference))
  return isTensor(x) ? fromData(out, x.shape) : fromData(out)
}
