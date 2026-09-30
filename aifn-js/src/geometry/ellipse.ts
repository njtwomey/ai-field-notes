/**
 * Covariance and precision ellipses of 2-D Gaussians: the level set (x − μ)ᵀ Σ⁻¹ (x − μ) = k², drawn at k standard
 * deviations or at the radius that encloses a chosen probability mass.
 */

import { eigh2, type Mat2 } from 'aifn/linalg'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn/tensor'

/** A 2 × 2 matrix given as a tensor or as rows. */
export type Matrix2Input = Tensor | readonly (readonly number[])[]
/** A 2-vector given as a tensor or an array. */
export type Point2Input = Tensor | readonly number[]

/** An ellipse: its outline, centre, semi-axes and orientation. */
export interface Ellipse {
  /** Closed outline, (points + 1) × 2 (the last point repeats the first). */
  points: Tensor
  center: [number, number]
  /** Semi-axis lengths, major first. */
  radii: [number, number]
  /** Angle of the major axis from the x-axis, radians in (−π/2, π/2]. */
  angle: number
  /** The Mahalanobis radius k of the level set. */
  k: number
  /** Probability mass of a 2-D Gaussian inside the ellipse, 1 − exp(−k²/2). */
  mass: number
  /** True when the matrix was not positive definite (a radius is then NaN or infinite). */
  degenerate: boolean
}

/** Options for the ellipse functions: the level as k (standard deviations) or as a probability mass. */
export interface EllipseOptions {
  /** Mahalanobis radius. Default 1 (one standard deviation) unless `mass` is given. */
  k?: number
  /** Probability mass enclosed, in (0, 1); sets k = √(−2 ln(1 − mass)), the χ²₂ quantile. */
  mass?: number
  /** Points on the outline. Default 100. */
  points?: number
}

export function readMatrix2(m: Matrix2Input, what: string): Mat2 {
  const v = isTensor(m) ? toFlat(m) : m.flat()
  if (v.length !== 4) throw new Error(`${what}: expected a 2 × 2 matrix`)
  return [
    [v[0], v[1]],
    [v[2], v[3]],
  ]
}

export function readPoint2(p: Point2Input, what: string): [number, number] {
  const v = isTensor(p) ? toFlat(p) : p
  if (v.length !== 2) throw new Error(`${what}: expected a point of length 2`)
  return [v[0], v[1]]
}

/** The Mahalanobis radius that encloses probability `mass` of a 2-D Gaussian: √(−2 ln(1 − mass)). */
export function massToRadius(mass: number): number {
  if (!(mass > 0 && mass < 1)) throw new RangeError(`massToRadius: mass must be in (0, 1), got ${mass}`)
  return Math.sqrt(-2 * Math.log1p(-mass))
}

function level(options: EllipseOptions): number {
  if (options.mass !== undefined) return massToRadius(options.mass)
  return options.k ?? 1
}

/** Ellipse with semi-axes along the eigenvectors of a symmetric matrix with eigenvalues `axisScale(λ)`. */
function ellipseOf(
  mean: Point2Input,
  m: Matrix2Input,
  axisScale: (lambda: number) => number,
  options: EllipseOptions,
  what: string,
): Ellipse {
  const [cx, cy] = readPoint2(mean, what)
  const mat = readMatrix2(m, what)
  const k = level(options)
  const count = options.points ?? 100
  const { values, vectors } = eigh2(mat)
  // eigh2 returns eigenvalues in descending order, so the first axis is the major one for a covariance.
  let [a, b] = [k * axisScale(values[0]), k * axisScale(values[1])]
  let [u, w] = vectors
  if (b > a) {
    ;[a, b] = [b, a]
    ;[u, w] = [w, u]
  }
  const degenerate = !(values[0] > 0 && values[1] > 0)
  let angle = Math.atan2(u[1], u[0])
  if (angle > Math.PI / 2) angle -= Math.PI
  if (angle <= -Math.PI / 2) angle += Math.PI
  const out = new Float64Array((count + 1) * 2)
  for (let i = 0; i <= count; i++) {
    const t = (2 * Math.PI * (i % count)) / count
    const p = a * Math.cos(t)
    const q = b * Math.sin(t)
    out[2 * i] = cx + p * u[0] + q * w[0]
    out[2 * i + 1] = cy + p * u[1] + q * w[1]
  }
  return {
    points: fromData(out, [count + 1, 2]),
    center: [cx, cy],
    radii: [a, b],
    angle,
    k,
    mass: -Math.expm1((-k * k) / 2),
    degenerate,
  }
}

/**
 * The covariance ellipse {x : (x − μ)ᵀ Σ⁻¹ (x − μ) = k²} of a 2-D Gaussian with mean μ and covariance Σ: semi-axes
 * k√λᵢ along the eigenvectors of Σ.
 */
export function covarianceEllipse(mean: Point2Input, covariance: Matrix2Input, options: EllipseOptions = {}): Ellipse {
  return ellipseOf(mean, covariance, (l) => Math.sqrt(l), options, 'covarianceEllipse')
}

/**
 * The same level set given the precision Λ = Σ⁻¹ (e.g. a Hessian or an information matrix): semi-axes k/√λᵢ along the
 * eigenvectors of Λ.
 */
export function precisionEllipse(mean: Point2Input, precision: Matrix2Input, options: EllipseOptions = {}): Ellipse {
  return ellipseOf(mean, precision, (l) => 1 / Math.sqrt(l), options, 'precisionEllipse')
}
