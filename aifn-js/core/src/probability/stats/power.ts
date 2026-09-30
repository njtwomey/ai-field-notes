/**
 * Box–Cox and Yeo–Johnson power transforms, with λ chosen by maximum likelihood, as scipy's `boxcox`/`yeojohnson` and
 * `boxcox_normmax`. The column transformer built on them (scikit-learn's `PowerTransformer`) is an application:
 * `powerTransform` in `aifn-applied/learning`.
 */

import type { Scalar } from 'aifn/foundation/contracts'
import { EPS, map, type Tensor } from 'aifn/foundation/tensor'
import { allValues, type Data } from './input'
import { varianceOf } from './descriptive'

/** Below this |λ| (or |λ − 2| for Yeo–Johnson's negative branch) the logarithmic limit is used, as scikit-learn does. */
// Below this |λ| the transforms switch to their λ → 0 (or λ → 2) limits.
const TINY = EPS

function boxCoxScalar(x: number, lambda: number): number {
  return Math.abs(lambda) < TINY ? Math.log(x) : (Math.pow(x, lambda) - 1) / lambda
}

function boxCoxInverseScalar(z: number, lambda: number): number {
  return Math.abs(lambda) < TINY ? Math.exp(z) : Math.pow(lambda * z + 1, 1 / lambda)
}

function yeoJohnsonScalar(x: number, lambda: number): number {
  if (x >= 0) return Math.abs(lambda) < TINY ? Math.log1p(x) : (Math.pow(x + 1, lambda) - 1) / lambda
  return Math.abs(lambda - 2) < TINY ? -Math.log1p(-x) : -(Math.pow(1 - x, 2 - lambda) - 1) / (2 - lambda)
}

function yeoJohnsonInverseScalar(z: number, lambda: number): number {
  if (z >= 0) return Math.abs(lambda) < TINY ? Math.expm1(z) : Math.pow(z * lambda + 1, 1 / lambda) - 1
  return Math.abs(lambda - 2) < TINY ? -Math.expm1(-z) : 1 - Math.pow(-(2 - lambda) * z + 1, 1 / (2 - lambda))
}

/** The Box–Cox transform (x^λ − 1)/λ (log x at λ = 0) of positive x, elementwise (Box and Cox, 1964, JRSS B 26). */
export function boxCox(x: number, lambda: number): number
export function boxCox(x: Tensor, lambda: number): Tensor
export function boxCox(x: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof x === 'number' ? boxCoxScalar(x, lambda) : map(x, (v) => boxCoxScalar(v, lambda))
}

/** The inverse Box–Cox transform (λz + 1)^{1/λ} (exp z at λ = 0), elementwise. */
export function boxCoxInverse(z: number, lambda: number): number
export function boxCoxInverse(z: Tensor, lambda: number): Tensor
export function boxCoxInverse(z: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof z === 'number' ? boxCoxInverseScalar(z, lambda) : map(z, (v) => boxCoxInverseScalar(v, lambda))
}

/**
 * The Yeo–Johnson transform of any real x, elementwise (Yeo and Johnson, 2000, "A new family of power transformations
 * to improve normality or symmetry", Biometrika 87): ((x + 1)^λ − 1)/λ for x ≥ 0 and −((1 − x)^{2−λ} − 1)/(2 − λ) for
 * x < 0, with logarithmic limits at λ = 0 and λ = 2.
 */
export function yeoJohnson(x: number, lambda: number): number
export function yeoJohnson(x: Tensor, lambda: number): Tensor
export function yeoJohnson(x: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof x === 'number' ? yeoJohnsonScalar(x, lambda) : map(x, (v) => yeoJohnsonScalar(v, lambda))
}

/** The inverse Yeo–Johnson transform, elementwise. */
export function yeoJohnsonInverse(z: number, lambda: number): number
export function yeoJohnsonInverse(z: Tensor, lambda: number): Tensor
export function yeoJohnsonInverse(z: Scalar | Tensor, lambda: number): Scalar | Tensor {
  return typeof z === 'number' ? yeoJohnsonInverseScalar(z, lambda) : map(z, (v) => yeoJohnsonInverseScalar(v, lambda))
}

// ── Maximum likelihood for λ ─────────────────────────────────────────────────────────────────────────────────────────

/**
 * Minimise a function of one variable: bracket a minimum by golden-ratio expansion from (a, b), then Brent's method
 * (Brent, 1973, "Algorithms for Minimization without Derivatives", ch. 5; Press et al., 2007, "Numerical Recipes",
 * §10.3), as scipy's `optimize.brent` does from `brack = (a, b)`. Private: `aifn/optim`'s `minimizeScalar`
 * sits in a sibling family of the same tier, which probability may not import.
 */
function brentMinimise(
  f: (x: number) => number,
  a0: number,
  b0: number,
  { tolerance = 1.48e-8, maxSteps = 500 }: { tolerance?: number; maxSteps?: number } = {},
): { x: number; fx: number; iterations: number; converged: boolean } {
  const GOLD = 1.618034
  let [a, b] = [a0, b0]
  let [fa, fb] = [f(a), f(b)]
  if (fb > fa) [a, b, fa, fb] = [b, a, fb, fa]
  let c = b + GOLD * (b - a)
  let fc = f(c)
  for (let grow = 0; fb > fc && grow < 100; grow++) {
    // Parabolic extrapolation limited to 100 times the step, falling back to golden expansion.
    const r = (b - a) * (fb - fc)
    const q = (b - c) * (fb - fa)
    const denom = 2 * Math.sign(q - r) * Math.max(Math.abs(q - r), 1e-21)
    let u = b - ((b - c) * q - (b - a) * r) / denom
    const ulim = b + 100 * (c - b)
    let fu: number
    if ((b - u) * (u - c) > 0) {
      fu = f(u)
      if (fu < fc) {
        ;[a, b, fa, fb] = [b, u, fb, fu]
        break
      } else if (fu > fb) {
        ;[c, fc] = [u, fu]
        break
      }
      u = c + GOLD * (c - b)
      fu = f(u)
    } else if ((c - u) * (u - ulim) > 0) {
      fu = f(u)
      if (fu < fc) {
        ;[b, c, u] = [c, u, u + GOLD * (u - c)]
        ;[fb, fc, fu] = [fc, fu, f(u)]
      }
    } else if ((u - ulim) * (ulim - c) >= 0) {
      u = ulim
      fu = f(u)
    } else {
      u = c + GOLD * (c - b)
      fu = f(u)
    }
    ;[a, b, c, fa, fb, fc] = [b, c, u, fb, fc, fu]
  }
  // Brent's method on the bracket (a, b, c) with b the lowest point.
  const CGOLD = 0.381966
  let lo = Math.min(a, c)
  let hi = Math.max(a, c)
  let [x, w, v] = [b, b, b]
  let [fx, fw, fv] = [fb, fb, fb]
  let d = 0
  let e = 0
  for (let it = 0; it < maxSteps; it++) {
    const xm = 0.5 * (lo + hi)
    const tol1 = tolerance * Math.abs(x) + 1e-11
    const tol2 = 2 * tol1
    if (Math.abs(x - xm) <= tol2 - 0.5 * (hi - lo)) return { x, fx, iterations: it, converged: true }
    if (Math.abs(e) > tol1) {
      let r = (x - w) * (fx - fv)
      let q = (x - v) * (fx - fw)
      let p = (x - v) * q - (x - w) * r
      q = 2 * (q - r)
      if (q > 0) p = -p
      q = Math.abs(q)
      const eTemp = e
      e = d
      if (Math.abs(p) >= Math.abs(0.5 * q * eTemp) || p <= q * (lo - x) || p >= q * (hi - x)) {
        e = x >= xm ? lo - x : hi - x
        d = CGOLD * e
      } else {
        d = p / q
        const u = x + d
        if (u - lo < tol2 || hi - u < tol2) d = Math.sign(xm - x) * tol1 || tol1
        r = 0
      }
    } else {
      e = x >= xm ? lo - x : hi - x
      d = CGOLD * e
    }
    const u = Math.abs(d) >= tol1 ? x + d : x + (Math.sign(d) || 1) * tol1
    const fu = f(u)
    if (fu <= fx) {
      if (u >= x) lo = x
      else hi = x
      ;[v, w, x] = [w, x, u]
      ;[fv, fw, fx] = [fw, fx, fu]
    } else {
      if (u < x) lo = u
      else hi = u
      if (fu <= fw || w === x) {
        ;[v, w, fv, fw] = [w, u, fw, fu]
      } else if (fu <= fv || v === x || v === w) {
        ;[v, fv] = [u, fu]
      }
    }
  }
  return { x, fx, iterations: maxSteps, converged: false }
}

/** The result of a λ search. */
export type PowerLambda = {
  lambda: number
  /** The profile log-likelihood at λ (up to a constant). */
  logLikelihood: number
  iterations: number
  converged: boolean
}

/**
 * The Box–Cox λ maximising the profile log-likelihood (λ − 1) Σ log xᵢ − (n/2) log σ̂²(λ), σ̂² the population
 * variance of the transformed values (Box and Cox, 1964), by Brent's method from the bracket (−2, 2) as scipy's
 * `boxcox_normmax(method='mle')`. x must be positive.
 */
export function boxCoxLambda(x: Data): PowerLambda {
  const v = Float64Array.from(allValues(x))
  if (v.some((a) => !(a > 0))) throw new Error('boxCoxLambda: Box–Cox needs positive data')
  let sumLog = 0
  for (const a of v) sumLog += Math.log(a)
  const n = v.length
  const nll = (lambda: number) => {
    const s2 = varianceOf(Float64Array.from(v, (a) => boxCoxScalar(a, lambda)), {})
    return -((lambda - 1) * sumLog - (n / 2) * Math.log(s2))
  }
  const r = brentMinimise(nll, -2, 2)
  return { lambda: r.x, logLikelihood: -r.fx, iterations: r.iterations, converged: r.converged }
}

/**
 * The Yeo–Johnson λ maximising −(n/2) log σ̂²(λ) + (λ − 1) Σ sign(xᵢ) log(1 + |xᵢ|), by Brent's method from the bracket
 * (−2, 2), as scikit-learn's `PowerTransformer`.
 */
export function yeoJohnsonLambda(x: Data): PowerLambda {
  const v = Float64Array.from(allValues(x))
  let jacobian = 0
  for (const a of v) jacobian += Math.sign(a) * Math.log1p(Math.abs(a))
  const n = v.length
  const nll = (lambda: number) => {
    const s2 = varianceOf(Float64Array.from(v, (a) => yeoJohnsonScalar(a, lambda)), {})
    return s2 === 0 ? Infinity : (n / 2) * Math.log(s2) - (lambda - 1) * jacobian
  }
  const r = brentMinimise(nll, -2, 2)
  return { lambda: r.x, logLikelihood: -r.fx, iterations: r.iterations, converged: r.converged }
}
