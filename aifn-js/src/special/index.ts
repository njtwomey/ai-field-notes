/**
 * `aifn/special`: special functions and numerically stable elementary forms, as primitives.
 *
 * Every function is defined once: its scalar forward rule and derivative rules live in the internal kernel table
 * (`kernels.ts`), and each export below lifts one entry with `defineUnary`, `defineBinary` or `defineOp` from
 * `aifn/tensor`. So every function accepts numbers, tensors of any rank and traced values alike: `erf(0.5)` is a number,
 * `erf(matrix)` a matrix of the same shape, and two-argument functions broadcast their arguments (NumPy rules). Integer
 * tensors give float64 results.
 *
 * Derivatives: where a derivative is itself one of these primitives (logΓ → ψ → ψ₁, Φ → φ, softplus → σ, …) it is
 * passed as the primitive, so second derivatives work. A partial derivative the kernel table marks as unavailable (a
 * shape parameter, an order, degrees of freedom) is an error when differentiated, never a silent zero. Methods and
 * sources are documented in the implementation files.
 */

import {
  add,
  defineBinary,
  defineTernary,
  defineUnary,
  div,
  exp,
  expm1,
  logsumexp,
  mul,
  neg,
  shapeOfValue,
  square,
  sub,
  type Binary,
  type BinaryPartial,
  type Ternary,
  type Reduction,
  type Tensor,
  type Traced,
  type Unary,
  type Value,
} from 'aifn/tensor'
import {
  binaryKernels as B,
  ternaryKernels as T,
  unaryKernels as U,
  type BinaryKernel,
  type TernaryKernel,
  type UnaryKernel,
} from './kernels'

const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI)
const HALF_SQRT_PI = Math.sqrt(Math.PI) / 2
const SQRT_2PI = Math.sqrt(2 * Math.PI)

// ── Lifting helpers ──────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Lift a unary kernel. `derivative(x, y)` is dy/dx written with primitives, for second derivatives; without it the
 * kernel's scalar `df` gives first derivatives and differentiating again is an error.
 */
function unary(k: UnaryKernel, derivative?: (x: Value, y: Value) => Value): Unary {
  return defineUnary(k.name, k.f, k.df, derivative ? { derivative } : {})
}

/** A partial derivative ∂y/∂(argument) of a binary primitive at (a, b) with output y, written with primitives. */
type Partial2 = BinaryPartial

/**
 * Lift a binary kernel. With `partials` (one per argument, written with primitives, `null` where the kernel has
 * none), the derivative is itself differentiable; otherwise the kernel's scalar partials give first derivatives only.
 * Either way, a partial the kernel marks as unavailable throws when its argument is differentiated.
 */
function binary(k: BinaryKernel, partials?: [Partial2 | null, Partial2 | null]): Binary {
  return defineBinary(k.name, k.f, k.dfa, k.dfb, partials ? { derivative: partials } : {})
}

export type { Ternary }

/**
 * Lift a ternary kernel (the incomplete beta family, differentiable in the last argument only). The partial is a
 * scalar rule, so second derivatives are an error.
 */
function ternary(k: TernaryKernel): Ternary {
  return defineTernary(k.name, k.f, ...k.partials)
}

// ── Error function family (erf.ts) ───────────────────────────────────────────────────────────────────────────────────

/** erf(x) = (2/√π) ∫₀ˣ e^{−t²} dt, elementwise; relative error about 1e-15. */
export const erf: Unary = unary(U.erf, (x) => mul(TWO_OVER_SQRT_PI, exp(neg(square(x)))))
/** erfc(x) = 1 − erf(x), elementwise, with full relative accuracy in the upper tail (to underflow near x = 27). */
export const erfc: Unary = unary(U.erfc, (x) => mul(-TWO_OVER_SQRT_PI, exp(neg(square(x)))))
/** erfcx(x) = e^{x²} erfc(x), the scaled complement, elementwise; no underflow for large x. */
export const erfcx: Unary = unary(U.erfcx, (x, y) => sub(mul(mul(2, x), y), TWO_OVER_SQRT_PI))
/** log erfc(x), elementwise, accurate far beyond the underflow of erfc. */
export const logErfc: Unary = unary(U.logErfc, (x) => div(-TWO_OVER_SQRT_PI, erfcx(x)))
/** The inverse error function on [−1, 1], elementwise. */
export const erfinv: Unary = unary(U.erfinv, (_x, y) => mul(HALF_SQRT_PI, exp(square(y))))
/** The inverse complementary error function on [0, 2], elementwise. */
export const erfcinv: Unary = unary(U.erfcinv, (_x, y) => mul(-HALF_SQRT_PI, exp(square(y))))

// ── Standard normal (normal.ts) ──────────────────────────────────────────────────────────────────────────────────────

/** The standard normal density φ(z), elementwise. */
export const normalPdf: Unary = unary(U.normalPdf, (x, y) => neg(mul(x, y)))
/** log φ(z), elementwise. */
export const normalLogPdf: Unary = unary(U.normalLogPdf, (x) => neg(x))
/**
 * The standard normal cdf Φ(z), elementwise, relatively accurate in the lower tail to underflow; use Φ(−z) for upper
 * tails.
 */
export const normalCdf: Unary = unary(U.normalCdf, (x) => normalPdf(x))
/** log Φ(z), elementwise, accurate in both tails (no underflow in the lower tail). */
export const normalLogCdf: Unary = unary(U.normalLogCdf, (x) => truncatedNormalV(x))
/** The standard normal quantile Φ⁻¹(p), elementwise (Wichura 1988, AS 241). */
export const normalQuantile: Unary = unary(U.normalQuantile, (_p, y) => mul(SQRT_2PI, exp(mul(0.5, square(y)))))
/** log(Φ(u) − Φ(l)) for l ≤ u, elementwise over broadcast (l, u), without cancellation in either tail. */
export const normalLogIntervalProbability: Binary = binary(B.normalLogIntervalProbability, [
  (l, _u, y) => neg(exp(sub(normalLogPdf(l), y))),
  (_l, u, y) => exp(sub(normalLogPdf(u), y)),
])
/** v(t) = φ(t)/Φ(t), elementwise: the mean of a standard normal truncated to (−t, ∞); accurate in both tails. */
export const truncatedNormalV: Unary = unary(U.truncatedNormalV, (t) => neg(truncatedNormalW(t)))
/**
 * w(t) = v(t)(v(t) + t) in (0, 1), elementwise: one minus the variance of the same truncation; accurate in both
 * tails.
 */
export const truncatedNormalW: Unary = unary(U.truncatedNormalW, (t, w) => {
  // w' = v − w(2v + t), from w = v(v + t) and v' = −w.
  const v = truncatedNormalV(t)
  return sub(v, mul(w, add(mul(2, v), t)))
})
/**
 * The mean of a standard normal truncated to [−ε − t, ε − t] (the TrueSkill draw factor), elementwise over broadcast
 * (t, ε). Differentiable in t only.
 */
export const truncatedNormalVDraw: Binary = binary(B.truncatedNormalVDraw, [
  (t, eps) => neg(truncatedNormalWDraw(t, eps)),
  null,
])
/**
 * One minus the variance of a standard normal truncated to [−ε − t, ε − t], elementwise over broadcast (t, ε). Not
 * differentiable.
 */
export const truncatedNormalWDraw: Binary = binary(B.truncatedNormalWDraw)

// ── Gamma family (gamma.ts) ──────────────────────────────────────────────────────────────────────────────────────────

/** log |Γ(x)|, elementwise; +∞ at the poles. */
export const logGamma: Unary = unary(U.logGamma, (x) => digamma(x))
/** Γ(x), elementwise; NaN at the poles, +∞ above 171.62. */
export const gamma: Unary = unary(U.gamma, (x, y) => mul(y, digamma(x)))
/** ψ(x) = d/dx log Γ(x), elementwise. */
export const digamma: Unary = unary(U.digamma, (x) => trigamma(x))
/** ψ₁(x) = d²/dx² log Γ(x), elementwise, for all real x except the poles. Its own derivative is first-order only. */
export const trigamma: Unary = unary(U.trigamma)
/**
 * ψ⁽ⁿ⁾(x) for integer n ≥ 1 and x > 0, elementwise over broadcast (n, x). Differentiable in x (∂x = ψ⁽ⁿ⁺¹⁾), not n.
 */
export const polygamma: Binary = binary(B.polygamma, [null, (n, x) => polygamma(add(n, 1), x)])
/** log B(a, b) for a, b > 0, elementwise over broadcast (a, b), accurate when either argument is large. */
export const logBeta: Binary = binary(B.logBeta, [
  (a, b) => sub(digamma(a), digamma(add(a, b))),
  (a, b) => sub(digamma(b), digamma(add(a, b))),
])
/** log n! = log Γ(n + 1), elementwise. */
export const logFactorial: Unary = unary(U.logFactorial, (n) => digamma(add(n, 1)))
/** log (n choose k), elementwise over broadcast (n, k), accurate for large n and small k; −∞ outside 0 ≤ k ≤ n. */
export const logChoose: Binary = binary(B.logChoose, [
  (n, k) => sub(digamma(add(n, 1)), digamma(add(sub(n, k), 1))),
  (n, k) => sub(digamma(add(sub(n, k), 1)), digamma(add(k, 1))),
])
/**
 * The regularised lower incomplete gamma function P(a, x), elementwise over broadcast (a, x), like
 * scipy.special.gammainc. Differentiable in x only.
 */
export const regularisedGammaP: Binary = binary(B.regularisedGammaP)
/** The regularised upper incomplete gamma Q(a, x) = 1 − P(a, x), accurate in the upper tail. Differentiable in x. */
export const regularisedGammaQ: Binary = binary(B.regularisedGammaQ)

// ── Beta family and derived distribution functions (beta.ts) ─────────────────────────────────────────────────────────

/**
 * The regularised incomplete beta function I_x(a, b), elementwise over broadcast (a, b, x), like
 * scipy.special.betainc. Differentiable in x only.
 */
export const regularisedBeta: Ternary = ternary(T.regularisedBeta)
/**
 * The inverse of I_x(a, b) in x, elementwise over broadcast (a, b, p), like scipy.special.betaincinv. Differentiable in
 * p only.
 */
export const regularisedBetaInverse: Ternary = ternary(T.regularisedBetaInverse)
/** The Student t cdf, elementwise over broadcast (t, ν); ν = ∞ gives Φ. Accurate in both tails. Differentiable in t. */
export const studentTCdf: Binary = binary(B.studentTCdf)
/** The Student t quantile, elementwise over broadcast (p, ν). Differentiable in p. */
export const studentTQuantile: Binary = binary(B.studentTQuantile)
/** The chi-square cdf P(k/2, x/2), elementwise over broadcast (x, k). Differentiable in x. */
export const chiSquareCdf: Binary = binary(B.chiSquareCdf)
/**
 * The chi-square survival function Q(k/2, x/2), elementwise over broadcast (x, k); use it for p-values.
 * Differentiable in x.
 */
export const chiSquareSf: Binary = binary(B.chiSquareSf)

// ── Stable elementary forms (stable.ts) ──────────────────────────────────────────────────────────────────────────────

/** softplus(x) = log(1 + eˣ), elementwise, without overflow; also known as log1pexp. */
export const softplus: Unary = unary(U.softplus, (x) => sigmoid(x))
/** log(1 + eˣ); the same primitive as {@link softplus}. */
export const log1pexp: Unary = softplus
/** The logistic sigmoid 1/(1 + e^{−x}), elementwise, stable for either sign. */
export const sigmoid: Unary = unary(U.sigmoid, (_x, y) => mul(y, sub(1, y)))
/** log σ(x) = −softplus(−x), elementwise. */
export const logSigmoid: Unary = unary(U.logSigmoid, (x) => sigmoid(neg(x)))
/** log(p/(1 − p)), elementwise. */
export const logit: Unary = unary(U.logit, (p) => div(1, mul(p, sub(1, p))))
/** log(1 − eˣ) for x ≤ 0, elementwise (Mächler 2012). */
export const log1mexp: Unary = unary(U.log1mexp, (x) => div(-1, expm1(neg(x))))
/** log(eˣ − 1) for x ≥ 0, elementwise. */
export const logExpm1: Unary = unary(U.logExpm1, (x) => div(-1, expm1(neg(x))))
/** log(1 + x) − x, elementwise, accurate near 0. */
export const log1pmx: Unary = unary(U.log1pmx, (x) => neg(div(x, add(1, x))))
/** log(eᵃ + eᵇ), elementwise over broadcast (a, b), handling −∞. */
export const logAddExp: Binary = binary(B.logAddExp, [(a, _b, y) => exp(sub(a, y)), (_a, b, y) => exp(sub(b, y))])
/** log(eᵃ − eᵇ) for a ≥ b, elementwise over broadcast (a, b). */
export const logDiffExp: Binary = binary(B.logDiffExp, [
  (a, _b, y) => exp(sub(a, y)),
  (_a, b, y) => neg(exp(sub(b, y))),
])

/**
 * Deprecated alias of `aifn/tensor`'s `logsumexp` (one name per operation: use `logsumexp`).
 * TODO(consolidation WP2): remove once pgm imports `logsumexp` from aifn/tensor.
 */
export const logSumExp: Reduction = logsumexp

// ── Elementwise products with logarithms and Bessel functions (bessel.ts) ───────────────────────────────────────────

/** x · log y with 0 · log y = 0 (so 0 · log 0 = 0), elementwise over broadcast (x, y); as `scipy.special.xlogy`. */
export const xlogy: Binary = binary(B.xlogy)
/** x · log(1 + y) with 0 · log(1 + y) = 0, elementwise over broadcast (x, y); as `scipy.special.xlog1py`. */
export const xlog1py: Binary = binary(B.xlog1py)
/** The modified Bessel function of the first kind I₀(x), elementwise (even in x). Its derivative is I₁. */
export const besselI0: Unary = unary(U.besselI0, (x) => besselI1(x))
/** The modified Bessel function of the first kind I₁(x), elementwise (odd in x). First-order derivative only. */
export const besselI1: Unary = unary(U.besselI1)
/** A(κ) = I₁(κ)/I₀(κ) for κ ≥ 0 (NaN below), elementwise: the mean resultant length of a von Mises distribution. */
export const besselRatio: Unary = unary(U.besselRatio)
/** log I₀(κ) for κ ≥ 0 (NaN below), elementwise, without overflow; its derivative is `besselRatio`. */
export const logBesselI0: Unary = unary(U.logBesselI0, (x) => besselRatio(x))

/** Options of `softmax` and `logSoftmax`. */
export type SoftmaxOptions = {
  /** The axis normalised over (default −1, the last). */
  axis?: number
  /** Temperature T > 0 (default 1): the result is softmax(x/T). */
  temperature?: number
}

/**
 * log softmax(x/T) along `axis` (default the last): xᵢ/T − log Σⱼ e^{xⱼ/T}, same shape as x (rank ≥ 1). Entries equal
 * to −∞ give −∞; a lane that is all −∞ gives NaN (there is no distribution). A composition of primitives, so it is
 * differentiable to any order.
 */
export function logSoftmax(x: Tensor, options?: SoftmaxOptions): Tensor
export function logSoftmax(x: Traced, options?: SoftmaxOptions): Traced
export function logSoftmax(x: Value, options?: SoftmaxOptions): Value
export function logSoftmax(x: Value, { axis = -1, temperature = 1 }: SoftmaxOptions = {}): Value {
  if (shapeOfValue(x).length === 0) throw new Error('logSoftmax: needs a tensor of rank ≥ 1')
  const z = temperature === 1 ? x : div(x, temperature)
  return sub(z, logsumexp(z, axis, true))
}

/**
 * softmax(x/T) along `axis` (default the last): e^{xᵢ/T} / Σⱼ e^{xⱼ/T}, same shape as x (rank ≥ 1), each lane summing
 * to 1; stable (computed as exp of `logSoftmax`). Entries equal to −∞ get probability 0; a lane that is all −∞ gives
 * NaN.
 */
export function softmax(x: Tensor, options?: SoftmaxOptions): Tensor
export function softmax(x: Traced, options?: SoftmaxOptions): Traced
export function softmax(x: Value, options?: SoftmaxOptions): Value
export function softmax(x: Value, options: SoftmaxOptions = {}): Value {
  return exp(logSoftmax(x, options))
}

const binaryEntropyNats: Unary = unary(U.binaryEntropy, (p) => neg(logit(p)))

/**
 * Binary entropy H(p) = −p log p − (1 − p) log(1 − p) elementwise, with 0 log 0 = 0 and NaN outside [0, 1], in nats,
 * or in the given `base` (2 for bits).
 */
export function binaryEntropy(p: number, base?: number): number
export function binaryEntropy(p: Tensor, base?: number): Tensor
export function binaryEntropy(p: Traced, base?: number): Traced
export function binaryEntropy(p: Value, base?: number): Value
export function binaryEntropy(p: Value, base = Math.E): Value {
  const h = binaryEntropyNats(p)
  return base === Math.E ? h : div(h, Math.log(base))
}
