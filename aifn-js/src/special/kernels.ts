/**
 * The kernel table of `aifn/special`: every function as a scalar forward rule with its derivative rules, ready to be
 * lifted by `defineUnary(name, f, df)`, `defineBinary(name, f, dfa, dfb)` and `defineOp` from `aifn/tensor` (see the
 * README, "One definition per operation"). The table is internal: `index.ts` exports only the lifted primitives.
 *
 * Conventions. A derivative rule receives the inputs and the output y = f(inputs), so rules can reuse y (e.g.
 * σ'(x) = y(1 − y)). `null` marks a partial derivative that is not implemented; differentiating through it must be an
 * error. Arguments that are discrete or act as fixed parameters (an order n, a temperature) have `null` partials.
 *
 * Partial derivatives available:
 * - all unary functions have df;
 * - logAddExp, logDiffExp, logBeta, logChoose, normalLogIntervalProbability: both arguments;
 * - regularisedGammaP/Q, chiSquareCdf/Sf, studentTCdf, studentTQuantile, polygamma: only the continuous variable
 *   (x, t or p), not the shape, order or degrees of freedom;
 * - regularisedBeta and regularisedBetaInverse: only x (resp. p), not a or b;
 * - xlogy, xlog1py: both arguments;
 * - truncatedNormalVDraw: t only (dv/dt = −w, as for the one-sided v); truncatedNormalWDraw: none.
 *
 * logSumExp, softmax and logSoftmax are not kernels: they are compositions of `aifn/tensor` primitives (index.ts).
 */

import {
  betaDensity,
  chiSquareCdf,
  chiSquareDensity,
  chiSquareSf,
  regularisedBeta,
  regularisedBetaInverse,
} from './beta'
import { studentTCdf, studentTDensity, studentTQuantile } from './beta'
import { erf, erfc, erfcx, logErfc } from './erf'
import { digamma, gamma, gammaDensity, logBeta, logChoose, logFactorial, logGamma, polygamma } from './gamma'
import { regularisedGammaP, regularisedGammaQ, sinPi, trigamma } from './gamma'
import { erfcinv, erfinv, normalCdf, normalLogCdf, normalLogIntervalProbability, normalLogPdf } from './normal'
import { normalPdf, normalQuantile, truncatedNormalV, truncatedNormalVDraw, truncatedNormalW } from './normal'
import { truncatedNormalWDraw } from './normal'
import { binaryEntropy, log1mexp, log1pmx, logAddExp, logDiffExp, logExpm1, logit, logSigmoid } from './stable'
import { sigmoid, softplus } from './stable'
import { besselI0, besselI1, besselI1Derivative, besselRatio, besselRatioDerivative, logBesselI0 } from './bessel'

/** A one-argument elementwise kernel: y = f(x), with dy/dx = df(x, y). */
export interface UnaryKernel {
  readonly name: string
  readonly f: (x: number) => number
  readonly df: ((x: number, y: number) => number) | null
}

/** A two-argument elementwise kernel: y = f(a, b), with ∂y/∂a = dfa(a, b, y) and ∂y/∂b = dfb(a, b, y). */
export interface BinaryKernel {
  readonly name: string
  readonly f: (a: number, b: number) => number
  readonly dfa: ((a: number, b: number, y: number) => number) | null
  readonly dfb: ((a: number, b: number, y: number) => number) | null
}

/** A three-argument elementwise kernel: y = f(a, b, c), with one partial rule per argument. */
export interface TernaryKernel {
  readonly name: string
  readonly f: (a: number, b: number, c: number) => number
  readonly partials: readonly [
    ((a: number, b: number, c: number, y: number) => number) | null,
    ((a: number, b: number, c: number, y: number) => number) | null,
    ((a: number, b: number, c: number, y: number) => number) | null,
  ]
}

const TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI)
const HALF_SQRT_PI = Math.sqrt(Math.PI) / 2
const SQRT_2PI = Math.sqrt(2 * Math.PI)

const unary = (name: string, f: UnaryKernel['f'], df: UnaryKernel['df']): UnaryKernel => ({ name, f, df })
const binary = (
  name: string,
  f: BinaryKernel['f'],
  dfa: BinaryKernel['dfa'],
  dfb: BinaryKernel['dfb'],
): BinaryKernel => ({ name, f, dfa, dfb })

/** d/dx ψ₁(x) = ψ₂(x), with the reflection ψ₁(x) = π²/sin²(πx) − ψ₁(1 − x) differentiated for x < 0. */
function trigammaDerivative(x: number): number {
  if (x > 0) return polygamma(2, x)
  const s = sinPi(x)
  const c = Math.cos(Math.PI * (x % 2))
  return (-2 * Math.PI ** 3 * c) / (s * s * s) + polygamma(2, 1 - x)
}

/** The unary kernels. */
export const unaryKernels = {
  erf: unary('erf', erf, (x) => TWO_OVER_SQRT_PI * Math.exp(-x * x)),
  erfc: unary('erfc', erfc, (x) => -TWO_OVER_SQRT_PI * Math.exp(-x * x)),
  erfcx: unary('erfcx', erfcx, (x, y) => 2 * x * y - TWO_OVER_SQRT_PI),
  logErfc: unary('logErfc', logErfc, (x) => -TWO_OVER_SQRT_PI / erfcx(x)),
  erfinv: unary('erfinv', erfinv, (_x, y) => HALF_SQRT_PI * Math.exp(y * y)),
  erfcinv: unary('erfcinv', erfcinv, (_x, y) => -HALF_SQRT_PI * Math.exp(y * y)),
  normalPdf: unary('normalPdf', normalPdf, (x, y) => -x * y),
  normalLogPdf: unary('normalLogPdf', normalLogPdf, (x) => -x),
  normalCdf: unary('normalCdf', normalCdf, (x) => normalPdf(x)),
  normalLogCdf: unary('normalLogCdf', normalLogCdf, (x) => truncatedNormalV(x)),
  normalQuantile: unary('normalQuantile', normalQuantile, (_p, y) => SQRT_2PI * Math.exp(0.5 * y * y)),
  truncatedNormalV: unary('truncatedNormalV', truncatedNormalV, (t) => -truncatedNormalW(t)),
  // w = v(v + t) and v' = −w give w' = v'(2v + t) + v = v − w(2v + t).
  truncatedNormalW: unary('truncatedNormalW', truncatedNormalW, (t, w) => {
    const v = truncatedNormalV(t)
    return v - w * (2 * v + t)
  }),
  logGamma: unary('logGamma', logGamma, (x) => digamma(x)),
  gamma: unary('gamma', gamma, (x, y) => y * digamma(x)),
  digamma: unary('digamma', digamma, (x) => trigamma(x)),
  trigamma: unary('trigamma', trigamma, trigammaDerivative),
  logFactorial: unary('logFactorial', logFactorial, (n) => digamma(n + 1)),
  softplus: unary('softplus', softplus, (x) => sigmoid(x)),
  sigmoid: unary('sigmoid', sigmoid, (_x, y) => y * (1 - y)),
  logSigmoid: unary('logSigmoid', logSigmoid, (x) => sigmoid(-x)),
  logit: unary('logit', logit, (p) => 1 / (p * (1 - p))),
  // Both −eˣ/(1 − eˣ) and eˣ/(eˣ − 1) equal −1/expm1(−x).
  log1mexp: unary('log1mexp', log1mexp, (x) => -1 / Math.expm1(-x)),
  logExpm1: unary('logExpm1', logExpm1, (x) => -1 / Math.expm1(-x)),
  log1pmx: unary('log1pmx', log1pmx, (x) => -x / (1 + x)),
  binaryEntropy: unary(
    'binaryEntropy',
    (p) => binaryEntropy(p),
    (p) => -logit(p),
  ),
  besselI0: unary('besselI0', besselI0, (x) => besselI1(x)),
  besselI1: unary('besselI1', besselI1, (x) => besselI1Derivative(x)),
  logBesselI0: unary('logBesselI0', logBesselI0, (x) => besselRatio(x)),
  besselRatio: unary('besselRatio', besselRatio, besselRatioDerivative),
} as const satisfies Record<string, UnaryKernel>

/** The binary kernels. */
export const binaryKernels = {
  // x log y with 0 · log y = 0 (so 0 · log 0 = 0), as scipy.special.xlogy; NaN y still gives NaN.
  xlogy: binary(
    'xlogy',
    (x, y) => (x === 0 && y === y ? 0 : x * Math.log(y)),
    (_x, y) => Math.log(y),
    (x, y) => (x === 0 ? 0 : x / y),
  ),
  xlog1py: binary(
    'xlog1py',
    (x, y) => (x === 0 && y === y ? 0 : x * Math.log1p(y)),
    (_x, y) => Math.log1p(y),
    (x, y) => (x === 0 ? 0 : x / (1 + y)),
  ),
  logAddExp: binary(
    'logAddExp',
    logAddExp,
    (a, _b, y) => Math.exp(a - y),
    (_a, b, y) => Math.exp(b - y),
  ),
  logDiffExp: binary(
    'logDiffExp',
    logDiffExp,
    (a, _b, y) => Math.exp(a - y),
    (_a, b, y) => -Math.exp(b - y),
  ),
  logBeta: binary(
    'logBeta',
    logBeta,
    (a, b) => digamma(a) - digamma(a + b),
    (a, b) => digamma(b) - digamma(a + b),
  ),
  logChoose: binary(
    'logChoose',
    logChoose,
    (n, k) => digamma(n + 1) - digamma(n - k + 1),
    (n, k) => digamma(n - k + 1) - digamma(k + 1),
  ),
  polygamma: binary('polygamma', polygamma, null, (n, x) => polygamma(n + 1, x)),
  regularisedGammaP: binary('regularisedGammaP', regularisedGammaP, null, (a, x) => gammaDensity(a, x)),
  regularisedGammaQ: binary('regularisedGammaQ', regularisedGammaQ, null, (a, x) => -gammaDensity(a, x)),
  chiSquareCdf: binary('chiSquareCdf', chiSquareCdf, (x, k) => chiSquareDensity(x, k), null),
  chiSquareSf: binary('chiSquareSf', chiSquareSf, (x, k) => -chiSquareDensity(x, k), null),
  studentTCdf: binary('studentTCdf', studentTCdf, (t, df) => studentTDensity(t, df), null),
  studentTQuantile: binary('studentTQuantile', studentTQuantile, (_p, df, y) => 1 / studentTDensity(y, df), null),
  normalLogIntervalProbability: binary(
    'normalLogIntervalProbability',
    normalLogIntervalProbability,
    (l, _u, y) => -Math.exp(normalLogPdf(l) - y),
    (_l, u, y) => Math.exp(normalLogPdf(u) - y),
  ),
  truncatedNormalVDraw: binary(
    'truncatedNormalVDraw',
    truncatedNormalVDraw,
    (t, eps) => -truncatedNormalWDraw(t, eps),
    null,
  ),
  truncatedNormalWDraw: binary('truncatedNormalWDraw', truncatedNormalWDraw, null, null),
} as const satisfies Record<string, BinaryKernel>

/** The ternary kernels. */
export const ternaryKernels: Record<'regularisedBeta' | 'regularisedBetaInverse', TernaryKernel> = {
  regularisedBeta: {
    name: 'regularisedBeta',
    f: regularisedBeta,
    partials: [null, null, (a, b, x) => betaDensity(a, b, x)],
  },
  // dx/dp = 1 / (∂I/∂x at x = y).
  regularisedBetaInverse: {
    name: 'regularisedBetaInverse',
    f: regularisedBetaInverse,
    partials: [null, null, (a, b, _p, y) => 1 / betaDensity(a, b, y)],
  },
}
