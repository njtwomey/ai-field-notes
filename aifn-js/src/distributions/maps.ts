/**
 * Maps of the real line for pushing distributions forward: intervals with open and closed ends, monotone bijectors
 * (each with its domain, codomain, inverse and log |f′|), and many-to-one maps given by monotone branches (y = x²).
 * `imageOf` computes the image of an interval under a map, which is how `Transformed` and `Pushforward` find their
 * supports and report a base whose support does not fit the map's domain.
 */

import { logSigmoid, logit, normalCdf, normalLogPdf, normalQuantile, sigmoid, softplus, logExpm1 } from 'aifn/special'
import {
  abs,
  add,
  div,
  exp,
  isTensor,
  log,
  log1p,
  mul,
  neg,
  pow,
  sqrt,
  square,
  sub,
  tanh,
  toFlat,
  unwrap,
  type Value,
} from 'aifn/tensor'
import type { Support } from './types'

// ── Intervals ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** An interval of the real line. Infinite ends are always open. */
export type Interval = { lower: number; upper: number; lowerOpen: boolean; upperOpen: boolean }

/**
 * The interval from `lower` to `upper` with the ends given as brackets: '[]' closed, '()' open, '[)' or '(]' mixed.
 * Infinite ends are open whatever the brackets say.
 */
export function interval(lower: number, upper: number, ends: '[]' | '()' | '[)' | '(]' = '[]'): Interval {
  return {
    lower,
    upper,
    lowerOpen: ends[0] === '(' || !Number.isFinite(lower),
    upperOpen: ends[1] === ')' || !Number.isFinite(upper),
  }
}

/** ℝ = (−∞, ∞). */
export const REALS: Interval = interval(-Infinity, Infinity)
/** (0, ∞). */
export const POSITIVE: Interval = interval(0, Infinity, '()')
/** (0, 1). */
export const UNIT_INTERVAL: Interval = interval(0, 1, '()')

/** The smallest and largest raw values of a bound (a batch of bounds gives the outermost). */
function extreme(v: Value, which: 'min' | 'max'): number {
  const r = unwrap(v)
  const values = typeof r === 'number' ? [r] : isTensor(r) ? toFlat(r) : [NaN]
  return which === 'min' ? Math.min(...values) : Math.max(...values)
}

/**
 * A univariate support as an interval: `real` is ℝ; `interval`, `integers` and `circle` take their bounds (the
 * outermost of a batch), closed at finite ends unless the support marks them open. Other supports throw.
 */
export function supportInterval(support: Support): Interval {
  if (support.type === 'real') return REALS
  if (support.type === 'interval' || support.type === 'integers' || support.type === 'circle') {
    const lower = extreme(support.lower, 'min')
    const upper = extreme(support.upper, 'max')
    const open = support.type === 'interval' ? support : { lowerOpen: false, upperOpen: false }
    return {
      lower,
      upper,
      lowerOpen: !!open.lowerOpen || !Number.isFinite(lower),
      upperOpen: !!open.upperOpen || !Number.isFinite(upper),
    }
  }
  throw new RangeError(`supportInterval: a ${support.type} support is not an interval of the real line`)
}

/** An interval as a `Support`: ℝ as `real`, anything else as `interval` with its open ends marked. */
export function intervalSupport(i: Interval): Support {
  if (i.lower === -Infinity && i.upper === Infinity) return { type: 'real' }
  return { type: 'interval', lower: i.lower, upper: i.upper, lowerOpen: i.lowerOpen, upperOpen: i.upperOpen }
}

// Four significant digits; negative ends take the minus sign (U+2212), as in typeset maths.
const formatEnd = (v: number) =>
  v === Infinity ? '∞' : v === -Infinity ? '−∞' : String(Number(v.toPrecision(4))).replace('-', '−')

/** Plain text for an interval, ends to four significant digits: 'ℝ', '(0, ∞)', '[0.3679, 2.718]'. */
export function formatInterval(i: Interval): string {
  if (i.lower === -Infinity && i.upper === Infinity) return 'ℝ'
  return `${i.lowerOpen ? '(' : '['}${formatEnd(i.lower)}, ${formatEnd(i.upper)}${i.upperOpen ? ')' : ']'}`
}

/** A point strictly inside an interval (the midpoint of a bounded one), e.g. to stand in for values outside it. */
export function interiorPoint(i: Interval): number {
  const { lower: a, upper: b } = i
  if (Number.isFinite(a) && Number.isFinite(b)) return (a + b) / 2
  if (Number.isFinite(a)) return a + 1
  if (Number.isFinite(b)) return b - 1
  return 0
}

/**
 * Whether `inner` lies inside `outer`. With `endpoints: 'ignore'` (the default, for continuous distributions) a closed
 * end of `inner` may sit on an open end of `outer`: a single point has probability zero, so [0, ∞) counts as inside
 * (0, ∞). With `endpoints: 'strict'` (for mass functions) it does not.
 */
export function intervalInside(
  inner: Interval,
  outer: Interval,
  { endpoints = 'ignore' }: { endpoints?: 'ignore' | 'strict' } = {},
): boolean {
  if (inner.lower < outer.lower || inner.upper > outer.upper) return false
  if (endpoints === 'ignore') return true
  const lowerOk = inner.lower > outer.lower || !outer.lowerOpen || inner.lowerOpen
  const upperOk = inner.upper < outer.upper || !outer.upperOpen || inner.upperOpen
  return lowerOk && upperOk
}

/** The intersection of two intervals; `null` when it is empty. */
function intersect(a: Interval, b: Interval): Interval | null {
  const lower = Math.max(a.lower, b.lower)
  const upper = Math.min(a.upper, b.upper)
  const lowerOpen = (a.lower === lower && a.lowerOpen) || (b.lower === lower && b.lowerOpen)
  const upperOpen = (a.upper === upper && a.upperOpen) || (b.upper === upper && b.upperOpen)
  if (lower > upper || (lower === upper && (lowerOpen || upperOpen))) return null
  return { lower, upper, lowerOpen, upperOpen }
}

/** The smallest interval holding every interval (their union, when they overlap). An end is open only if open in all. */
function hull(parts: readonly Interval[]): Interval {
  const lower = Math.min(...parts.map((p) => p.lower))
  const upper = Math.max(...parts.map((p) => p.upper))
  return {
    lower,
    upper,
    lowerOpen: parts.every((p) => p.lower !== lower || p.lowerOpen),
    upperOpen: parts.every((p) => p.upper !== upper || p.upperOpen),
  }
}

// ── Bijectors ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A monotone bijection y = f(x) from `domain` onto `codomain`, with its inverse and log |f′(x)|. All three are
 * compositions of primitives, so a transformed log-density stays differentiable.
 */
export type Bijector = {
  name: string
  forward(x: Value): Value
  inverse(y: Value): Value
  /** log |dy/dx| at x. */
  logAbsDetJacobian(x: Value): Value
  /** True when f is increasing (then cdfs map directly; otherwise the survival function is used). */
  increasing: boolean
  /** Where f is defined. */
  domain: Interval
  /** f(domain). */
  codomain: Interval
}

/** y = eˣ, from ℝ to (0, ∞). Pushes a normal forward to a log-normal. */
export const expBijector: Bijector = {
  name: 'exp',
  forward: exp,
  inverse: log,
  logAbsDetJacobian: (x) => x,
  increasing: true,
  domain: REALS,
  codomain: POSITIVE,
}

/** y = log x, from (0, ∞) to ℝ; log |dy/dx| = −log x. Pushes a log-normal back to a normal. */
export const logBijector: Bijector = {
  name: 'log',
  forward: log,
  inverse: exp,
  logAbsDetJacobian: (x) => neg(log(x)),
  increasing: true,
  domain: POSITIVE,
  codomain: REALS,
}

/** y = σ(x) = 1/(1 + e^{−x}), from ℝ to (0, 1) (the logit-normal and the like). */
export const sigmoidBijector: Bijector = {
  name: 'sigmoid',
  forward: sigmoid,
  inverse: logit,
  logAbsDetJacobian: (x) => add(logSigmoid(x), logSigmoid(neg(x))),
  increasing: true,
  domain: REALS,
  codomain: UNIT_INTERVAL,
}

/**
 * y = tanh x, from ℝ to (−1, 1). The inverse is artanh y = ½(log1p y − log1p(−y)); log(1 − tanh² x) is computed as
 * 2(log 2 − x − softplus(−2x)), which stays finite in the tails where 1 − tanh² x underflows.
 */
export const tanhBijector: Bijector = {
  name: 'tanh',
  forward: tanh,
  inverse: (y) => mul(0.5, sub(log1p(y), log1p(neg(y)))),
  logAbsDetJacobian: (x) => mul(2, sub(sub(Math.LN2, x), softplus(mul(-2, x)))),
  increasing: true,
  domain: REALS,
  codomain: interval(-1, 1, '()'),
}

/** y = softplus x = log(1 + eˣ), from ℝ to (0, ∞); the inverse is log(eʸ − 1) and dy/dx = σ(x). */
export const softplusBijector: Bijector = {
  name: 'softplus',
  forward: softplus,
  inverse: logExpm1,
  logAbsDetJacobian: logSigmoid,
  increasing: true,
  domain: REALS,
  codomain: POSITIVE,
}

/**
 * y = Φ(x), the standard normal cdf, from ℝ to (0, 1); the inverse is the probit Φ⁻¹ and dy/dx = φ(x). Pushes N(0, 1)
 * forward to Uniform(0, 1) (the probability integral transform).
 */
export const normalCdfBijector: Bijector = {
  name: 'Φ',
  forward: normalCdf,
  inverse: normalQuantile,
  logAbsDetJacobian: normalLogPdf,
  increasing: true,
  domain: REALS,
  codomain: UNIT_INTERVAL,
}

/** y = loc + scale · x, for a non-zero scale (a number, so that the direction is known). */
export function affineBijector(loc: Value, scale: number): Bijector {
  if (!(scale !== 0 && Number.isFinite(scale)))
    throw new RangeError('affineBijector: scale must be finite and non-zero')
  return {
    name: 'affine',
    forward: (x) => add(loc, mul(scale, x)),
    inverse: (y) => div(sub(y, loc), scale),
    logAbsDetJacobian: (x) => add(mul(0, x), Math.log(Math.abs(scale))),
    increasing: scale > 0,
    domain: REALS,
    codomain: REALS,
  }
}

/**
 * y = xᵖ on (0, ∞), for a finite non-zero power p (increasing for p > 0, decreasing for p < 0); log |dy/dx| =
 * log |p| + (p − 1) log x.
 */
export function powerBijector(p: number): Bijector {
  if (!(p !== 0 && Number.isFinite(p))) throw new RangeError('powerBijector: the power must be finite and non-zero')
  return {
    name: `power ${p}`,
    forward: (x) => pow(x, p),
    inverse: (y) => pow(y, 1 / p),
    logAbsDetJacobian: (x) => add(Math.log(Math.abs(p)), mul(p - 1, log(x))),
    increasing: p > 0,
    domain: POSITIVE,
    codomain: POSITIVE,
  }
}

/**
 * The composition that applies `bijectors` in order, first to last: y = fₙ(… f₁(x)). The log-Jacobians add along the
 * chain; the domain is the first's, the codomain the image of that domain through the chain. Throws if one bijector's
 * codomain does not fit the next one's domain.
 *
 * @example chainBijectors(affineBijector(0, 1 / T), sigmoidBijector) // y = σ(x / T), a sigmoid with temperature T
 */
export function chainBijectors(...bijectors: Bijector[]): Bijector {
  if (bijectors.length === 0) throw new RangeError('chainBijectors: needs at least one bijector')
  let codomain = bijectors[0].codomain
  for (const b of bijectors.slice(1)) codomain = imageOf(b, codomain, { where: 'chainBijectors' })
  return {
    name: bijectors.map((b) => b.name).join(' then '),
    forward: (x) => bijectors.reduce((v, b) => b.forward(v), x),
    inverse: (y) => bijectors.reduceRight((v, b) => b.inverse(v), y),
    logAbsDetJacobian: (x) => {
      let total: Value = 0
      let v = x
      for (const b of bijectors) {
        total = add(total, b.logAbsDetJacobian(v))
        v = b.forward(v)
      }
      return total
    },
    increasing: bijectors.filter((b) => !b.increasing).length % 2 === 0,
    domain: bijectors[0].domain,
    codomain,
  }
}

// ── Many-to-one maps ─────────────────────────────────────────────────────────────────────────────────────────────────

/** One monotone piece of a many-to-one map: on `domain` the map is a bijection with this inverse and log |f′|. */
export type Branch = {
  domain: Interval
  inverse(y: Value): Value
  logAbsDetJacobian(x: Value): Value
  increasing: boolean
}

/**
 * A map y = f(x) whose domain is covered by monotone branches (overlapping at most at their ends). A value y then has
 * one preimage per branch whose image holds it, and the pushforward density sums over them.
 */
export type ManyToOneMap = {
  name: string
  forward(x: Value): Value
  branches: readonly Branch[]
  domain: Interval
  codomain: Interval
}

/**
 * y = x², from ℝ onto [0, ∞): two branches, x = −√y on (−∞, 0] and x = √y on [0, ∞), each with log |dy/dx| = log |2x|.
 * A standard normal pushed forward is the χ²₁ distribution.
 */
export const squareMap: ManyToOneMap = {
  name: 'square',
  forward: square,
  branches: [
    {
      domain: interval(-Infinity, 0, '(]'),
      inverse: (y) => neg(sqrt(y)),
      logAbsDetJacobian: (x) => log(abs(mul(2, x))),
      increasing: false,
    },
    {
      domain: interval(0, Infinity, '[)'),
      inverse: sqrt,
      logAbsDetJacobian: (x) => log(abs(mul(2, x))),
      increasing: true,
    },
  ],
  domain: REALS,
  codomain: interval(0, Infinity, '[)'),
}

/** A bijector as a many-to-one map with one branch. */
export function asManyToOne(b: Bijector | ManyToOneMap): ManyToOneMap {
  if ('branches' in b) return b
  return { name: b.name, forward: b.forward, branches: [{ ...b }], domain: b.domain, codomain: b.codomain }
}

/**
 * The image of a subinterval `x` of a monotone branch's domain. An end of the image is open when the end of `x` is
 * open or infinite, when the image end is infinite, or when the end of `x` sits on an open end of the domain (log at 0,
 * say): those values are approached but not attained.
 */
function monotoneImage(forward: (x: Value) => Value, increasing: boolean, x: Interval, domain: Interval): Interval {
  const at = (v: number) => forward(v)
  const lowOpen = (a: number) =>
    x.lowerOpen || !Number.isFinite(x.lower) || !Number.isFinite(a) || (x.lower === domain.lower && domain.lowerOpen)
  const highOpen = (b: number) =>
    x.upperOpen || !Number.isFinite(x.upper) || !Number.isFinite(b) || (x.upper === domain.upper && domain.upperOpen)
  const fl = at(x.lower)
  const fu = at(x.upper)
  if (increasing) {
    const a = extreme(fl, 'min')
    const b = extreme(fu, 'max')
    return { lower: a, upper: b, lowerOpen: lowOpen(a), upperOpen: highOpen(b) }
  }
  const a = extreme(fu, 'min')
  const b = extreme(fl, 'max')
  return { lower: a, upper: b, lowerOpen: highOpen(a), upperOpen: lowOpen(b) }
}

/**
 * The image f(x) of an interval under a bijector or a many-to-one map (the hull of its branches' images). Throws a
 * RangeError naming `where` when `x` is not inside the map's domain (see `intervalInside` for `endpoints`).
 *
 * @example imageOf(sigmoidBijector, REALS) // (0, 1)
 * @example imageOf(affineBijector(0, -2), interval(0, 1)) // [−2, 0]: a negative scale flips the ends
 * @example imageOf(squareMap, interval(0, Infinity, '[)')) // [0, ∞)
 */
export function imageOf(
  map: Bijector | ManyToOneMap,
  x: Interval,
  { where = 'imageOf', endpoints = 'ignore' }: { where?: string; endpoints?: 'ignore' | 'strict' } = {},
): Interval {
  if (!intervalInside(x, map.domain, { endpoints }))
    throw new RangeError(
      `${where}: ${formatInterval(x)} is not inside the domain ${formatInterval(map.domain)} of ${map.name}`,
    )
  return hull(branchImages(map, x).map((p) => p.image))
}

/**
 * For each branch of a map (one for a bijector) that meets the interval `x`: the piece of `x` in the branch's domain
 * and that piece's image. Branches that miss `x` are left out. Does not check that `x` is inside the domain.
 */
export function branchImages(
  map: Bijector | ManyToOneMap,
  x: Interval,
): { branch: Branch; piece: Interval; image: Interval }[] {
  const m = asManyToOne(map)
  return m.branches.flatMap((branch) => {
    const piece = intersect(x, branch.domain)
    return piece ? [{ branch, piece, image: monotoneImage(m.forward, branch.increasing, piece, branch.domain) }] : []
  })
}
