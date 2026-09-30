/**
 * Adaptive quadrature with error estimates: adaptive Simpson (local, depth-first) and globally adaptive Gauss–Kronrod
 * 7–15 (bisect the interval with the largest error estimate, as QUADPACK's QAG), both traceable, plus `integrate`,
 * which handles infinite limits by a change of variables.
 */

import { run, type Algorithm } from 'aifn/trace'
import type { Integrand } from './rules'

/** One piece of the subdivision: the interval, its estimate and its error estimate. */
export type Interval = { a: number; b: number; value: number; error: number }

// ---------------------------------------------------------------------------------------------------------------------
// Adaptive Simpson.

type Pending = {
  a: number
  b: number
  fa: number
  fm: number
  fb: number
  whole: number
  tolerance: number
  depth: number
}

/** The state of `adaptiveSimpson`. */
export type AdaptiveSimpsonState = {
  t: number
  /** Intervals accepted so far, in the order they were accepted. */
  accepted: Interval[]
  /** Intervals still to examine (a stack; the last is examined next). */
  pending: Pending[]
  /** The sum of the accepted estimates (the integral once `pending` is empty). */
  value: number
  /** The sum of the accepted error estimates. */
  error: number
  /** The interval examined on the last step and whether it was split. */
  examined: { a: number; b: number; split: boolean } | null
  evaluations: number
  /** True once every interval met its tolerance (none was accepted only because of the depth limit). */
  converged: boolean
  /** Intervals accepted at the depth limit without meeting their tolerance. */
  unresolved: number
}

/**
 * Adaptive Simpson quadrature (Kuncir, 1962; Lyness, 1969): on each interval compare Simpson's rule S with the sum of
 * Simpson's rule on its halves S₂; if |S₂ − S| ≤ 15·tol accept S₂ + (S₂ − S)/15 (Richardson) with error estimate
 * |S₂ − S|/15, else split with tol halved. Each step examines one interval (depth first). `init` takes `{ a, b }`;
 * options `tolerance` (default 1e-10) and `maxDepth` (default 50).
 */
export function adaptiveSimpson(
  f: Integrand,
  { tolerance = 1e-10, maxDepth = 50 }: { tolerance?: number; maxDepth?: number } = {},
): Algorithm<{ a: number; b: number }, AdaptiveSimpsonState> {
  const simpson = (a: number, fa: number, fm: number, b: number, fb: number) => ((b - a) / 6) * (fa + 4 * fm + fb)
  return {
    name: 'adaptive-simpson',
    init: ({ a, b }) => {
      const fa = f(a)
      const fb = f(b)
      const fm = f((a + b) / 2)
      return {
        t: 0,
        accepted: [],
        pending: [{ a, b, fa, fm, fb, whole: simpson(a, fa, fm, b, fb), tolerance, depth: 0 }],
        value: 0,
        error: 0,
        examined: null,
        evaluations: 3,
        converged: false,
        unresolved: 0,
      }
    },
    step: (s) => {
      const pending = s.pending.slice(0, -1)
      const p = s.pending[s.pending.length - 1]
      const m = (p.a + p.b) / 2
      const flm = f((p.a + m) / 2)
      const frm = f((m + p.b) / 2)
      const left = simpson(p.a, p.fa, flm, m, p.fm)
      const right = simpson(m, p.fm, frm, p.b, p.fb)
      const delta = left + right - p.whole
      const ok = Math.abs(delta) <= 15 * p.tolerance
      const atLimit = p.depth >= maxDepth
      let { accepted, value, error, unresolved } = s
      if (ok || atLimit || !Number.isFinite(delta)) {
        const estimate = left + right + delta / 15
        accepted = [...accepted, { a: p.a, b: p.b, value: estimate, error: Math.abs(delta) / 15 }]
        value += estimate
        error += Math.abs(delta) / 15
        if (!ok) unresolved++
      } else {
        const tol = p.tolerance / 2
        // Push the right half first so the left half is examined next.
        pending.push({ a: m, b: p.b, fa: p.fm, fm: frm, fb: p.fb, whole: right, tolerance: tol, depth: p.depth + 1 })
        pending.push({ a: p.a, b: m, fa: p.fa, fm: flm, fb: p.fm, whole: left, tolerance: tol, depth: p.depth + 1 })
      }
      return {
        t: s.t + 1,
        accepted,
        pending,
        value,
        error,
        examined: { a: p.a, b: p.b, split: !(ok || atLimit || !Number.isFinite(delta)) },
        evaluations: s.evaluations + 2,
        converged: pending.length === 0 && unresolved === 0,
        unresolved,
      }
    },
    done: (s) => s.pending.length === 0,
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Gauss–Kronrod 7–15.

// QUADPACK qk15 (Piessens et al., 1983): Kronrod abscissae (the odd entries are the 7-point Gauss abscissae) and weights.
const XGK = [
  0.9914553711208126, 0.9491079123427585, 0.8648644233597691, 0.7415311855993945, 0.5860872354676911,
  0.4058451513773972, 0.20778495500789848, 0,
]
const WGK = [
  0.022935322010529224, 0.06309209262997856, 0.10479001032225019, 0.14065325971552592, 0.1690047266392679,
  0.19035057806478542, 0.20443294007529889, 0.20948214108472782,
]
const WG = [0.1294849661688697, 0.27970539148927664, 0.3818300505051189, 0.4179591836734694]
const EPMACH = 2 ** -52
const UFLOW = 2 ** -1022

/**
 * The 15-point Kronrod estimate on [a, b] and QUADPACK's error estimate from its difference with the embedded 7-point
 * Gauss estimate, scaled as in `qk15` (the (200·|K − G|/resasc)^1.5 heuristic).
 */
export function kronrod15(f: Integrand, a: number, b: number): Interval {
  const centre = 0.5 * (a + b)
  const half = 0.5 * (b - a)
  const absHalf = Math.abs(half)
  const fc = f(centre)
  let resg = fc * WG[3]
  let resk = fc * WGK[7]
  let resabs = Math.abs(resk)
  const fv1 = new Array<number>(7)
  const fv2 = new Array<number>(7)
  for (let j = 0; j < 7; j++) {
    const dx = half * XGK[j]
    const f1 = f(centre - dx)
    const f2 = f(centre + dx)
    fv1[j] = f1
    fv2[j] = f2
    resk += WGK[j] * (f1 + f2)
    resabs += WGK[j] * (Math.abs(f1) + Math.abs(f2))
    if (j % 2 === 1) resg += WG[(j - 1) / 2] * (f1 + f2)
  }
  const mean = resk * 0.5
  let resasc = WGK[7] * Math.abs(fc - mean)
  for (let j = 0; j < 7; j++) resasc += WGK[j] * (Math.abs(fv1[j] - mean) + Math.abs(fv2[j] - mean))
  const value = resk * half
  resabs *= absHalf
  resasc *= absHalf
  let error = Math.abs((resk - resg) * half)
  if (resasc !== 0 && error !== 0) error = resasc * Math.min(1, ((200 * error) / resasc) ** 1.5)
  if (resabs > UFLOW / (50 * EPMACH)) error = Math.max(EPMACH * 50 * resabs, error)
  return { a, b, value, error }
}

/** The state of `gaussKronrod`. */
export type GaussKronrodState = {
  t: number
  /** The current subdivision, in order of a. */
  intervals: Interval[]
  /** The sum of the interval estimates, and of their error estimates. */
  value: number
  error: number
  /** The interval bisected on the last step (null at t = 0). */
  split: { a: number; b: number } | null
  evaluations: number
  converged: boolean
}

/**
 * Globally adaptive Gauss–Kronrod 7–15 quadrature (QUADPACK's QAG with key 1; Piessens et al., 1983): start with
 * [a, b] and, each step, bisect the interval with the largest error estimate, until the total error is at most
 * max(atol, rtol·|value|) (defaults 1.49e-8, as scipy's `quad`). `init` takes finite `{ a, b }`; use `integrate`
 * for infinite limits.
 */
export function gaussKronrod(
  f: Integrand,
  { atol = 1.49e-8, rtol = 1.49e-8 }: { atol?: number; rtol?: number } = {},
): Algorithm<{ a: number; b: number }, GaussKronrodState> {
  const summarise = (intervals: Interval[]) => {
    let value = 0
    let error = 0
    for (const i of intervals) {
      value += i.value
      error += i.error
    }
    return { value, error, converged: error <= Math.max(atol, rtol * Math.abs(value)) }
  }
  return {
    name: 'gauss-kronrod-15',
    init: ({ a, b }) => {
      const intervals = [kronrod15(f, a, b)]
      return { t: 0, intervals, ...summarise(intervals), split: null, evaluations: 15 }
    },
    step: (s) => {
      let worst = 0
      for (let i = 1; i < s.intervals.length; i++) if (s.intervals[i].error > s.intervals[worst].error) worst = i
      const { a, b } = s.intervals[worst]
      const m = 0.5 * (a + b)
      const intervals = [
        ...s.intervals.slice(0, worst),
        kronrod15(f, a, m),
        kronrod15(f, m, b),
        ...s.intervals.slice(worst + 1),
      ]
      return { t: s.t + 1, intervals, ...summarise(intervals), split: { a, b }, evaluations: s.evaluations + 30 }
    },
    // Stop also when the worst interval can no longer be bisected in floating point.
    done: (s) =>
      s.converged ||
      !Number.isFinite(s.value) ||
      s.intervals.some((i) => {
        const m = 0.5 * (i.a + i.b)
        return i.error > 0 && (m <= Math.min(i.a, i.b) || m >= Math.max(i.a, i.b))
      }),
  }
}

/** The result of `integrate`. */
export type IntegrationResult = {
  value: number
  /** The estimated absolute error. */
  error: number
  evaluations: number
  /** Subintervals used (in the transformed variable for infinite limits). */
  intervals: number
  converged: boolean
}

/**
 * ∫ₐᵇ f(x) dx by globally adaptive Gauss–Kronrod 7–15, like scipy's `quad`, at most `maxIntervals` subintervals
 * (default 200). Infinite limits are mapped to (0, 1] as in QUADPACK's QAGI: x = a + (1 − t)/t for [a, ∞),
 * x = b − (1 − t)/t for (−∞, b], and f(x) + f(−x) on [0, ∞) for (−∞, ∞). The Kronrod nodes never touch t = 0.
 */
export function integrate(
  f: Integrand,
  a: number,
  b: number,
  options: { atol?: number; rtol?: number; maxIntervals?: number } = {},
): IntegrationResult {
  if (a === b) return { value: 0, error: 0, evaluations: 0, intervals: 0, converged: true }
  if (a > b) {
    const r = integrate(f, b, a, options)
    return { ...r, value: -r.value }
  }
  let g: Integrand = f
  let lo = a
  let hi = b
  if (a === -Infinity && b === Infinity) {
    g = (t) => {
      const x = (1 - t) / t
      return (f(x) + f(-x)) / (t * t)
    }
    ;[lo, hi] = [0, 1]
  } else if (b === Infinity) {
    g = (t) => f(a + (1 - t) / t) / (t * t)
    ;[lo, hi] = [0, 1]
  } else if (a === -Infinity) {
    g = (t) => f(b - (1 - t) / t) / (t * t)
    ;[lo, hi] = [0, 1]
  }
  const s = run(gaussKronrod(g, options), { a: lo, b: hi }, (options.maxIntervals ?? 200) - 1)
  return {
    value: s.value,
    error: s.error,
    evaluations: s.evaluations,
    intervals: s.intervals.length,
    converged: s.converged,
  }
}
