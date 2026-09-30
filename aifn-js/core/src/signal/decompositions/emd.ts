/**
 * Empirical mode decomposition (Huang et al., 1998, Proc. R. Soc. Lond. A 454): a signal is split into intrinsic mode
 * functions (IMFs) by sifting, x = Σ imfs + residue, fastest mode first, returned as a `Decomposition`. `siftSteps` is
 * the sifting of one IMF as a traceable algorithm; `eemd` is ensemble EMD (Wu and Huang, 2009, Adv. Adapt. Data Anal.
 * 1(1)).
 */

import { child, normals, type Stream } from 'aifn/foundation/random'
import { dense, fromData, type Tensor } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import type { Decomposition, Scalar, Size, Status } from 'aifn/foundation/contracts'
import { readSamples, type Samples, type SignalInput } from '../signal'
import { emd as emdCore, findExtrema, sift, siftStep, stopTest, type StopRule } from './emd-core'

export type { StopRule }

/** The stopping rules for sifting one IMF; see `StopRule`. Rilling's defaults: θ₁ = 0.05, θ₂ = 0.5, α = 0.05. */
export const RILLING_RULE: StopRule = { kind: 'rilling', theta1: 0.05, theta2: 0.5, alpha: 0.05 }

/** Options for `emd` and `eemd`. */
export interface EmdOptions {
  /** Most IMFs to extract (−1 for no limit). Default −1. */
  maxImfs?: Size
  /** When to stop sifting one IMF. Default PyEMD's rule, `{ kind: 'pyemd' }`. */
  rule?: StopRule
  /** Mirror extrema at the ends before fitting envelopes. Default true. */
  mirror?: boolean
}

/** A `Decomposition` of the samples into IMFs (named `imf 1`, `imf 2`, …, fastest first) and the residue. */
function decomposition(
  method: string,
  input: Samples,
  imfs: readonly Float64Array[],
  residue: Float64Array,
): Decomposition {
  const n = input.values.length
  const vec = (a: Float64Array): Tensor => fromData(a, [a.length])
  return {
    kind: 'decomposition',
    method,
    axis: fromData(
      Float64Array.from({ length: n }, (_, k) => input.t0 + k / input.fs),
      [n],
    ),
    components: imfs.map((imf, k) => ({ name: `imf ${k + 1}`, values: vec(imf) })),
    residual: vec(residue),
    original: vec(Float64Array.from(input.values)),
  }
}

/**
 * Empirical mode decomposition of a real signal, as a `Decomposition` (`method: 'emd'`) over the signal's time axis:
 * the IMFs are the components and the residue is `residual`, so Σ components + residual = original.
 */
export function emd(x: SignalInput, options: EmdOptions = {}): Decomposition {
  const input = readSamples(x, 'emd')
  const d = emdCore(input.values, options)
  return decomposition('emd', input, d.imfs, d.residue)
}

/** Local maxima and minima (indices, int32) and the number of zero crossings of a signal. */
export function extrema(x: SignalInput): { maxima: Tensor; minima: Tensor; zeroCrossings: Size } {
  const e = findExtrema(readSamples(x, 'extrema').values)
  return {
    maxima: fromData(Int32Array.from(e.maxima), [e.maxima.length]),
    minima: fromData(Int32Array.from(e.minima), [e.minima.length]),
    zeroCrossings: e.zeroCrossings,
  }
}

/** Options for `siftSteps`. */
export interface SiftOptions {
  /** When to stop. Default PyEMD's rule. */
  rule?: StopRule
  /** Mirror extrema at the ends before fitting envelopes. Default true. */
  mirror?: boolean
}

/** A state of sifting: the candidate IMF h and the envelopes of the step that produced it. */
export interface SiftState extends Status {
  /** Sifting steps made (h_t). */
  t: Size
  /** The candidate after t sifting steps (h₀ = x). */
  h: Tensor
  /** The envelopes and their mean at the last step (empty tensors at step 0). */
  upper: Tensor
  lower: Tensor
  mean: Tensor
  maxima: Tensor
  minima: Tensor
  /** Consecutive balanced steps (S-number rule). */
  balancedRun: Size
  /** The stopping rule holds: h is an IMF. */
  converged: boolean
  /** Too few extrema to continue: h is then a residue, not an IMF. */
  terminated: boolean
}

const empty = fromData(new Float64Array(0), [0])
const emptyInt = fromData(new Int32Array(0), [0])

/**
 * Sifting one IMF out of x as a traceable algorithm (Rilling, Flandrin and Gonçalvès, 2003): each step fits
 * cubic-spline envelopes through the maxima and minima of h and subtracts their mean. Stops (`converged`) when the
 * stopping rule holds, or (`terminated`) when h has too few extrema. `init` takes no start.
 */
export function siftSteps(x: SignalInput, options: SiftOptions = {}): Algorithm<undefined, SiftState> {
  const { rule = { kind: 'pyemd' }, mirror = true } = options
  const values = readSamples(x, 'siftSteps').values
  return {
    name: 'sift',
    init: () => ({
      t: 0,
      h: fromData(Float64Array.from(values), [values.length]),
      upper: empty,
      lower: empty,
      mean: empty,
      maxima: emptyInt,
      minima: emptyInt,
      balancedRun: 0,
      converged: false,
      terminated: false,
    }),
    step: (s) => {
      const prev = dense.data(s.h)
      const step = siftStep(prev, mirror)
      if (!step) return { ...s, terminated: true }
      const t = s.t + 1
      const { converged, balancedRun } = stopTest(rule, prev, step, t, s.balancedRun, mirror)
      const n = step.next.length
      return {
        t,
        h: fromData(step.next, [n]),
        upper: fromData(step.upper, [n]),
        lower: fromData(step.lower, [n]),
        mean: fromData(step.mean, [n]),
        maxima: fromData(Int32Array.from(step.maxima), [step.maxima.length]),
        minima: fromData(Int32Array.from(step.minima), [step.minima.length]),
        balancedRun,
        converged,
        terminated: false,
      }
    },
  }
}

/** Sift one IMF out of x (the result of `siftSteps` run to the end, at most `maxSteps` sifts). */
export function siftImf(
  x: SignalInput,
  options: SiftOptions & { maxSteps?: Size } = {},
): { imf: Tensor; sifts: Size; oscillating: boolean } {
  const r = sift(readSamples(x, 'siftImf').values, options.rule, options.mirror, options.maxSteps)
  return { imf: fromData(r.imf, [r.imf.length]), sifts: r.steps.length, oscillating: r.oscillating }
}

/**
 * Ensemble EMD: the IMFs of x + ε σ_x w_i averaged over `trials` white-noise realisations w_i (trial i draws from
 * `child(s, 'trial', i)`), slot by slot; every trial is decomposed into exactly `maxImfs` IMFs. The sum of the
 * averaged IMFs and residue differs from x by the mean added noise, of size ε σ_x / √trials. Returns a
 * `Decomposition` (`method: 'eemd'`).
 */
export function eemd(
  s: Stream,
  x: SignalInput,
  options: { trials?: Size; epsilon?: Scalar; maxImfs: Size; rule?: StopRule },
): Decomposition {
  const { trials = 50, epsilon = 0.2, maxImfs, rule = { kind: 'fixed', sifts: 10 } } = options
  const input = readSamples(x, 'eemd')
  const v = input.values
  const n = v.length
  let mean = 0
  for (const u of v) mean += u / n
  let sd = 0
  for (const u of v) sd += (u - mean) ** 2 / n
  const scale = epsilon * Math.sqrt(sd)
  const imfs = Array.from({ length: maxImfs }, () => new Float64Array(n))
  const residue = new Float64Array(n)
  for (let trial = 0; trial < trials; trial++) {
    const w = dense.data(normals(child(s, 'trial', trial), [n]))
    const y = v.map((u, i) => u + scale * w[i])
    const d = emdCore(y, { maxImfs, rule })
    d.imfs.forEach((imf, j) => {
      for (let i = 0; i < n; i++) imfs[j][i] += imf[i] / trials
    })
    for (let i = 0; i < n; i++) residue[i] += d.residue[i] / trials
  }
  return decomposition('eemd', input, imfs, residue)
}
