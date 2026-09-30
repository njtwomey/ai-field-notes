/**
 * Empirical mode decomposition (Huang et al., 1998, Proc. R. Soc. Lond. A 454): a signal is split into intrinsic mode
 * functions (IMFs) by sifting, x = Σ imfs + residue, fastest mode first. `siftSteps` is the sifting of one IMF as a
 * traceable algorithm; `eemd` is ensemble EMD (Wu and Huang, 2009, Adv. Adapt. Data Anal. 1(1)).
 */

import { normal, type Stream } from 'aifn/random'
import { fromData, type Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { readSignal, type Signal } from './complex'
import {
  emd as emdCore,
  findExtrema,
  pyemdCheck,
  rillingStats,
  sdCriterion,
  sift,
  siftStep,
  type StopRule,
} from './emd-core'

export type { StopRule }

/** The stopping rules for sifting one IMF; see `StopRule`. Rilling's defaults: θ₁ = 0.05, θ₂ = 0.5, α = 0.05. */
export const RILLING_RULE: StopRule = { kind: 'rilling', theta1: 0.05, theta2: 0.5, alpha: 0.05 }

/** An empirical mode decomposition: IMFs (fastest first) and the residue; Σ imfs + residue = x. */
export interface EmdResult {
  /** IMFs as a [k, n] tensor. */
  imfs: Tensor
  residue: Tensor
}

/** Options for `emd` and `eemd`. */
export interface EmdOptions {
  /** Most IMFs to extract (−1 for no limit). Default −1. */
  maxImfs?: number
  /** When to stop sifting one IMF. Default PyEMD's rule, `{ kind: 'pyemd' }`. */
  rule?: StopRule
  /** Mirror extrema at the ends before fitting envelopes. Default true. */
  mirror?: boolean
}

function stack(rows: Float64Array[], n: number): Tensor {
  const out = new Float64Array(rows.length * n)
  rows.forEach((r, i) => out.set(r, i * n))
  return fromData(out, [rows.length, n])
}

/** Empirical mode decomposition of a real signal. */
export function emd(x: Signal, options: EmdOptions = {}): EmdResult {
  const v = readSignal(x, 'emd')
  const d = emdCore(v, options)
  return { imfs: stack(d.imfs, v.length), residue: fromData(d.residue) }
}

/** Local maxima and minima (indices) and the number of zero crossings of a signal. */
export function extrema(x: Signal): { maxima: Tensor; minima: Tensor; zeroCrossings: number } {
  const e = findExtrema(readSignal(x, 'extrema'))
  return {
    maxima: fromData(Int32Array.from(e.maxima)),
    minima: fromData(Int32Array.from(e.minima)),
    zeroCrossings: e.zeroCrossings,
  }
}

/** Options for `siftSteps`. */
export interface SiftOptions {
  x: Signal
  rule?: StopRule
  mirror?: boolean
}

/** A state of sifting: the candidate IMF h and the envelopes of the step that produced it. */
export interface SiftState {
  /** The candidate after `sifts` sifting steps (h₀ = x). */
  h: Tensor
  /** The envelopes and their mean at the last step (empty tensors at step 0). */
  upper: Tensor
  lower: Tensor
  mean: Tensor
  maxima: Tensor
  minima: Tensor
  sifts: number
  /** Consecutive balanced steps (S-number rule). */
  balancedRun: number
  /** Too few extrema to continue: h is then a residue, not an IMF. */
  exhausted: boolean
  converged: boolean
  rule: StopRule
  mirror: boolean
}

const empty = fromData(new Float64Array(0))
const emptyInt = fromData(new Int32Array(0))

/**
 * Sifting one IMF as a traceable algorithm: each step fits cubic-spline envelopes through the maxima and minima of h
 * and subtracts their mean. Done when the stopping rule holds or h has too few extrema (`exhausted`).
 */
export const siftSteps: Algorithm<SiftOptions, SiftState> = {
  name: 'sift',
  init({ x, rule = { kind: 'pyemd' }, mirror = true }) {
    return {
      h: fromData(readSignal(x, 'siftSteps')),
      upper: empty,
      lower: empty,
      mean: empty,
      maxima: emptyInt,
      minima: emptyInt,
      sifts: 0,
      balancedRun: 0,
      exhausted: false,
      converged: false,
      rule,
      mirror,
    }
  },
  step(s) {
    const prev = s.h.data as Float64Array
    const step = siftStep(prev, s.mirror)
    if (!step) return { ...s, exhausted: true }
    const h = step.next
    const n = s.sifts + 1
    const e = findExtrema(h)
    const balanced = Math.abs(e.maxima.length + e.minima.length - e.zeroCrossings) < 2
    let converged = false
    let balancedRun = s.balancedRun
    const rule = s.rule
    if (rule.kind === 'fixed') converged = n >= rule.sifts
    else if (rule.kind === 'pyemd') converged = balanced && pyemdCheck(h, prev, step)
    else if (rule.kind === 'sd') converged = sdCriterion(prev, h) < rule.threshold
    else if (rule.kind === 'snumber') {
      balancedRun = balanced ? balancedRun + 1 : 0
      converged = balancedRun >= rule.s
    } else {
      const next = siftStep(h, s.mirror)
      if (!next) converged = true
      else {
        const { fractionAbove, maxSigma } = rillingStats(next, rule.theta1)
        converged = balanced && fractionAbove <= rule.alpha && maxSigma < rule.theta2
      }
    }
    return {
      ...s,
      h: fromData(h),
      upper: fromData(step.upper),
      lower: fromData(step.lower),
      mean: fromData(step.mean),
      maxima: fromData(Int32Array.from(step.maxima)),
      minima: fromData(Int32Array.from(step.minima)),
      sifts: n,
      balancedRun,
      converged,
    }
  },
  done: (s) => s.exhausted || s.converged,
}

/** Sift one IMF out of x (the result of `siftSteps` run to the end, at most `maxSifts` steps). */
export function siftImf(
  x: Signal,
  options: { rule?: StopRule; mirror?: boolean; maxSifts?: number } = {},
): { imf: Tensor; sifts: number; oscillating: boolean } {
  const r = sift(readSignal(x, 'siftImf'), options.rule, options.mirror, options.maxSifts)
  return { imf: fromData(r.imf), sifts: r.steps.length, oscillating: r.oscillating }
}

/**
 * Ensemble EMD: the IMFs of x + ε σ_x w_i averaged over `trials` white-noise realisations w_i (drawn from
 * `s.child('trial', i)`), slot by slot; every trial is decomposed into exactly `maxImfs` IMFs. The sum of the averaged
 * IMFs and residue differs from x by the mean added noise, of size ε σ_x / √trials.
 */
export function eemd(
  s: Stream,
  x: Signal,
  options: { trials?: number; epsilon?: number; maxImfs: number; rule?: StopRule },
): EmdResult {
  const { trials = 50, epsilon = 0.2, maxImfs, rule = { kind: 'fixed', sifts: 10 } } = options
  const v = readSignal(x, 'eemd')
  const n = v.length
  let mean = 0
  for (const u of v) mean += u / n
  let sd = 0
  for (const u of v) sd += (u - mean) ** 2 / n
  const scale = epsilon * Math.sqrt(sd)
  const imfs = Array.from({ length: maxImfs }, () => new Float64Array(n))
  const residue = new Float64Array(n)
  for (let trial = 0; trial < trials; trial++) {
    const r = s.child('trial', trial)
    const y = v.map((u) => u + scale * normal(r))
    const d = emdCore(y, { maxImfs, rule })
    d.imfs.forEach((imf, j) => {
      for (let i = 0; i < n; i++) imfs[j][i] += imf[i] / trials
    })
    for (let i = 0; i < n; i++) residue[i] += d.residue[i] / trials
  }
  return { imfs: stack(imfs, n), residue: fromData(residue) }
}
