/**
 * Expectile GAMs by asymmetric least squares (Newey and Powell, 1987; Schnabel and Eilers, 2009, "Optimal expectile
 * smoothing"): the τ-expectile curve minimises Σ |τ − 1(yᵢ < fᵢ)| (yᵢ − fᵢ)², fitted by iterating weighted Gaussian
 * GAM fits with weight τ above the curve and 1 − τ below it until no point changes side (LAWS).
 */

import type { Status } from 'aifn/foundation/contracts'
import { gaussianFamily } from 'aifn/probability/likelihoods'
import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/foundation/trace'
import type { Estimator } from 'aifn/learning/estimators'
import { gam, type GamData, type GamModel, type GamParams } from './model'

/** A LAWS state. */
export type ExpectileState = Status & {
  /** LAWS iterations done. */
  t: number
  /** Asymmetric weights τ or 1 − τ per point, [n]. */
  weights: Tensor
  /** The curve at the data, [n]. */
  fitted: Tensor
  /** Points whose side of the curve (and so weight) differs from the weights this fit used. */
  switched: number
  /** Share of points below the curve. */
  below: number
  converged: boolean
}

/** Hyperparameters of `expectileGam`. */
export type ExpectileGamParams = Omit<GamParams, 'family' | 'link'> & {
  /** The asymmetry τ ∈ (0, 1); ½ gives the mean. */
  tau: number
  /** Most LAWS iterations (default 50). */
  maxLawsSteps?: number
}

/** LAWS as a traceable algorithm; each step is one weighted GAM fit. */
export function expectileLaws(params: ExpectileGamParams, data: GamData): Algorithm<void, ExpectileState> {
  const { tau } = params
  if (!(tau > 0 && tau < 1)) throw new Error('expectileGam: τ must be in (0, 1)')
  const y = Float64Array.from(toFlat(data.y))
  const n = y.length
  const prior = data.weights ? Float64Array.from(toFlat(data.weights)) : new Float64Array(n).fill(1)
  const estimator = gam({ ...params, family: gaussianFamily() })
  const fitWith = (w: Float64Array) =>
    estimator.fit({
      ...data,
      weights: fromData(
        Float64Array.from(w, (v, i) => v * prior[i]),
        [n],
      ),
    })
  /** The asymmetric weights the curve f implies: τ above it, 1 − τ below. */
  const sides = (f: Float64Array) => Float64Array.from(y, (v, i) => (v > f[i] ? tau : 1 - tau))
  const stateOf = (w: Float64Array, model: GamModel, t: number): ExpectileState => {
    const f = Float64Array.from(toFlat(model.fitted))
    const next = sides(f)
    let switched = 0
    let below = 0
    for (let i = 0; i < n; i++) {
      if (t > 0 && next[i] !== w[i]) switched++
      if (y[i] < f[i]) below++
    }
    return {
      t,
      weights: fromData(w, [n]),
      fitted: model.fitted,
      switched,
      below: below / n,
      converged: t > 0 && switched === 0,
    }
  }
  return {
    name: 'expectile-laws',
    init: () => {
      const w = new Float64Array(n).fill(0.5)
      return stateOf(w, fitWith(w), 0)
    },
    step: (state) => {
      const w = sides(Float64Array.from(toFlat(state.fitted)))
      return stateOf(w, fitWith(w), state.t + 1)
    },
  }
}

/**
 * A τ-expectile GAM: the model of the last LAWS iteration (a Gaussian GAM fitted with the final asymmetric weights),
 * with the LAWS run in `laws`.
 */
export function expectileGam(
  params: ExpectileGamParams,
): Estimator<GamData, GamModel & { tau: number; laws: Trace<ExpectileState> }> {
  const { maxLawsSteps = 50 } = params
  return {
    name: 'expectile-gam',
    params,
    fit(data) {
      const laws = trace(expectileLaws(params, data), undefined, maxLawsSteps, {
        record: { switched: (s) => s.switched, below: (s) => s.below },
      })
      const final = laws.final
      const n = final.weights.shape[0]
      const prior = data.weights ? toFlat(data.weights) : null
      const w = Float64Array.from(toFlat(final.weights), (v, i) => v * (prior ? prior[i] : 1))
      const model = gam({ ...params, family: gaussianFamily() }).fit({ ...data, weights: fromData(w, [n]) })
      return { ...model, tau: params.tau, laws }
    },
  }
}
