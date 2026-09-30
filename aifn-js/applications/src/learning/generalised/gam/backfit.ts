/**
 * Backfitting a GAM (Hastie and Tibshirani, 1990, "Generalized Additive Models", §4.4 and §6.5): the terms are built
 * on the data as in `gam` (centred smooths, λ-weighted penalties) and handed to the generalised models' shared
 * `backfitting`, so its fixed point is the penalised fit `gam` finds at the same λ (for the Gaussian family).
 */

import { fromData, toFlat, type Tensor } from 'aifn/foundation/tensor'
import type { Algorithm } from 'aifn/foundation/trace'
import type { Family, Link, LinkName } from 'aifn/probability/likelihoods'
import { backfitting, type BackfitState } from '../backfitting'
import { buildTerms, times, type TermSpec } from './terms'

/** The problem a GAM backfitting run solves. */
export type GamBackfitProblem = {
  terms: readonly TermSpec[]
  x: Tensor
  y: Tensor
  weights?: Tensor
  family?: Family
  link?: LinkName | Link
  /** One λ per penalty in term order (default: each term's fixed λ, else 1). */
  lambdas?: number[]
  /** Converged when the largest change of any fⱼ(xᵢ) is below tolerance × (1 + max |f|) (default 1e-8). */
  tolerance?: number
}

/** Backfitting of a GAM's terms as an `Algorithm` (no start). */
export function gamBackfitting(problem: GamBackfitProblem): Algorithm<void, BackfitState> {
  const [n, d] = problem.x.shape
  const X = Float64Array.from(toFlat(problem.x))
  const terms = buildTerms(problem.terms, X, n, d)
  let k = 0
  const designs = terms.map((t) => fromData(times(t.raw(X, n, d), n, t.rawSize, t.Z, t.size), [n, t.size]))
  const penalties = terms.map((t) => {
    const S = new Float64Array(t.size * t.size)
    t.penalties.forEach((Sk, j) => {
      const l = problem.lambdas?.[k] ?? (Number.isNaN(t.fixedLambda[j]) ? 1 : t.fixedLambda[j])
      k++
      for (let i = 0; i < S.length; i++) S[i] += l * Sk[i]
    })
    return fromData(S, [t.size, t.size])
  })
  return {
    ...backfitting({
      designs,
      penalties,
      y: problem.y,
      weights: problem.weights,
      family: problem.family,
      link: problem.link,
      tolerance: problem.tolerance,
    }),
    name: 'gam-backfitting',
  }
}
