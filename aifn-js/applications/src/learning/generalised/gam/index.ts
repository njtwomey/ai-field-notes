/**
 * `aifn-applied/learning/generalised/gam`: generalised additive models: terms (smooths, cyclic, thin-plate, tensor
 * products, factor and linear terms, shape constraints), `gam` with smoothing selection (REML, GCV), backfitting,
 * expectile GAMs and the explainable boosting machine.
 */

export {
  cyclic,
  factorTerm,
  linearTerm,
  s,
  te,
  thinPlate,
  type BuiltTerm,
  type ShapeConstraint,
  type SmoothOptions,
  type TermSpec,
} from './terms'
export { gam, type GamData, type GamModel, type GamParams, type PartialEffect, type SmoothingMethod } from './model'
export { expectileGam, expectileLaws, type ExpectileGamParams, type ExpectileState } from './expectile'
export { ebmBoosting, explainableBoostingMachine, type EbmModel, type EbmParams, type EbmState } from './ebm'
export { gamBackfitting, type GamBackfitProblem } from './backfit'
