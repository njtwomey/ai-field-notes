/**
 * `aifn-applied/learning/explanation`: studies that compare explanation methods of `aifn/learning/explain` on planted
 * truths, for the lab's workers: data attribution (influence, TracIn, KNN- and data Shapley) against planted label
 * noise.
 */

export {
  dataValuationStudy,
  VALUATION_METHODS,
  type ValuationMethod,
  type ValuationOptions,
  type ValuationSnapshot,
} from './valuation'
export { explanationFunctions } from './registry'
