/**
 * `aifn/signal/statistical`: statistical signal processing. Autoregressive (AR) estimation by the Yule–Walker
 * equations (`yuleWalker`, through `aifn/numerics/linalg`'s `levinsonDurbin`) and by Burg's method (`burg`); adaptive
 * FIR filters as step-through algorithms (`lms`, `nlms`, `rls`), one sample per step.
 */

export { burg, yuleWalker, type AutoregressiveFit } from './autoregression'
export {
  lms,
  nlms,
  rls,
  type AdaptiveFilterOptions,
  type AdaptiveFilterStart,
  type AdaptiveFilterState,
  type RlsState,
} from './adaptive'
export { statisticalAlgorithms, statisticalFunctions } from './registry'
