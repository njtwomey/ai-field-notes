/**
 * `aifn/probability/bijectors`: bijectors and supports: intervals (`interval`, `REALS`, `POSITIVE`, `UNIT`),
 * invertible maps with log-Jacobians (exp, log, softplus, sigmoid, affine, power, the normal CDF, chains) and
 * many-to-one maps with their branches, the ordered bijector onto increasing vectors (ordinal thresholds), for transformed distributions and constrained parameters.
 */

export {
  affineBijector,
  asManyToOne,
  branchImages,
  chainBijectors,
  expBijector,
  formatInterval,
  imageOf,
  interval,
  intervalInside,
  intervalSupport,
  logBijector,
  normalCdfBijector,
  orderedBijector,
  POSITIVE,
  powerBijector,
  REALS,
  sigmoidBijector,
  softplusBijector,
  squareMap,
  supportInteriorPoint,
  supportInterval,
  tanhBijector,
  UNIT_INTERVAL,
  type Bijector,
  type Branch,
  type Interval,
  type ManyToOneMap,
  type OrderedOptions,
} from './maps'
