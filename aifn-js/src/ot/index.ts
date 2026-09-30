/**
 * `aifn/ot`: optimal transport.
 *
 * - Discrete: `costMatrix` (‖x − y‖^p), `uniformWeights`, `exactTransport` (Hungarian for equal uniform masses, the
 *   simplex method otherwise, via `aifn/programming`), `sinkhornSteps` (log-domain, traceable: potentials and the plan
 *   per iteration) and `sinkhorn`.
 * - On the line: `wasserstein1d` (any p, weighted), `monotonePlan`, `barycenter1d` (quantile averaging).
 * - Higher dimensions: `slicedWasserstein`.
 * - Structure matching: `gromovWassersteinSteps` and `gromovWasserstein` (entropic, square loss).
 */

export {
  costMatrix,
  exactTransport,
  sinkhorn,
  sinkhornSteps,
  uniformWeights,
  type CostInput,
  type PointsInput,
  type SinkhornOptions,
  type SinkhornState,
  type TransportPlan,
  type WeightsInput,
} from './discrete'
export {
  barycenter1d,
  monotonePlan,
  slicedWasserstein,
  wasserstein1d,
  type Barycenter1d,
  type MonotonePlan,
  type SlicedWasserstein,
} from './oneD'
export { gromovWasserstein, gromovWassersteinSteps, type GromovOptions, type GromovState } from './gromov'
