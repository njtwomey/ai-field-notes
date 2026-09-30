/**
 * `aifn/fields`: vector and scalar fields and the geometry of their flows x′ = f(x).
 *
 * - Grids: `gridAxes`, `sampleField`, `sampleScalar`, `directionField`, `slopeField`; level sets by marching squares
 *   (`contour`, `levelSet`) and `nullclines`.
 * - Calculus by autodiff: `jacobianAt`, `divergence`, `curl`, `gradientAt`; fields from scalars: `gradientField`,
 *   `hamiltonianField`.
 * - Flows: `flowMap`, `trajectory`, `streamline`, `streamlines`, `pushForward`; `poincareSection` and `limitCycle`;
 *   `transportDensity` (Liouville), and `transportDensityFrames` and `pushForwardFrames` at every frame of a span.
 * - Fixed points: `fixedPoints` (Newton from seeds), `linearise`, `classifyLinear`, `invariantManifolds`; Lyapunov
 *   functions: `lyapunovDerivative`, `lyapunovCheck`.
 *
 * Fields are functions of a rank-1 tensor; written with `aifn/tensor` primitives they can be differentiated. Grid
 * samples are ny × nx matrices with entry [i, j] at (x_j, y_i), the layout of a heatmap.
 */

export {
  contour,
  directionField,
  gridAxes,
  levelSet,
  nullclines,
  sampleField,
  sampleScalar,
  slopeField,
  type FieldSample,
  type Grid2,
  type ScalarField,
  type Segments,
  type VectorField,
} from './grid'
export { curl, divergence, gradientAt, gradientField, hamiltonianField, jacobianAt } from './calculus'
export {
  flowMap,
  limitCycle,
  poincareSection,
  pushForward,
  pushForwardFrames,
  streamline,
  streamlines,
  trajectory,
  transportDensity,
  transportDensityFrames,
  type Box,
  type FlowOptions,
  type LimitCycle,
  type Section,
  type SectionOptions,
  type StreamlineOptions,
  type TransportFrameOptions,
  type TransportOptions,
} from './flow'
export {
  classifyLinear,
  fixedPoints,
  invariantManifolds,
  linearise,
  lyapunovCheck,
  lyapunovDerivative,
  type Classification,
  type FixedPoint,
  type FixedPointKind,
  type FixedPointOptions,
  type LyapunovCheck,
  type Manifolds,
} from './fixed'
