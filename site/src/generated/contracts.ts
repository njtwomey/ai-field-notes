/* Generated from python/mlc/core/contracts.py via contracts.schema.json by `make contracts`. Do not edit. */

/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Slug".
 */
export type Slug = string
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "FigureId".
 */
export type FigureId = string
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Output".
 */
export type Output = TextOutput | MetricsOutput | TableOutput | ChartOutput | HeatmapOutput | FileOutput
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Cell".
 */
export type Cell = number | string | null
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Penalty".
 */
export type Penalty = 'l2' | 'l1' | 'elasticnet'

/**
 * Generated from python/mlc/core/contracts.py by `uv run mlc schema`. Do not edit.
 */
export interface Contracts {
  Manifest?: Manifest
  RunRecord?: RunRecord
  Registry?: Registry
  Point2d?: Point2d
  PointCloud2d?: PointCloud2d
  Samples1d?: Samples1d
  Curve?: Curve
  CurveSet?: CurveSet
  Grid2d?: Grid2d
  ContingencyTable?: ContingencyTable
  RocCurve?: RocCurve
  ElasticNetPaths?: ElasticNetPaths
  ImageSvd?: ImageSvd
  LarsPaths?: LarsPaths
  LassoPath?: LassoPath
  LogisticValley?: LogisticValley
  LossSurface?: LossSurface
  ModelSelectionTable?: ModelSelectionTable
  RegressionSurface?: RegressionSurface
  RidgePath?: RidgePath
  SgdTrajectories?: SgdTrajectories
  TypeIErrorMaps?: TypeIErrorMaps
}
/**
 * Written to ``generated/manifest.json``. The site's single entry point to generated assets.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Manifest".
 */
export interface Manifest {
  examples: {
    [k: string]: ExampleIndex
  }
  figures: {
    [k: string]: FigureIndex
  }
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "ExampleIndex".
 */
export interface ExampleIndex {
  id: Slug
  title: string
  module: string
  /**
   * Repo-relative paths of every source file, in display order.
   */
  sources: string[]
  runs: RunIndex[]
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "RunIndex".
 */
export interface RunIndex {
  name: Slug
  title: string
  description: string
  command: string
  exit_code: number
  /**
   * Relative to ``generated/``.
   */
  path: string
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "FigureIndex".
 */
export interface FigureIndex {
  id: FigureId
  title: string
  /**
   * Relative to ``generated/``.
   */
  path: string
}
/**
 * Written to ``generated/runs/<example>/<run>/run.json``.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "RunRecord".
 */
export interface RunRecord {
  example: Slug
  run: Slug
  title: string
  description: string
  command: string
  exit_code: number
  duration_s: number
  hash: string
  created: string
  stdout: string
  stderr: string
  outputs: Output[]
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "TextOutput".
 */
export interface TextOutput {
  title: string | null
  kind: 'text'
  mime: 'text/plain'
  text: string
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "MetricsOutput".
 */
export interface MetricsOutput {
  title: string | null
  kind: 'metrics'
  mime: 'application/vnd.mlc.metrics+json'
  values: {
    [k: string]: number | string
  }
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "TableOutput".
 */
export interface TableOutput {
  title: string | null
  kind: 'table'
  mime: 'application/vnd.mlc.table+json'
  columns: string[]
  rows: Cell[][]
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "ChartOutput".
 */
export interface ChartOutput {
  title: string | null
  kind: 'chart'
  mime: 'application/vnd.mlc.chart+json'
  x_label: string
  y_label: string
  /**
   * @minItems 1
   */
  series: [Series, ...Series[]]
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Series".
 */
export interface Series {
  name: string
  type: 'scatter' | 'line'
  x: number[]
  y: number[]
  /**
   * Optional categorical slot index per point (scatter only). Colours follow design/palette.json.
   */
  group: number[] | null
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "HeatmapOutput".
 */
export interface HeatmapOutput {
  title: string | null
  kind: 'heatmap'
  mime: 'application/vnd.mlc.heatmap+json'
  x: number[]
  y: number[]
  /**
   * Row-major: ``z[i][j]`` is the value at ``(x[j], y[i])``.
   */
  z: number[][]
  x_label: string
  y_label: string
  overlay: Series[]
}
/**
 * A binary or large asset written next to the run record, e.g. a PNG.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "FileOutput".
 */
export interface FileOutput {
  title: string | null
  kind: 'file'
  mime: string
  /**
   * Relative to the run directory.
   */
  path: string
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Registry".
 */
export interface Registry {
  examples: ExampleSpec[]
}
/**
 * A runnable implementation attached to a note. ``id`` is referenced by a note's ``code`` frontmatter field.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "ExampleSpec".
 */
export interface ExampleSpec {
  id: Slug
  title: string
  module: string
  /**
   * @minItems 1
   */
  runs: [RunSpec, ...RunSpec[]]
}
/**
 * One CLI invocation of an example. Its outputs are cached as static assets.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "RunSpec".
 */
export interface RunSpec {
  name: Slug
  title: string
  description: string
  args: string[]
}
/**
 * A single point, e.g. an optimum in parameter space. Use instead of a tuple: tuples lose their types in the
 * generated TypeScript.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Point2d".
 */
export interface Point2d {
  x: number
  y: number
}
/**
 * Points in the plane, optionally labelled with a categorical group per point.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "PointCloud2d".
 */
export interface PointCloud2d {
  x: number[]
  y: number[]
  /**
   * Categorical palette slot per point.
   */
  group: number[] | null
  /**
   * Display name per group index.
   */
  group_names: string[] | null
}
/**
 * Scalar samples, optionally labelled with a categorical group per sample, e.g. a 1-D classification dataset.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Samples1d".
 */
export interface Samples1d {
  x: number[]
  /**
   * Categorical palette slot per sample.
   */
  group: number[] | null
  /**
   * Display name per group index.
   */
  group_names: string[] | null
}
/**
 * One named y-against-x line, e.g. a training-loss curve.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Curve".
 */
export interface Curve {
  name: string
  x: number[]
  y: number[]
}
/**
 * Curves that share axes, e.g. train and validation loss.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "CurveSet".
 */
export interface CurveSet {
  x_label: string
  y_label: string
  /**
   * @minItems 1
   */
  curves: [Curve, ...Curve[]]
}
/**
 * Values on a regular grid, e.g. a loss surface or a probability map.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "Grid2d".
 */
export interface Grid2d {
  x: number[]
  y: number[]
  /**
   * Row-major: ``z[i][j]`` is the value at ``(x[j], y[i])``.
   */
  z: number[][]
  x_label: string
  y_label: string
  z_label: string
}
/**
 * Counts cross-tabulated by two categorical variables. A confusion matrix is the special case rows=actual.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "ContingencyTable".
 */
export interface ContingencyTable {
  row_label: string
  col_label: string
  rows: string[]
  cols: string[]
  counts: number[][]
}
/**
 * Receiver operating characteristic: true-positive rate against false-positive rate over thresholds.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "RocCurve".
 */
export interface RocCurve {
  fpr: number[]
  tpr: number[]
  thresholds: number[]
  auc: number
}
/**
 * Elastic-net paths for several mixing ratios; ratio 1 is the lasso.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "ElasticNetPaths".
 */
export interface ElasticNetPaths {
  l1_ratios: number[]
  paths: CoefficientPath[]
}
/**
 * Coefficients of a regularised fit along a grid of penalty values, one row per feature.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "CoefficientPath".
 */
export interface CoefficientPath {
  features: string[]
  /**
   * Penalty values, decreasing: from all-zero (or near) to least penalised.
   */
  penalty: number[]
  /**
   * Indexed [feature][penalty index].
   */
  coef: number[][]
}
/**
 * A test image and its thin SVD. The site rebuilds any rank-k approximation from these factors.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "ImageSvd".
 */
export interface ImageSvd {
  /**
   * Row-major, first row at the top, values in [0, 1].
   */
  image: number[][]
  /**
   * Shape (size, size): left singular vectors as columns.
   */
  u: number[][]
  /**
   * Singular values, largest first.
   */
  s: number[]
  /**
   * Shape (size, size): right singular vectors as rows.
   */
  vt: number[][]
}
/**
 * LARS and the LARS–lasso modification on the diabetes data (Efron et al. 2004).
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "LarsPaths".
 */
export interface LarsPaths {
  features: string[]
  lar: KnotPath
  lasso: KnotPath
}
/**
 * A piecewise-linear path given by its knots; coefficients are linear in ‖w‖₁ between consecutive knots.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "KnotPath".
 */
export interface KnotPath {
  l1_norm: number[]
  alphas: number[]
  /**
   * Indexed [feature][knot].
   */
  coef: number[][]
  /**
   * Features entering or leaving the active set, in order.
   */
  events: PathEvent[]
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "PathEvent".
 */
export interface PathEvent {
  /**
   * Index of the knot at which the event happens.
   */
  knot: number
  /**
   * Penalty at that knot, in scikit-learn's lasso scaling.
   */
  alpha: number
  feature: number
  kind: 'enter' | 'drop'
}
/**
 * Lasso coefficients against α on the diabetes data, with the OLS solution.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "LassoPath".
 */
export interface LassoPath {
  path: CoefficientPath
  /**
   * Number of nonzero coefficients at each α.
   */
  nonzero: number[]
  ols: number[]
}
/**
 * Log-cross-entropy of a 1-D logistic regression over (w, b), with the data it was computed on.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "LogisticValley".
 */
export interface LogisticValley {
  surface: Grid2d
  data: Samples1d
  optimum: Point2D1
  optimum_loss: number
}
/**
 * A single point, e.g. an optimum in parameter space. Use instead of a tuple: tuples lose their types in the
 * generated TypeScript.
 */
export interface Point2D1 {
  x: number
  y: number
}
/**
 * Cross-entropy over the two weights, with the dataset it was computed on.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "LossSurface".
 */
export interface LossSurface {
  surface: Grid2d
  data: PointCloud2d
}
/**
 * Deviance −2 log L of the best EM fit for every sample size, restart budget and k. AIC and BIC follow from it.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "ModelSelectionTable".
 */
export interface ModelSelectionTable {
  ks: number[]
  /**
   * Sample sizes. Each uses the first n points of `blobs`, which are a random subsample.
   */
  ns: number[]
  max_restarts: number
  /**
   * Indexed [restarts − 1][n index][k index].
   */
  deviance: number[][][]
  /**
   * The best fit behind each deviance, indexed likewise.
   */
  fits: FittedMixture[][][]
}
/**
 * A diagonal-covariance Gaussian mixture in 2-D.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "FittedMixture".
 */
export interface FittedMixture {
  weights: number[]
  means: Point2d[]
  /**
   * Per-component variances along x and y.
   */
  variances: Point2d[]
}
/**
 * Log-MSE of y ≈ w·x + b over (w, b), with the data it was computed on.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "RegressionSurface".
 */
export interface RegressionSurface {
  surface: Grid2d
  data: PointCloud2d
  optimum: Point2D2
  optimum_loss: number
}
/**
 * A single point, e.g. an optimum in parameter space. Use instead of a tuple: tuples lose their types in the
 * generated TypeScript.
 */
export interface Point2D2 {
  x: number
  y: number
}
/**
 * Ridge coefficients against λ on the diabetes data, with effective degrees of freedom and the OLS solution.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "RidgePath".
 */
export interface RidgePath {
  path: CoefficientPath
  /**
   * Effective degrees of freedom Σ dᵢ²/(dᵢ² + λ) at each λ.
   */
  dof: number[]
  ols: number[]
}
/**
 * SGDRegressor weights per epoch for several penalties, against the exact solutions.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "SgdTrajectories".
 */
export interface SgdTrajectories {
  features: string[]
  alpha: number
  l1_ratio: number
  runs: SgdRun[]
}
/**
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "SgdRun".
 */
export interface SgdRun {
  penalty: Penalty
  label: string
  /**
   * Indexed [feature][epoch]; epoch 0 is the zero start.
   */
  coef: number[][]
  /**
   * The exact minimiser of the same objective.
   */
  exact: number[]
  /**
   * Objective minus the exact optimum at each epoch.
   */
  gap: number[]
  /**
   * Number of coefficients exactly zero at each epoch.
   */
  zeros: number[]
  exact_zeros: number
}
/**
 * Type I error rates of the pooled and Welch tests over the variance ratio and the second group's size.
 *
 * This interface was referenced by `Contracts`'s JSON-Schema
 * via the `definition` "TypeIErrorMaps".
 */
export interface TypeIErrorMaps {
  pooled: Grid2d
  welch: Grid2d
  n1: number
  alpha: number
}
