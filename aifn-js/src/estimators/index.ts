/**
 * `aifn/estimators`: typed capabilities of fitted models and the mixins that build them (plan §5.1), the estimator
 * shape (`fit(data, options)` returning a plain fitted object), `evaluate`, and two reference estimators.
 *
 * - Capabilities: `Fitted` (`forward`), `Decides`, `Predicts`, `Expects`, `Scores`, `Transforms`, `Samples`,
 *   `Trained`; guards `hasPredictive` and the like; `capabilities(m)`.
 * - Mixins: `withDecision(model, rule)` (argmax, mode, threshold, cost matrix), `withExpectation`, `withSampling`,
 *   `readout(model, complete)`.
 * - Data: `Dataset`, `Supervised`, `Table`, `rowCount`, `takeRows`, `takeData`; `Estimator`, `FitOptions`.
 * - Reference estimators: `linearRegression` (least squares or ridge, Gaussian predictive) and `logisticRegression`
 *   (Newton/IRLS as a traceable `Algorithm`, Bernoulli or categorical predictive).
 * - Evaluation: `evaluate(model, data, metrics)` with metrics typed by the capability they need.
 *
 * Predictive distributions follow the README's distribution contract; until `aifn/distributions` lands they are the
 * stopgaps `gaussianPredictive`, `bernoulliPredictive` and `categoricalPredictive` (see distribution.ts).
 */

export {
  capabilities,
  hasDecide,
  hasExpect,
  hasForward,
  hasPredictive,
  hasSample,
  hasScore,
  hasTraining,
  hasTransform,
  type Capability,
  type DecisionOf,
  type Decides,
  type Expects,
  type Fitted,
  type HeadOf,
  type InputOf,
  type PredictiveOf,
  type Predicts,
  type Samples,
  type Scores,
  type Trained,
  type Transforms,
} from './capabilities'
export {
  rowCount,
  takeData,
  takeRows,
  type Column,
  type DataOf,
  type Dataset,
  type Estimator,
  type Features,
  type FitOptions,
  type ModelOf,
  type Supervised,
  type Table,
} from './data'
export {
  asTensor,
  bernoulliPredictive,
  categoricalPredictive,
  classProbabilities,
  expectation,
  gaussHermite,
  gaussianPredictive,
  isClassDistribution,
  isUnivariate,
  meanOf,
  varianceOf,
  type BernoulliPredictive,
  type CategoricalPredictive,
  type ClassDistribution,
  type Distribution,
  type GaussianPredictive,
  type UnivariateDistribution,
} from './distribution'
export {
  readout,
  withDecision,
  withExpectation,
  withSampling,
  type Completers,
  type DecisionRule,
  type ReadoutOf,
} from './mixins'
export { linearRegression, type LinearRegressionModel, type LinearRegressionParams } from './linear'
export {
  logisticIrls,
  logisticRegression,
  type IrlsProblem,
  type IrlsState,
  type LogisticRegressionModel,
  type LogisticRegressionParams,
} from './logistic'
export {
  accuracy,
  adaptMetric,
  defineMetric,
  evaluate,
  logLoss,
  meanAbsoluteError,
  meanSquaredError,
  outputFor,
  outputs,
  predictiveMean,
  rSquared,
  type AnyMetric,
  type Metric,
  type MetricFunctionLike,
  type Need,
  type OutputFor,
  type Requirement,
} from './evaluate'
