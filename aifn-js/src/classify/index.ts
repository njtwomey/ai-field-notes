/**
 * `aifn/classify`: classifiers as estimators in the `aifn/estimators` capability style (`forward`, `score`, `decide`,
 * and `predictive` where the model gives probabilities). Iterative fits are traceable `Algorithm`s (`…Steps`), kept on
 * the fitted model as `training`.
 *
 * - Neighbours: `kNearestNeighbours` (uniform or distance weights; Euclidean, Manhattan, Chebyshev, Minkowski),
 *   `kNearestNeighboursRegression`; the neighbours are exposed.
 * - Generative: `gaussianNaiveBayes`, `multinomialNaiveBayes`, `bernoulliNaiveBayes`, `linearDiscriminant` (with its
 *   discriminant projection), `quadraticDiscriminant`.
 * - Linear and kernel machines: `perceptron` (`perceptronSteps`), `supportVectorMachine` by SMO (`smoSteps`: working
 *   pair, dual variables, gradient, bias and KKT gap per step), `linearSvm` by dual coordinate descent
 *   (`dualCoordinateSteps`) or Pegasos (`pegasosSteps`). Kernels come from `aifn/kernels` (`rbf` by lengthscale,
 *   `linear`, `polynomial`, …).
 * - Trees: `decisionTree`, `regressionTree` (CART; Gini, entropy, squared error), growth as `treeGrowthSteps`, the
 *   split search `splitSearch`, `growTree`, `applyTree`, `predictTree`, `decisionPath`, `featureImportances`,
 *   `treeSize`, `costComplexityPath`, `pruneTree`. Trees are plain `{ nodes, root }` data.
 * - Ensembles: `randomForest`, `adaBoost` (SAMME; `adaBoostSteps`), `gradientBoosting` (squared, logistic;
 *   `gradientBoostingSteps`).
 * - Multiclass: `oneVersusRest`, `oneVersusOne`, `outputCode` (Hamming or loss decoding) with `oneVersusRestCode`,
 *   `oneVersusOneCode`, `exhaustiveCode`, `randomCode`, `codeDistance`; `nestedDichotomies` with `dichotomyTree`;
 *   `crammerSinger` (`crammerSingerSteps`).
 */

export {
  kNearestNeighbours,
  kNearestNeighboursRegression,
  type Metric,
  type Neighbours,
  type NeighboursClassifier,
  type NeighboursParams,
  type NeighboursRegressor,
} from './neighbours'
export {
  bernoulliNaiveBayes,
  gaussianNaiveBayes,
  linearDiscriminant,
  multinomialNaiveBayes,
  quadraticDiscriminant,
  type DiscreteNaiveBayesModel,
  type DiscriminantModel,
  type GaussianNaiveBayesModel,
  type GenerativeClassifier,
  type LinearDiscriminantModel,
} from './bayes'
export {
  perceptron,
  perceptronSteps,
  type PerceptronModel,
  type PerceptronProblem,
  type PerceptronState,
} from './perceptron'
export {
  dualCoordinateSteps,
  dualDecision,
  linearSvm,
  pegasosSteps,
  smoSteps,
  supportVectorMachine,
  type LinearSvmModel,
  type LinearSvmProblem,
  type LinearSvmState,
  type SmoProblem,
  type SmoState,
  type SupportVectorMachineModel,
} from './svm'
export {
  applyTree,
  costComplexityPath,
  decisionPath,
  featureImportances,
  growTree,
  nodeLabel,
  predictTree,
  pruneTree,
  splitSearch,
  treeGrowthSteps,
  treeSize,
  type Criterion,
  type DecisionTree,
  type FeatureSplits,
  type SplitSearch,
  type TreeGrowthState,
  type DecisionNode,
  type SplitData,
  type TreeParams,
  type TreeProblem,
} from './tree'
export {
  adaBoost,
  adaBoostSteps,
  decisionTree,
  gradientBoosting,
  gradientBoostingSteps,
  randomForest,
  regressionTree,
  type AdaBoostModel,
  type AdaBoostProblem,
  type AdaBoostState,
  type BoostingLoss,
  type DecisionTreeModel,
  type GradientBoostingModel,
  type GradientBoostingProblem,
  type GradientBoostingState,
  type RandomForestModel,
  type RegressionTreeModel,
  type WeightedData,
} from './ensembles'
export {
  codeDistance,
  crammerSinger,
  crammerSingerSteps,
  dichotomyTree,
  exhaustiveCode,
  nestedDichotomies,
  oneVersusOne,
  oneVersusOneCode,
  oneVersusRest,
  oneVersusRestCode,
  outputCode,
  randomCode,
  softmaxScores,
  type BinaryEstimator,
  type BinaryModel,
  type CrammerSingerModel,
  type CrammerSingerProblem,
  type CrammerSingerState,
  type Dichotomy,
  type NestedDichotomyModel,
  type ReductionModel,
} from './multiclass'
