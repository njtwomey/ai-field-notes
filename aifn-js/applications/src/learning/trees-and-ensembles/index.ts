/**
 * `aifn-applied/learning/trees-and-ensembles`: decision trees and their ensembles. The shared layer holds CART
 * (growth, split search, cost-complexity pruning, importances) and the `decisionTree` and `regressionTree` estimators;
 * children: bagging, boosting.
 */

export {
  applyTree,
  costComplexityPath,
  decisionPath,
  decisionTree,
  featureImportances,
  growTree,
  nodeLabel,
  predictTree,
  pruneTree,
  regressionTree,
  splitSearch,
  treeGrowthSteps,
  treeSize,
  type Criterion,
  type DecisionNode,
  type DecisionTree,
  type DecisionTreeModel,
  type FeatureSplits,
  type RegressionTreeModel,
  type SplitData,
  type SplitSearch,
  type TreeGrowthState,
  type TreeParams,
  type TreeProblem,
  type WeightedData,
} from './tree'
export { randomForest } from './bagging'
export { adaBoost, gradientBoosting } from './boosting'
