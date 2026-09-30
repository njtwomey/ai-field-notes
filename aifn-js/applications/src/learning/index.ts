/**
 * `aifn-applied/learning`: supervised models over the estimator protocol of `aifn/learning/estimators`. Groups:
 * generalised (glm, gam), trees-and-ensembles (bagging, boosting); modules: linear, generative-classifiers,
 * kernel-methods, gaussian-processes, neighbours, reductions, preprocessing.
 */

export { glm } from './generalised/glm'
export { gam } from './generalised/gam'
export { linearRegression, perceptron } from './linear'
export { gaussianNaiveBayes, linearDiscriminant, quadraticDiscriminant } from './generative-classifiers'
export { supportVectorMachine } from './kernel-methods'
export { gaussianProcessRegressor, gpClassifier } from './gaussian-processes'
export { randomForest } from './trees-and-ensembles/bagging'
export { adaBoost, gradientBoosting } from './trees-and-ensembles/boosting'
export { kNearestNeighbours } from './neighbours'
export { oneVersusRest } from './reductions'
export { standardScaler } from './preprocessing'
