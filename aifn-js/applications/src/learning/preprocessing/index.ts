/**
 * `aifn-applied/learning/preprocessing`: preprocessing transforms fitted with `fit({ x, y? })`: scalers, encoders,
 * imputers, power transforms, feature maps (polynomial, spline, random Fourier) and whitening.
 */

export { checkColumns, fitTransform, type FittedTransform, type Invertible, type Transformer } from './transformer'
export {
  maxAbsScaler,
  minMaxScaler,
  robustScaler,
  standardScaler,
  type AffineScaler,
  type MaxAbsScaler,
  type MinMaxScaler,
  type RobustScaler,
  type StandardScaler,
} from './scaling'
export {
  oneHotEncoder,
  ordinalEncoder,
  targetEncodeCrossFit,
  targetEncoder,
  type CategoricalInput,
  type Category,
  type OneHotEncoder,
  type OrdinalEncoder,
  type TargetEncoder,
} from './encoding'
export { simpleImputer, type SimpleImputer } from './impute'
export {
  polynomialFeatures,
  randomFourierFeatures,
  splineFeatures,
  type PolynomialFeatures,
  type RandomFourierFeatures,
  type SplineFeatures,
} from './features'
export { whitening, type Whitening } from './whitening'
export {
  boxCox,
  boxCoxInverse,
  boxCoxLambda,
  powerTransform,
  yeoJohnson,
  yeoJohnsonInverse,
  yeoJohnsonLambda,
  type PowerLambda,
  type PowerTransform,
} from './power'
export { preprocessingFunctions } from './registry'
