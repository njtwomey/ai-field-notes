/**
 * `aifn/preprocess`: transforms fitted with `fit({ x, y? }, options)` that return plain fitted objects with
 * `transform` (and `inverse` where it exists) and their fitted state as public fields.
 *
 * - Scaling: `standardScaler`, `minMaxScaler`, `robustScaler`, `maxAbsScaler`.
 * - Encoding: `oneHotEncoder`, `ordinalEncoder`, `targetEncoder` (with smoothing) and `targetEncodeCrossFit`.
 * - Imputation: `simpleImputer`.
 * - Features: `polynomialFeatures`, `splineFeatures`, `randomFourierFeatures` (drawn from the fit's stream).
 * - Whitening: `whitening` (PCA or ZCA).
 * - Power transforms: `boxCox`, `yeoJohnson` and their inverses, `boxCoxLambda` and `yeoJohnsonLambda` (λ by maximum
 *   likelihood), and `powerTransform`.
 *
 * Numeric transforms take matrices [n, d]; encoders take label lists or numeric tensors of codes.
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
