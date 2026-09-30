/**
 * `aifn/optim/proximal`: proximal and projected methods: ISTA, FISTA, projected gradient, proximal operators (L1, L2,
 * squared L2, box, non-negative) and projections (ball, box, simplex, non-negative orthant).
 */

export {
  fista,
  ista,
  projectBall,
  projectBox,
  projectNonnegative,
  projectSimplex,
  projectedGradient,
  proximalGradient,
  proxBox,
  proxL1,
  proxL2,
  proxNonnegative,
  proxSquaredL2,
  proxZero,
  type Bound,
  type Prox,
  type ProximalGradientOptions,
  type ProximalGradientState,
} from './proximal'
