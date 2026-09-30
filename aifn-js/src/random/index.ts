/**
 * `aifn/random`: keyed, counter-based random streams (Philox4x32-10) and samplers that take the stream first.
 *
 * ```ts
 * const s = stream(7)
 * const chain = s.child('chain', 3) // key "7/chain:3"; the same draws however much `s` has drawn
 * const x = normal(chain, 0, 1)
 * const runs = replicate(20, s.child('runs'), (r) => simulate(r))
 * ```
 *
 * Samplers take number or tensor parameters with broadcasting and an optional `{ shape }`; numbers in give a number
 * out, otherwise a tensor whose elements are drawn in row-major order. They are not differentiable primitives (see
 * samplers.ts).
 */

export { stream, type Stream } from './stream'
export { philox4x32 } from './philox'
export {
  aliasSample,
  aliasTable,
  bernoulli,
  beta,
  binomial,
  categorical,
  chiSquare,
  choice,
  dirichlet,
  exponential,
  gamma,
  logGammaVariate,
  multinomial,
  multivariateNormal,
  normal,
  normals,
  permutation,
  poisson,
  shuffle,
  studentT,
  uniform,
  type AliasTable,
  type ChoiceOptions,
  type Drawn,
  type Param,
  type SampleOptions,
  type Spread,
} from './samplers'
export { replicate, type ReplicateCache } from './replicate'
