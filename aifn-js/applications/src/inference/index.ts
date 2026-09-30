/**
 * `aifn-applied/inference`: named probabilistic models on the engines of `aifn/inference`: sequence models (HMM
 * problems, the linear-chain CRF), topic models (LDA), rating models (TrueSkill), lattice models (Ising), mixture
 * models, conjugate models and classifier models (the Bayes point machine).
 */

export { dishonestCasino, linearChainCrf } from './sequence-models'
export { ldaModel } from './topic-models'
export { trueSkillUpdate } from './rating-models'
