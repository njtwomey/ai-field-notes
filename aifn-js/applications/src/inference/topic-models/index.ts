/**
 * `aifn-applied/inference/topic-models`: latent Dirichlet allocation: the model, its match, collapsed Gibbs sampling
 * and estimates, and its engine (`ldaEngine`, and `ldaEngines` = the built-in table extended with it, for `infer`).
 */
export {
  ldaCollapsedGibbs,
  ldaEngine,
  ldaEngines,
  ldaEstimates,
  ldaModel,
  ldaOptions,
  matchLda,
  type LdaOptions,
  type LdaState,
} from './lda'
