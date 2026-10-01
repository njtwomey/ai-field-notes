/**
 * `aifn-applied/inference/sequence-models`: sequence models: the hidden Markov model (`Hmm`, `hmm`, its chain of
 * potentials `hmmChain` for the generic engines of `aifn/inference/exact`, the occasionally dishonest casino,
 * `sampleHmm`, `hmmModel` in the model language), and the linear-chain
 * CRF with its structure (`crfStructure`) and factor graph (`crfFactorGraph`), both chain-shaped.
 */

export {
  crfFactorGraph,
  crfGradient,
  crfLogLikelihood,
  crfMarginals,
  crfPotentials,
  crfScore,
  crfStructure,
  crfViterbi,
  linearChainCrf,
  type CrfGradient,
  type LinearChainCrf,
} from './crf'
export { dishonestCasino, hmm, hmmChain, hmmModel, sampleHmm, type Hmm } from './hmm'
export { sequenceModelFunctions } from './registry'
