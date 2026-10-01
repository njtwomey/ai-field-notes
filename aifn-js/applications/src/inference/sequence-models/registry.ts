/**
 * The registry of `aifn-applied/inference/sequence-models`.
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as crf from './crf'
import * as hmm from './hmm'

const fn = definer<FunctionInfo>('function', 'inference/sequence-models')
const HMM = ['hidden-markov-model']
const CRF = ['conditional-random-field']

fn({ key: 'hmm', name: 'Hidden Markov model', role: 'construction', notes: HMM, cite: ['rabiner1989'] }, hmm.hmm)
fn({ key: 'hmmChain', name: 'HMM chain potentials', role: 'construction', notes: HMM }, hmm.hmmChain)
fn({ key: 'hmmModel', name: 'HMM model specification', role: 'construction', notes: HMM }, hmm.hmmModel)
fn(
  {
    key: 'dishonestCasino',
    name: 'Occasionally dishonest casino',
    role: 'construction',
    notes: ['occasionally-dishonest-casino', ...HMM],
    cite: ['durbin1998'],
  },
  hmm.dishonestCasino,
)
fn({ key: 'sampleHmm', name: 'Sample an HMM', role: 'simulation', random: true, notes: HMM }, hmm.sampleHmm)
fn({ key: 'crfStructure', name: 'Linear-chain CRF structure', role: 'construction', notes: CRF }, crf.crfStructure)
fn(
  {
    key: 'linearChainCrf',
    name: 'Linear-chain CRF',
    role: 'construction',
    notes: CRF,
    cite: ['lafferty2001', 'sutton2012'],
  },
  crf.linearChainCrf,
)
fn({ key: 'crfPotentials', name: 'CRF log-potentials', role: 'construction', notes: CRF }, crf.crfPotentials)
fn(
  { key: 'crfFactorGraph', name: 'CRF factor graph', role: 'construction', notes: [...CRF, 'factor-graph'] },
  crf.crfFactorGraph,
)
fn({ key: 'crfMarginals', name: 'CRF marginals (forward–backward)', role: 'inference', notes: CRF }, crf.crfMarginals)
fn({ key: 'crfViterbi', name: 'CRF decoding (Viterbi)', role: 'inference', notes: CRF }, crf.crfViterbi)
fn({ key: 'crfScore', name: 'CRF score of a labelling', role: 'estimator', notes: CRF }, crf.crfScore)
fn(
  {
    key: 'crfLogLikelihood',
    name: 'CRF conditional log-likelihood',
    role: 'estimator',
    notes: CRF,
    cite: ['sutton2012'],
  },
  crf.crfLogLikelihood,
)
fn(
  {
    key: 'crfGradient',
    name: 'CRF log-likelihood gradient',
    summary: 'Observed minus expected feature counts.',
    role: 'estimator',
    notes: CRF,
    cite: ['sutton2012'],
  },
  crf.crfGradient,
)

/** The functions of the module. */
export const sequenceModelFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', hmm, crf) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
