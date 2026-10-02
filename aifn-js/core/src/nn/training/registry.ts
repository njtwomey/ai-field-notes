/**
 * The algorithms of `aifn/nn/training`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as adversarial from './adversarial'
import * as energy from './energy'
import * as fullBatch from './fullBatch'
import * as train from './train'

const algorithm = definer<AlgorithmInfo>('algorithm', 'nn/training')

algorithm(
  {
    key: 'trainingLoop',
    name: 'Training loop',
    summary: 'Minibatch training of a network: forward, loss, gradients and an optimiser update per step.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', grad: 'grads', flags: ['diverged'] },
    random: true,
    notes: ['backpropagation'],
  },
  train.trainingLoop,
)

algorithm(
  {
    key: 'adversarialTraining',
    name: 'Adversarial training',
    summary: 'Alternating gradient steps on a critic and a generator, k critic steps per generator step.',
    problem: 'network',
    state: { iterate: 'generator', objective: 'generatorLoss', flags: ['diverged'] },
    random: true,
    notes: ['generative-adversarial-network'],
    cite: ['goodfellow2014', 'gulrajani2017'],
  },
  adversarial.adversarialTraining,
)

algorithm(
  {
    key: 'contrastiveDivergence',
    name: 'Persistent contrastive divergence',
    summary:
      'Energy-based-model training with Langevin negatives from a replay buffer, plus an optional supervised term.',
    problem: 'network',
    state: { iterate: 'params', objective: 'loss', flags: ['diverged'] },
    random: true,
    notes: ['energy-based-models', 'joint-energy-models'],
    cite: ['tieleman2008', 'du2019', 'grathwohl2019'],
  },
  energy.contrastiveDivergence,
)

algorithm(
  {
    key: 'fullBatchTraining',
    name: 'Full-batch training (L-BFGS)',
    summary:
      'A network trained on the whole set by a vector method (L-BFGS by default): the parameter tree raveled to θ, the loss and its gradient on all the data.',
    problem: 'network',
    state: { iterate: 'params', objective: 'value', stepSize: 'stepSize', flags: ['converged', 'diverged', 'stalled'] },
    notes: ['quasi-newton-methods', 'line-search', 'multilayer-perceptron'],
    cite: ['liu1989', 'nocedal2006'],
  },
  fullBatch.fullBatchTraining,
)

/** Every algorithm of the module, keyed by factory name. */
export const trainingAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', train, adversarial, energy, fullBatch) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
