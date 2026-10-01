/**
 * The algorithms of `aifn/optim/first-order`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as conjugateGradient from './conjugateGradient'
import * as coordinateDescent from './coordinateDescent'
import * as firstOrder from './firstOrder'

const algorithm = definer<AlgorithmInfo>('algorithm', 'optim/first-order')

algorithm(
  {
    key: 'gradientDescent',
    name: 'Gradient descent',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['gradient-descent', 'convergence-of-gradient-descent'],
  },
  firstOrder.gradientDescent,
)
algorithm(
  {
    key: 'momentum',
    name: 'Momentum',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['momentum-and-nesterov'],
    cite: ['polyak1964'],
  },
  firstOrder.momentum,
)
algorithm(
  {
    key: 'nesterov',
    name: 'Nesterov accelerated gradient',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['momentum-and-nesterov', 'accelerated-gradient-methods'],
    cite: ['nesterov1983'],
  },
  firstOrder.nesterov,
)
algorithm(
  {
    key: 'adagrad',
    name: 'AdaGrad',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['adagrad-and-rmsprop'],
    cite: ['duchi2011'],
  },
  firstOrder.adagrad,
)
algorithm(
  {
    key: 'rmsprop',
    name: 'RMSProp',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    notes: ['adagrad-and-rmsprop'],
    cite: ['tieleman2012'],
  },
  firstOrder.rmsprop,
)
algorithm(
  {
    key: 'adam',
    name: 'Adam',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    glossary: 'adam',
    notes: ['adam'],
    cite: ['kingma2014'],
  },
  firstOrder.adam,
)
algorithm(
  {
    key: 'adamw',
    name: 'AdamW',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    glossary: 'adamw',
    notes: ['decoupled-weight-decay', 'adam'],
    cite: ['loshchilov2019'],
  },
  firstOrder.adamw,
)
algorithm(
  {
    key: 'conjugateGradient',
    name: 'Nonlinear conjugate gradient',
    problem: 'objective',
    state: {
      iterate: 'x',
      objective: 'value',
      grad: 'grad',
      stepSize: 'stepSize',
      flags: ['converged', 'diverged', 'stalled'],
    },
    cite: ['nocedal2006'],
  },
  conjugateGradient.conjugateGradient,
)
algorithm(
  {
    key: 'linearConjugateGradient',
    name: 'Linear conjugate gradient',
    summary: 'Conjugate gradient for Ax = b with A symmetric positive definite.',
    problem: 'quadratic-program',
    state: { iterate: 'x', objective: 'residualNorm', stepSize: 'alpha', flags: ['converged', 'diverged', 'stalled'] },
    cite: ['hestenes1952'],
  },
  conjugateGradient.linearConjugateGradient,
)
algorithm(
  {
    key: 'coordinateDescent',
    name: 'Coordinate descent',
    problem: 'objective',
    state: { iterate: 'x', objective: 'value', grad: 'grad', flags: ['converged', 'diverged'] },
    notes: ['coordinate-descent'],
    cite: ['wright2015'],
  },
  coordinateDescent.coordinateDescent,
)

/** Every algorithm of the module, keyed by factory name. */
export const firstOrderAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', conjugateGradient, coordinateDescent, firstOrder) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
