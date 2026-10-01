/**
 * The algorithms of `aifn/inference/stochastic`, registered with what each factory takes (`problem`) and the roles of its
 * state's fields (`state`: iterate, objective, grad, stepSize, and the `Status` flags it sets), so a generic trace
 * view picks default series and a worker can address an algorithm by key (design S §2.3).
 */

import { definer, entries, type AlgorithmInfo, type Entry } from 'aifn/foundation/registry'
import * as factorGibbs from './factorGibbs'
import * as gibbs from './gibbs'
import * as hamiltonian from './hamiltonian'
import * as langevin from './langevin'
import * as metropolis from './metropolis'
import * as smc from './smc'

const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/stochastic')

algorithm(
  {
    key: 'metropolisHastings',
    name: 'Metropolis–Hastings',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['metropolis-hastings', 'markov-chain-monte-carlo'],
    cite: ['metropolis1953', 'hastings1970'],
  },
  metropolis.metropolisHastings,
)
algorithm(
  {
    key: 'randomWalkMetropolis',
    name: 'Random-walk Metropolis',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['metropolis-hastings', 'markov-chain-monte-carlo'],
    cite: ['metropolis1953'],
  },
  metropolis.randomWalkMetropolis,
)
algorithm(
  {
    key: 'independenceMetropolis',
    name: 'Independence Metropolis',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['metropolis-hastings'],
    cite: ['tierney1994'],
  },
  metropolis.independenceMetropolis,
)
algorithm(
  {
    key: 'gibbs',
    name: 'Gibbs sampling',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['gibbs-sampling', 'markov-chain-monte-carlo'],
    cite: ['geman1984'],
  },
  gibbs.gibbs,
)
algorithm(
  {
    key: 'sliceSampler',
    name: 'Slice sampling',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', flags: ['diverged'] },
    random: true,
    notes: ['markov-chain-monte-carlo'],
  },
  gibbs.sliceSampler,
)
algorithm(
  {
    key: 'hmc',
    name: 'Hamiltonian Monte Carlo',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    glossary: 'hmc',
    notes: ['hamiltonian-monte-carlo', 'markov-chain-monte-carlo'],
    cite: ['neal2011'],
  },
  hamiltonian.hmc,
)
algorithm(
  {
    key: 'nuts',
    name: 'No-U-Turn sampler',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    glossary: 'nuts',
    notes: ['hamiltonian-monte-carlo'],
    cite: ['hoffman2014', 'betancourt2017'],
  },
  hamiltonian.nuts,
)
algorithm(
  {
    key: 'unadjustedLangevin',
    name: 'Unadjusted Langevin',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['langevin-dynamics'],
    cite: ['roberts1996'],
  },
  langevin.unadjustedLangevin,
)
algorithm(
  {
    key: 'mala',
    name: 'Metropolis-adjusted Langevin',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'grad', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['langevin-dynamics'],
    cite: ['roberts1996'],
  },
  langevin.mala,
)
algorithm(
  {
    key: 'sgld',
    name: 'Stochastic gradient Langevin dynamics',
    problem: 'log-density',
    state: { iterate: 'x', objective: 'logDensity', grad: 'gradEstimate', stepSize: 'stepSize', flags: ['diverged'] },
    random: true,
    notes: ['stochastic-gradient-langevin-dynamics'],
    cite: ['welling2011'],
  },
  langevin.sgld,
)
algorithm(
  {
    key: 'particleFilter',
    name: 'Particle filter',
    problem: 'sequence',
    state: { iterate: 'mean', objective: 'logEvidence', flags: ['diverged'] },
    random: true,
    notes: ['particle-filter'],
    cite: ['gordon1993'],
  },
  smc.particleFilter,
)
algorithm(
  {
    key: 'temperedSmc',
    name: 'Tempered sequential Monte Carlo',
    problem: 'log-density',
    state: { iterate: 'particles', objective: 'logEvidence', flags: ['diverged'] },
    random: true,
  },
  smc.temperedSmc,
)
algorithm(
  {
    key: 'factorGraphGibbs',
    name: 'Gibbs sampling on a factor graph',
    problem: 'factor-graph',
    state: { iterate: 'assignment', flags: [] },
    random: true,
    notes: ['gibbs-sampling', 'factor-graph'],
    cite: ['geman1984'],
  },
  factorGibbs.factorGraphGibbs,
)
algorithm(
  {
    key: 'modelGibbs',
    name: 'Gibbs sampling on a model',
    problem: 'factor-graph',
    state: { iterate: 'values', objective: 'logJoint', flags: [] },
    random: true,
    notes: ['gibbs-sampling'],
    cite: ['geman1984'],
  },
  factorGibbs.modelGibbs,
)

/** Every algorithm of the module, keyed by factory name. */
export const stochasticAlgorithms: Readonly<Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>> =
  entries<AlgorithmInfo>('algorithm', factorGibbs, gibbs, hamiltonian, langevin, metropolis, smc) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, AlgorithmInfo>>
  >
