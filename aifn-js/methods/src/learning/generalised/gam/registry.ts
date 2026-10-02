/**
 * The registry of `aifn-applied/learning/generalised/gam` besides its fitters (registered in `fitters.ts`): the EBM
 * boosting algorithm, term constructors, the problem and smoothing-path functions, and the expectile set-ups.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as ebm from './ebm'
import * as expectile from './expectile'
import * as expectileTraining from './expectile-training'
import * as fitters from './fitters'
import * as model from './model'
import * as problem from './problem'
import * as terms from './terms'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const fn = definer<FunctionInfo>('function', 'learning/generalised/gam')
const GAM = ['generalised-additive-model']

definer<AlgorithmInfo>('algorithm', 'learning/generalised/gam')(
  {
    key: 'ebmBoosting',
    name: 'Explainable boosting machine',
    summary: 'Cyclic gradient boosting of one binned shape function per feature, with a small learning rate.',
    problem: 'objective',
    state: { iterate: 'shapes', objective: 'loss', flags: ['diverged'] },
    notes: ['explainable-boosting-machines', 'interpreting-generalised-additive-models'],
    cite: ['lou2012', 'nori2019'],
  },
  ebm.ebmBoosting,
)
fn({ key: 's', name: 'Smooth term', role: 'construction', notes: [...GAM, 'p-splines'], cite: ['wood2017'] }, terms.s)
fn(
  {
    key: 'te',
    name: 'Tensor-product smooth',
    role: 'construction',
    notes: ['tensor-product-smooths'],
    cite: ['wood2017'],
  },
  terms.te,
)
fn({ key: 'cyclic', name: 'Cyclic smooth', role: 'construction', notes: ['cyclic-factor-and-by-terms'] }, terms.cyclic)
fn(
  {
    key: 'thinPlate',
    name: 'Thin-plate smooth',
    role: 'construction',
    notes: ['thin-plate-regression-splines'],
    cite: ['wood2003'],
  },
  terms.thinPlate,
)
fn(
  { key: 'factorTerm', name: 'Factor term', role: 'construction', notes: ['cyclic-factor-and-by-terms'] },
  terms.factorTerm,
)
fn({ key: 'linearTerm', name: 'Linear term', role: 'construction', notes: GAM }, terms.linearTerm)
fn(
  {
    key: 'termBasis',
    name: 'Term basis and penalties',
    role: 'construction',
    notes: ['smoothing-penalties-and-curvature'],
  },
  terms.termBasis,
)
fn({ key: 'gamDesign', name: 'GAM design matrix', role: 'construction', notes: GAM }, problem.gamDesign)
fn({ key: 'gamProblem', name: 'GAM problem', role: 'construction', notes: GAM }, problem.gamProblem)
fn(
  { key: 'resolveLikelihood', name: 'Resolve a GAM likelihood', role: 'construction', notes: GAM },
  problem.resolveLikelihood,
)
fn(
  {
    key: 'smoothingPath',
    name: 'Smoothing-parameter path',
    role: 'estimator',
    notes: [
      'additive-model-smoothing-parameter-selection',
      'additive-model-inference-and-effective-degrees-of-freedom',
    ],
    cite: ['wood2011'],
  },
  problem.smoothingPath,
)
fn(
  {
    key: 'smoothingProfile',
    name: 'Smoothing criterion profile',
    role: 'estimator',
    notes: ['additive-model-smoothing-parameter-selection'],
  },
  problem.smoothingProfile,
)
fn({ key: 'gamFitter', name: 'GAM fitter by method', role: 'construction', notes: GAM }, fitters.gamFitter)
fn(
  {
    key: 'gamProblemChoices',
    name: 'GAM problem choices',
    role: 'construction',
    notes: ['generalised-additive-models-in-practice'],
  },
  fitters.gamProblemChoices,
)
fn(
  {
    key: 'gamTrainingRun',
    name: 'GAM training run',
    role: 'construction',
    notes: ['generalised-additive-models-in-practice'],
  },
  fitters.gamTrainingRun,
)
fn({ key: 'gamModel', name: 'GAM model', role: 'construction', notes: GAM }, model.gamModel)
fn(
  {
    key: 'gamLinkBand',
    name: 'GAM confidence band on the link scale',
    role: 'inference',
    notes: ['additive-model-inference-and-effective-degrees-of-freedom'],
  },
  model.gamLinkBand,
)
fn(
  {
    key: 'expectileFan',
    name: 'Expectile fan',
    role: 'fit',
    notes: ['expectile-generalised-additive-models'],
    cite: ['newey1987'],
  },
  expectile.expectileFan,
)
fn(
  {
    key: 'expectileLaws',
    name: 'Expectile laws',
    role: 'construction',
    notes: ['expectile-generalised-additive-models'],
  },
  expectile.expectileLaws,
)
fn(
  {
    key: 'expectileProblem',
    name: 'Expectile GAM problem',
    role: 'construction',
    notes: ['expectile-generalised-additive-models'],
  },
  expectileTraining.expectileProblem,
)
fn(
  {
    key: 'expectileTrainingRun',
    name: 'Expectile GAM training run',
    role: 'construction',
    notes: ['expectile-generalised-additive-models'],
  },
  expectileTraining.expectileTrainingRun,
)

/** The EBM algorithm, keyed by factory name. */
export const gamAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>('algorithm', ebm) as Table<AlgorithmInfo>
/** The functions of the module, keyed by name. */
export const gamFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  terms,
  problem,
  fitters,
  model,
  expectile,
  expectileTraining,
) as Table<FunctionInfo>
