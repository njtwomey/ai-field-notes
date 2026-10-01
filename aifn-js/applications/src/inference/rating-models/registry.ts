/**
 * The registry of `aifn-applied/inference/rating-models`.
 */

import { definer, entries, type AlgorithmInfo, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as examples from './examples'

type Table<I extends AlgorithmInfo | FunctionInfo> = Readonly<Record<string, Entry<(...args: never[]) => unknown, I>>>
const algorithm = definer<AlgorithmInfo>('algorithm', 'inference/rating-models')
const fn = definer<FunctionInfo>('function', 'inference/rating-models')
const notes = ['trueskill', 'skill-rating', 'assessing-skills']

algorithm(
  {
    key: 'trueSkillEp',
    name: 'TrueSkill by expectation propagation',
    summary: 'Skills from a sequence of matches, by EP over the match factor graphs.',
    problem: 'factor-graph',
    state: { iterate: 'means', flags: ['converged'] },
    notes: [...notes, 'expectation-propagation'],
    cite: ['herbrich2006'],
  },
  examples.trueSkillEp,
)
fn(
  {
    key: 'trueSkillUpdate',
    name: 'TrueSkill update',
    summary: 'The closed-form two-player update of skill means and variances after one match.',
    role: 'inference',
    notes,
    cite: ['herbrich2006'],
  },
  examples.trueSkillUpdate,
)
fn(
  { key: 'trueSkillModel', name: 'TrueSkill model', role: 'construction', notes, cite: ['herbrich2006'] },
  examples.trueSkillModel,
)
fn({ key: 'drawMargin', name: 'TrueSkill draw margin', role: 'property', notes: ['trueskill'] }, examples.drawMargin)

/** The algorithms of the module. */
export const ratingModelAlgorithms: Table<AlgorithmInfo> = entries<AlgorithmInfo>(
  'algorithm',
  examples,
) as Table<AlgorithmInfo>
/** The functions of the module. */
export const ratingModelFunctions: Table<FunctionInfo> = entries<FunctionInfo>(
  'function',
  examples,
) as Table<FunctionInfo>
