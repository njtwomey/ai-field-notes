/**
 * The functions of `aifn-applied/algorithms/dynamic-programming`: worked dynamic programs (each `…Program` is the
 * problem for `aifn/optim/programming`'s `dp` algorithm; the plain function solves it).
 */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as problems from './problems'

const fn = definer<FunctionInfo>('function', 'algorithms/dynamic-programming')

fn(
  { key: 'knapsackProgram', name: '0/1 knapsack as a dynamic program', role: 'construction', cite: ['bellman1957'] },
  problems.knapsackProgram,
)
fn({ key: 'knapsack', name: '0/1 knapsack', role: 'solver' }, problems.knapsack)
fn(
  { key: 'unboundedKnapsackProgram', name: 'Unbounded knapsack as a dynamic program', role: 'construction' },
  problems.unboundedKnapsackProgram,
)
fn({ key: 'unboundedKnapsack', name: 'Unbounded knapsack', role: 'solver' }, problems.unboundedKnapsack)
fn(
  { key: 'lcsProgram', name: 'Longest common subsequence as a dynamic program', role: 'construction' },
  problems.lcsProgram,
)
fn({ key: 'lcs', name: 'Longest common subsequence', role: 'solver', notes: ['rouge'] }, problems.lcs)
fn(
  {
    key: 'editDistanceProgram',
    name: 'Edit distance as a dynamic program',
    role: 'construction',
    notes: ['word-and-character-error-rates'],
  },
  problems.editDistanceProgram,
)
fn(
  { key: 'editDistance', name: 'Levenshtein edit distance', role: 'solver', notes: ['word-and-character-error-rates'] },
  problems.editDistance,
)
fn(
  { key: 'alignmentProgram', name: 'Sequence alignment as a dynamic program', role: 'construction' },
  problems.alignmentProgram,
)
fn({ key: 'needlemanWunsch', name: 'Needleman–Wunsch global alignment', role: 'solver' }, problems.needlemanWunsch)
fn({ key: 'smithWaterman', name: 'Smith–Waterman local alignment', role: 'solver' }, problems.smithWaterman)

/** The functions of the module, keyed by name. */
export const dynamicProgrammingFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', problems) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
