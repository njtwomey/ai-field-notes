import type { LffBias, LffProblem } from 'aifn-compute/logic/induction'
import { PARENT_TEXT } from '../_shared/family'

/** A learning-from-failures task: the problem, the bounds of the space and a label. */
export interface LffTask {
  readonly label: string
  readonly problem: LffProblem
  readonly bias: LffBias
}

/** last(L, X): X is the last element of the list L, from head, tail and empty (Cropper and Morel's list tasks). */
export const LAST_TASK: LffTask = {
  label: 'last element of a list',
  problem: {
    background: 'head([H|_], H). tail([_|T], T). empty([]).',
    target: { name: 'last', arity: 2, types: ['list', 'element'] },
    body: [
      { name: 'head', arity: 2, types: ['list', 'element'] },
      { name: 'tail', arity: 2, types: ['list', 'list'] },
      { name: 'empty', arity: 1, types: ['list'] },
    ],
    positives: ['last([a, b, c], c)', 'last([b, a], a)', 'last([c], c)', 'last([a, c, b, a], a)'],
    negatives: ['last([a, b, c], a)', 'last([a, b, c], b)', 'last([b, a], b)', 'last([a, c, b, a], c)'],
  },
  bias: { maxVars: 3, maxBody: 3, maxClauses: 2 },
}

/** ancestor(X, Y) from the family's parent facts, with the examples of the meta-interpretive learning note. */
export const ANCESTOR_TASK: LffTask = {
  label: 'ancestor in the family',
  problem: {
    background: PARENT_TEXT,
    target: { name: 'ancestor', arity: 2 },
    body: [{ name: 'parent', arity: 2 }],
    positives: ['ancestor(ann, bob)', 'ancestor(ann, dee)', 'ancestor(ann, gus)', 'ancestor(cid, fay)'],
    negatives: ['ancestor(bob, ann)', 'ancestor(dee, bob)', 'ancestor(cid, gus)'],
  },
  bias: { maxVars: 3, maxBody: 2, maxClauses: 2 },
}
