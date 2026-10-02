import type { Specimen } from '../../specimen'
import { PrologShowcase } from './_resolution/figures'

export const specimens: Specimen[] = [
  {
    module: 'logic/resolution',
    title: 'Showcase: Prolog, logic as a program',
    description:
      'A small Prolog in aifn: edit a program (family tree, append and reverse, cut and negation, map colouring), ask a query, and play the SLD search tree from the query, node by node, with the goal, clause, unifier and bindings at each node, successes, failures and branches pruned by cut.',
    tags: [
      'showcase',
      'sldSteps',
      'sldTree',
      'prologProgram',
      'unify',
      'parseProgram',
      'Prolog',
      'logic programming',
      'unification',
      'backtracking',
      'TreeView',
      'CodeEditor',
    ],
    render: () => <PrologShowcase />,
  },
]
