import type { Specimen } from '../../specimen'
import { FoilShowcase } from './_induction/figures'

export const specimens: Specimen[] = [
  {
    module: 'logic/induction',
    title: 'Learning rules from examples: FOIL',
    description:
      "Quinlan's FOIL learns Prolog clauses for daughter, grandparent and the recursive ancestor relation from a family tree: click examples positive or negative, step the clause growing literal by literal with every candidate's FOIL gain and the examples covered, then run the learned program on the Prolog engine.",
    tags: [
      'showcase',
      'foilSteps',
      'foilProblem',
      'foilRefinements',
      'foilGain',
      'solveQuery',
      'inductive logic programming',
      'sequential covering',
      'rule learning',
      'GraphView',
    ],
    render: () => <FoilShowcase />,
  },
]
