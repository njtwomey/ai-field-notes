import type { Specimen } from '@lab/specimen'
import { BoostingSpecimen } from '../_shared/trees'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/supervised-learning/tree-based-methods',
    title: 'Boosting rounds',
    description:
      'adaBoost (SAMME) and gradientBoosting, stage by stage: the score, the boundary and the sample weights.',
    tags: ['adaBoost', 'gradientBoosting', 'ensembles'],
    render: () => <BoostingSpecimen />,
  },
]
