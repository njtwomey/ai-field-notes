import type { Specimen } from '../../../../specimen'
import { BoostingSpecimen } from '../_shared/trees'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/trees-and-ensembles/boosting',
    title: 'Boosting rounds',
    description:
      'adaBoost (SAMME) and gradientBoosting, stage by stage: the score, the boundary and the sample weights.',
    tags: ['adaBoost', 'gradientBoosting', 'ensembles'],
    render: () => <BoostingSpecimen />,
  },
]
