import type { Specimen } from '@lab/specimen'
import { SubgroupDiscoveryFigure } from './_subgroups/discovery'
import { ExceptionalModelFigure } from './_subgroups/emm'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/supervised-learning/subgroup-discovery',
    title: 'Subgroup discovery: finding where the target is unusual',
    description:
      'Steer a refinement search over conjunctions of selectors (beam, best-first, depth-first, breadth-first; branch and bound by optimistic estimates) on a table with planted subgroups or the Titanic: play it from step 0, pin a subgroup from the ranked list to see its cover and target, and add or remove selectors by hand.',
    tags: [
      'subgroup discovery',
      'WRAcc',
      'binomial test',
      'lift',
      'χ²',
      'mean shift',
      'optimistic estimate',
      'branch and bound',
      'beam search',
      'refinementSearchSteps',
      'subgroupDiscoverySteps',
    ],
    render: () => <SubgroupDiscoveryFigure />,
  },
  {
    module: 'part-5-learning-paradigms/supervised-learning/subgroup-discovery',
    title: 'Exceptional model mining',
    description:
      'Subgroups whose fitted model differs from the rest: a flipped correlation, a different regression slope (t test or Cook’s distance), a reversed logistic classifier, a reversed association. Pin a subgroup to see its fit against its complement’s, and refine it by hand.',
    tags: [
      'exceptional model mining',
      'correlation',
      'regression slope',
      'Cook’s distance',
      'logistic regression',
      'Yule’s Q',
      'correlationModel',
      'regressionModel',
      'logisticModel',
      'associationModel',
    ],
    render: () => <ExceptionalModelFigure />,
  },
]
