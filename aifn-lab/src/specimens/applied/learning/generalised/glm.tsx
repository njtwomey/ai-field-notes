import type { Specimen } from '../../../../specimen'
import { IrlsSteps } from './_glm/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/generalised/glm',
    title: 'GLM families with IRLS steps',
    description:
      'glm for Gaussian, Bernoulli, Poisson, gamma, inverse Gaussian and negative binomial families with their links, stepped through the IRLS trace.',
    tags: ['glm', 'irls', 'families', 'links', 'deviance', 'Player'],
    render: () => <IrlsSteps />,
  },
]
