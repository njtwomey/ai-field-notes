import type { Specimen } from '../specimen'
import { BackfittingSweeps, EbmShapes, PartialEffects } from './_gam/figures'

export const specimens: Specimen[] = [
  {
    module: 'gam',
    title: 'A GAM with partial effects and bands',
    description:
      'gam with three P-spline smooths, λ by REML or GCV: partial effects, Bayesian ± 2 se bands, partial residuals and posterior draws, EDF per term.',
    tags: ['gam', 'REML', 'GCV', 'partial effects', 'posterior draws'],
    render: () => <PartialEffects />,
  },
  {
    module: 'gam',
    title: 'Backfitting sweep by sweep',
    description: 'backfitting as a traceable algorithm: each curve after each sweep, and the residual sum of squares.',
    tags: ['backfitting', 'trace', 'Player'],
    render: () => <BackfittingSweeps />,
  },
  {
    module: 'gam',
    title: 'Explainable boosting machine shapes',
    description:
      'explainableBoostingMachine: step-function shapes from cyclic boosting of one-split trees, by number of rounds.',
    tags: ['explainableBoostingMachine', 'boosting', 'shapes'],
    render: () => <EbmShapes />,
  },
]
