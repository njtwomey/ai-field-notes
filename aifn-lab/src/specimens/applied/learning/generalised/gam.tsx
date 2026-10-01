import type { Specimen } from '../../../../specimen'
import { BackfittingSweeps, EbmShapes, PartialEffects } from './_gam/figures'
import { GamShowcase } from './_gam/showcase-gam'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/generalised/gam',
    title: 'A GAM with partial effects and bands',
    description:
      'gam with three P-spline smooths, λ by REML or GCV: partial effects, Bayesian ± 2 se bands, partial residuals and posterior draws, EDF per term.',
    tags: ['gam', 'REML', 'GCV', 'partial effects', 'posterior draws'],
    render: () => <PartialEffects />,
  },
  {
    module: 'applied/learning/generalised/gam',
    title: 'Backfitting sweep by sweep',
    description: 'backfitting as a traceable algorithm: each curve after each sweep, and the residual sum of squares.',
    tags: ['backfitting', 'trace', 'Player'],
    render: () => <BackfittingSweeps />,
  },
  {
    module: 'applied/learning/generalised/gam',
    title: 'Explainable boosting machine shapes',
    description:
      'explainableBoostingMachine: step-function shapes from cyclic boosting of one-split trees, by number of rounds.',
    tags: ['explainableBoostingMachine', 'boosting', 'shapes'],
    render: () => <EbmShapes />,
  },
  {
    module: 'applied/learning/generalised/gam',
    title: 'Showcase: interactive GAM',
    description:
      'additiveData from a Gaussian, binomial, Poisson or gamma GAM with any valid link; per-feature terms (P-spline, cyclic, thin plate, linear, off) with k, degree, penalty order and shape constraints; λ by REML, GCV or fixed; fitted by P-IRLS, backfitting, gradient descent, SGD, Adam or L-BFGS (gamTrainingRun in a worker) and played iteration by iteration: partial effects with bands against the truth, the basis and penalty, the response scale, the training trace against a second method, and the REML profile with a draggable λ.',
    tags: ['showcase', 'gam', 'P-IRLS', 'backfitting', 'SGD', 'Adam', 'L-BFGS', 'REML', 'links', 'basis'],
    render: () => <GamShowcase />,
  },
]
