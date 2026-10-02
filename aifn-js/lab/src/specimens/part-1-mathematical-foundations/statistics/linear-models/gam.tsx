import type { Specimen } from '@lab/specimen'
import { BackfittingSweeps, EbmShapes, PartialEffects } from './_gam/figures'
import { GamShowcase } from './_gam/showcase-gam'
import { ExpectileShowcase } from './_gam/showcase-expectile'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/statistics/linear-models',
    title: 'A GAM with partial effects and bands',
    description:
      'gam with three P-spline smooths, λ by REML or GCV: partial effects, Bayesian ± 2 se bands, partial residuals and posterior draws, EDF per term.',
    tags: ['gam', 'REML', 'GCV', 'partial effects', 'posterior draws'],
    render: () => <PartialEffects />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/linear-models',
    title: 'Backfitting sweep by sweep',
    description: 'backfitting as a traceable algorithm: each curve after each sweep, and the residual sum of squares.',
    tags: ['backfitting', 'trace', 'Player'],
    render: () => <BackfittingSweeps />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/linear-models',
    title: 'Explainable boosting machine shapes',
    description:
      'explainableBoostingMachine: step-function shapes from cyclic boosting of one-split trees, by number of rounds.',
    tags: ['explainableBoostingMachine', 'boosting', 'shapes'],
    render: () => <EbmShapes />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/linear-models',
    title: 'Showcase: interactive GAM',
    description:
      'additiveData from a Gaussian, binomial, Poisson or gamma GAM with any valid link; per-feature terms (P-spline, cyclic, thin plate, linear, off) with k, degree, penalty order and shape constraints; λ by REML, GCV or fixed; fitted by P-IRLS, backfitting, gradient descent, SGD, Adam or L-BFGS (gamTrainingRun in a worker) and played iteration by iteration: partial effects with bands against the truth, the basis and penalty, the response scale, the training trace against a second method, and the REML profile with a draggable λ.',
    tags: ['showcase', 'gam', 'P-IRLS', 'backfitting', 'SGD', 'Adam', 'L-BFGS', 'REML', 'links', 'basis'],
    render: () => <GamShowcase />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/linear-models',
    title: 'Showcase: smoothing hyperparameters and expectile bands',
    description:
      'curve1d (a sine with spreading and pinching noise, a bump with skewed noise, Poisson counts, binary outcomes, gamma responses), each with its true law: a P-spline smooth with k, degree, penalty order and λ by REML, GCV or fixed (the criterion and EDF against λ, with a draggable λ); a band from two expectile GAMs at τ_lo = (1 − level)/2 and τ_hi = 1 − τ_lo, each trained by P-IRLS (LAWS), gradient descent, SGD, Adam or L-BFGS on the asymmetric squared loss (expectileTrainingRun in a worker), played iteration by iteration against the true expectile band, with the share of points inside against the level, the basis × coefficients and the training trace; for counts, binary and gamma data the family GAM on the response and link scales.',
    tags: ['showcase', 'gam', 'expectiles', 'LAWS', 'Adam', 'P-spline', 'REML', 'GCV', 'EDF', 'links'],
    render: () => <ExpectileShowcase />,
  },
]
