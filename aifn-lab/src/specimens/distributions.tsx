import type { Specimen } from '../specimen'
import { FamiliesSpecimen, MixtureSpecimen, MultivariateNormalSpecimen } from './_distributions/figures'
import { KlSpecimen } from './_distributions/kl'
import { TransformedSpecimen } from './_distributions/transformed'

export const specimens: Specimen[] = [
  {
    module: 'distributions',
    title: 'Every univariate family',
    description:
      'Pick any of the 24 univariate families and move its parameters: DistributionView draws the density or mass over a histogram of draws, the cdf against the empirical cdf, and the closed-form moments, entropy and mode.',
    tags: ['DistributionView', 'density', 'cdf', 'quantile', 'sample', 'moments'],
    render: () => <FamiliesSpecimen />,
  },
  {
    module: 'distributions',
    title: 'Conditioning a multivariate normal',
    description:
      'MultivariateNormal.condition gives the Gaussian of x₁ given x₂ in closed form; it matches the joint density sliced at x₂ and renormalised.',
    tags: ['MultivariateNormal', 'condition', 'marginal', 'Schur complement'],
    render: () => <MultivariateNormalSpecimen />,
  },
  {
    module: 'distributions',
    title: 'A normal mixture',
    description:
      'Mixture of two normals: log-sum-exp density, weighted cdf, numerical quantile and total-variance moments, against 2,000 draws.',
    tags: ['Mixture', 'logsumexp', 'quantile'],
    render: () => <MixtureSpecimen />,
  },
  {
    module: 'distributions',
    title: 'Transformed distributions',
    description:
      'The change of variables in three aligned panels: an input density, a map g and the output density, with draws. Transformed pushes through bijectors (affine, exp, log, sigmoid, tanh, softplus, power, Φ); Pushforward sums over both preimages of x². The Jacobian factor makes the output integrate to 1; supports are images of supports.',
    tags: ['Transformed', 'Pushforward', 'bijector', 'change of variables', 'Jacobian', 'support', 'LogNormal', 'χ²'],
    render: () => <TransformedSpecimen />,
  },
  {
    module: 'distributions',
    title: 'KL divergence',
    description:
      'KL(p ‖ q) as the area under p log(p/q) for any two of six families, in both directions, with klAuto (closed form or quadrature), a Monte Carlo estimate with its standard error, the cross-entropy identity and the Jensen–Shannon divergence. Then forward against reverse KL: the best normal for a bimodal mixture by L-BFGS (normalProjection), mode covering against mode seeking from a draggable start. Last, every registered closed form checked against quadrature and Monte Carlo.',
    tags: [
      'kl',
      'klAuto',
      'klNumerical',
      'klIntegrand',
      'Kullback–Leibler',
      'cross-entropy',
      'Jensen–Shannon',
      'Monte Carlo',
      'normalProjection',
      'mode seeking',
    ],
    render: () => <KlSpecimen />,
  },
]
