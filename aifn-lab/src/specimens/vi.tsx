import type { Specimen } from '../specimen'
import { BlackBoxVi, CaviMixture, CaviNormalGamma, GradientVariance } from './_vi/figures'

export const specimens: Specimen[] = [
  {
    module: 'vi',
    title: 'Black-box VI fits a Gaussian',
    description:
      'Mean-field and full-rank Gaussians fitted by Adam on stochastic ELBO gradients to a correlated Gaussian, a banana and a two-mode mixture, with the ELBO trace and KL(q ‖ p).',
    tags: ['BBVI', 'ELBO', 'reparameterisation', 'score function', 'mean field', 'reverse KL'],
    render: () => <BlackBoxVi />,
  },
  {
    module: 'vi',
    title: 'ELBO gradient estimators and their variance',
    description:
      'The variance of the reparameterisation and score-function estimators, with and without baselines, against the number of draws.',
    tags: ['gradient estimator', 'REINFORCE', 'baseline', 'control variate', 'variance'],
    render: () => <GradientVariance />,
  },
  {
    module: 'vi',
    title: 'CAVI on a Gaussian mixture',
    description:
      'Coordinate-ascent VI for a Bayesian Gaussian mixture (Bishop §10.2): predictive density, components and the ELBO.',
    tags: ['CAVI', 'Gaussian mixture', 'Dirichlet', 'Wishart', 'ELBO'],
    render: () => <CaviMixture />,
  },
  {
    module: 'vi',
    title: 'CAVI for an unknown mean and precision',
    description:
      'The mean-field q(μ)q(τ) of a Gaussian with unknown mean and precision against the exact normal-gamma posterior, and its ELBO against log p(x).',
    tags: ['CAVI', 'normal-gamma', 'mean field', 'ElboView'],
    render: () => <CaviNormalGamma />,
  },
]
