import type { Specimen } from '../../specimen'
import {
  ContinuousSamplersSpecimen,
  DirichletSpecimen,
  DiscreteSamplersSpecimen,
  MultivariateNormalSpecimen,
  ReplicateSpecimen,
  StreamIndependenceSpecimen,
} from './_random/figures'

export const specimens: Specimen[] = [
  {
    module: 'foundation/random',
    title: 'Continuous samplers against their densities',
    description:
      'Histograms of normal, exponential, gamma (Marsaglia–Tsang, including shape < 1), beta, Student t and χ² draws under the exact density, with sample moments and the KS distance.',
    tags: ['sampler', 'gamma', 'beta', 'histogram', 'ks'],
    render: () => <ContinuousSamplersSpecimen />,
  },
  {
    module: 'foundation/random',
    title: 'Poisson and binomial draws',
    description:
      'Relative frequencies against the mass function: Poisson by inversion below λ = 10 and PTRS above; binomial by inversion or the beta order-statistic recursion.',
    tags: ['sampler', 'poisson', 'binomial', 'discrete'],
    render: () => <DiscreteSamplersSpecimen />,
  },
  {
    module: 'foundation/random',
    title: 'Stream independence',
    description:
      'Pairs of uniforms from a parent and child, two siblings or successive draws fill the square evenly; the correlation matrix of six streams is near zero off the diagonal.',
    tags: ['stream', 'philox', 'independence', 'child'],
    render: () => <StreamIndependenceSpecimen />,
  },
  {
    module: 'foundation/random',
    title: 'Dirichlet draws on the simplex',
    description:
      'Dirichlet(α) draws in barycentric coordinates. Small concentrations push mass to the corners; the log-space construction keeps them finite.',
    tags: ['sampler', 'dirichlet', 'simplex'],
    render: () => <DirichletSpecimen />,
  },
  {
    module: 'foundation/random',
    title: 'Multivariate normal from a Cholesky factor',
    description: 'μ + L z for a 2 × 2 covariance, with the 2σ ellipse L · (circle of radius 2).',
    tags: ['sampler', 'multivariate normal', 'cholesky'],
    render: () => <MultivariateNormalSpecimen />,
  },
  {
    module: 'foundation/random',
    title: 'Replicates with prefix reuse',
    description:
      'Random walks on child(stream("walks"), k). Raising the count computes only the new walks; the existing ones never change.',
    tags: ['replicate', 'cache', 'common random numbers'],
    render: () => <ReplicateSpecimen />,
  },
]
