import type { Specimen } from '../../../specimen'
import { EvidenceOverLengthscale, PosteriorDraws, SparseInducingPoints } from './_gaussian-processes/figures'
import { GpClassification } from './_gaussian-processes/showcase-gpc'
import { GplvmFonts } from './_gaussian-processes/showcase-gplvm'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/gaussian-processes',
    title: 'Prior and posterior draws',
    description:
      'gpPosterior with draggable observations: the posterior mean, ± 2 sd and five draws from fixed normals, or the prior draws; jitter reported.',
    tags: ['gpPosterior', 'samplePrior', 'draws', 'handles'],
    render: () => <PosteriorDraws />,
  },
  {
    module: 'applied/learning/gaussian-processes',
    title: 'Log marginal likelihood over the lengthscale',
    description:
      'logMarginalLikelihood split into data fit and complexity over log ℓ, with the posterior at the ℓ on a draggable line.',
    tags: ['logMarginalLikelihood', 'evidence', 'type-II maximum likelihood'],
    render: () => <EvidenceOverLengthscale />,
  },
  {
    module: 'applied/learning/gaussian-processes',
    title: 'Sparse GP inducing points',
    description:
      'sparseGp (VFE, FITC, DTC, SoR) against the exact GP with draggable inducing inputs, and fitSparseGp to optimise them.',
    tags: ['sparseGp', 'fitSparseGp', 'Titsias', 'FITC', 'inducing points'],
    render: () => <SparseInducingPoints />,
  },
  {
    module: 'applied/learning/gaussian-processes',
    title: 'Showcase: Gaussian-process classification',
    description:
      'Laplace or EP GP classification on two moons: the predictive probability and latent mean as fields, the latent band along a draggable line cut, the hyperparameters against the evidence, and an L-BFGS hyperparameter fit (evidence gradients by implicit differentiation) played step by step.',
    tags: [
      'showcase',
      'GP classification',
      'Laplace',
      'expectation propagation',
      'fitGpClassifier',
      'evidence',
      'L-BFGS',
    ],
    render: () => <GpClassification />,
  },
  {
    module: 'applied/learning/gaussian-processes',
    title: 'Showcase: a GPLVM manifold of fonts',
    description:
      'gplvmFitSteps played from PCA on 66 real fonts, the latent space by design class over the predictive sd, and a draggable latent point rendered as glyph outlines by project(x) beside its nearest real fonts.',
    tags: ['showcase', 'GPLVM', 'fonts', 'latent variable model', 'L-BFGS', 'project'],
    render: () => <GplvmFonts />,
  },
]
