import type { Specimen } from '@lab/specimen'
import { EvidenceOverLengthscale, PosteriorDraws, SparseInducingPoints } from './_gaussian-processes/figures'
import { GpClassification } from './_gaussian-processes/showcase-gpc'
import { GplvmFonts } from './_gaussian-processes/showcase-gplvm'
import { RvmShowcase } from './_gaussian-processes/showcase-rvm'
import { SparseGpShowcase } from './_gaussian-processes/showcase-sparse'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'Prior and posterior draws',
    description:
      'gpPosterior with draggable observations: the posterior mean, ± 2 sd and five draws from fixed normals, or the prior draws; jitter reported.',
    tags: ['gpPosterior', 'samplePrior', 'draws', 'handles'],
    render: () => <PosteriorDraws />,
  },
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'Log marginal likelihood over the lengthscale',
    description:
      'logMarginalLikelihood split into data fit and complexity over log ℓ, with the posterior at the ℓ on a draggable line.',
    tags: ['logMarginalLikelihood', 'evidence', 'type-II maximum likelihood'],
    render: () => <EvidenceOverLengthscale />,
  },
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'Sparse GP inducing points',
    description:
      'sparseGp (VFE, FITC, DTC, SoR) against the exact GP with draggable inducing inputs, and fitSparseGp to optimise them.',
    tags: ['sparseGp', 'fitSparseGp', 'Titsias', 'FITC', 'inducing points'],
    render: () => <SparseInducingPoints />,
  },
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
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
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'Showcase: a GPLVM manifold of fonts',
    description:
      'gplvmFitSteps played from PCA on 66 real fonts, the latent space by design class over the predictive sd, and a draggable latent point rendered as glyph outlines by project(x) beside its nearest real fonts.',
    tags: ['showcase', 'GPLVM', 'fonts', 'latent variable model', 'L-BFGS', 'project'],
    render: () => <GplvmFonts />,
  },
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'Showcase: sparse GPs, inducing point by inducing point',
    description:
      'sparseGpGrowSteps (greedy inducing-point selection by the ELBO) or sparseGpFitSteps (L-BFGS on θ, σ² and Z) played step by step on gapped, uneven data: the sparse band against the exact GP, Z appearing or moving, the objective and the gap to the exact evidence; Z draggable.',
    tags: ['showcase', 'sparse GP', 'inducing points', 'VFE', 'FITC', 'Titsias', 'greedy selection', 'L-BFGS'],
    render: () => <SparseGpShowcase />,
  },
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'Showcase: relevance vector machine, basis by basis',
    description:
      'rvmFastSteps (Tipping and Faul: add, re-estimate or delete one basis function per step) or rvmReestimationSteps on sinc and gapped data: the predictive band with relevance vectors ringed, the evidence and the count per step, log αᵢ per candidate, and a probe showing sᵢ, qᵢ.',
    tags: ['showcase', 'RVM', 'relevance vector machine', 'sparse Bayesian learning', 'ARD', 'Tipping', 'evidence'],
    render: () => <RvmShowcase />,
  },
]
