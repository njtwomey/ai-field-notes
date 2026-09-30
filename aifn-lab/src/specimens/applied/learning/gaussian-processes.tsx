import type { Specimen } from '../../../specimen'
import { EvidenceOverLengthscale, PosteriorDraws, SparseInducingPoints } from './_gaussian-processes/figures'

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
]
