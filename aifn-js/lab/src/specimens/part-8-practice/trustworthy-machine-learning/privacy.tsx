import type { Specimen } from '@lab/specimen'
import { CompositionSpecimen, DpSgdSpecimen, UtilitySpecimen } from './_privacy/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-8-practice/trustworthy-machine-learning',
    title: 'Differential privacy: mechanisms and accounting',
    description:
      'The error of a private survey share against ε for the Laplace and analytic Gaussian mechanisms and randomised response; how the total ε grows over many releases under sequential, advanced, Rényi-DP and zCDP accounting, with amplification by subsampling; and DP-SGD training a small MLP in the browser for several noise multipliers, test accuracy against the ε spent.',
    tags: [
      'showcase',
      'differential privacy',
      'Laplace mechanism',
      'Gaussian mechanism',
      'randomised response',
      'composition',
      'Rényi DP',
      'zCDP',
      'DP-SGD',
      'privacy accounting',
    ],
    render: () => (
      <>
        <UtilitySpecimen />
        <CompositionSpecimen />
        <DpSgdSpecimen />
      </>
    ),
  },
]
