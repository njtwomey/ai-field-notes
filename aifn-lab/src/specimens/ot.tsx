import type { Specimen } from '../specimen'
import { SinkhornSpecimen, Wasserstein1dSpecimen } from './_ot/figures'

export const specimens: Specimen[] = [
  {
    module: 'ot',
    title: 'Sinkhorn plans',
    description:
      'Log-domain Sinkhorn between two point clouds: the plan per iteration as ε changes, against the exact cost.',
    tags: ['Sinkhorn', 'entropic', 'transport plan', 'trace'],
    render: () => <SinkhornSpecimen />,
  },
  {
    module: 'ot',
    title: 'Wasserstein distance on the line',
    description: 'W₁ and W₂ between two samples as the distance between their quantile functions.',
    tags: ['Wasserstein', 'quantile', '1-D'],
    render: () => <Wasserstein1dSpecimen />,
  },
]
