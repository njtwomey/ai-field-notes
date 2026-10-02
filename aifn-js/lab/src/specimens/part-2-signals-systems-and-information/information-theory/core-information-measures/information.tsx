import type { Specimen } from '@lab/specimen'
import { DivergencesSpecimen, KsgSpecimen } from './_information/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/information-theory/core-information-measures',
    title: 'Divergences between Bernoullis',
    description:
      'KL in both directions, Jensen–Shannon, total variation, Hellinger and Pearson χ² between Bernoulli(p) and Bernoulli(q).',
    tags: ['klDivergence', 'jensenShannonDivergence', 'totalVariation', 'hellingerDistance', 'fDivergence'],
    render: () => <DivergencesSpecimen />,
  },
  {
    module: 'part-2-signals-systems-and-information/information-theory/core-information-measures',
    title: 'Mutual information from samples (KSG)',
    description: 'ksgMutualInformation on correlated Gaussian samples against the closed form −½ log(1 − ρ²).',
    tags: ['ksgMutualInformation', 'mutual information', 'k-nearest neighbours'],
    render: () => <KsgSpecimen />,
  },
]
