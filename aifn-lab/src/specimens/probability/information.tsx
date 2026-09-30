import type { Specimen } from '../../specimen'
import { DivergencesSpecimen, KsgSpecimen } from './_information/figures'

export const specimens: Specimen[] = [
  {
    module: 'probability/information',
    title: 'Divergences between Bernoullis',
    description:
      'KL in both directions, Jensen–Shannon, total variation, Hellinger and Pearson χ² between Bernoulli(p) and Bernoulli(q).',
    tags: ['klDivergence', 'jensenShannonDivergence', 'totalVariation', 'hellingerDistance', 'fDivergence'],
    render: () => <DivergencesSpecimen />,
  },
  {
    module: 'probability/information',
    title: 'Mutual information from samples (KSG)',
    description: 'ksgMutualInformation on correlated Gaussian samples against the closed form −½ log(1 − ρ²).',
    tags: ['ksgMutualInformation', 'mutual information', 'k-nearest neighbours'],
    render: () => <KsgSpecimen />,
  },
]
