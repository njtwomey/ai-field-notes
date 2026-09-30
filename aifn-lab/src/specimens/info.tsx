import type { Specimen } from '../specimen'
import {
  CapacitySpecimen,
  DivergencesSpecimen,
  HuffmanSpecimen,
  KsgSpecimen,
  RateDistortionSpecimen,
} from './_info/figures'

export const specimens: Specimen[] = [
  {
    module: 'info',
    title: 'Blahut–Arimoto converging to capacity',
    description:
      'The traced capacity algorithm on five channels: upper and lower bounds meet at the capacity, which matches the closed forms for the binary symmetric, erasure and Z channels.',
    tags: ['blahutArimotoCapacity', 'channelCapacity', 'trace', 'channel'],
    render: () => <CapacitySpecimen />,
  },
  {
    module: 'info',
    title: 'Rate–distortion curve',
    description:
      'rateDistortionCurve for a binary source with Hamming distortion, one Blahut–Arimoto run per slope, against R(D) = H(p) − H(D).',
    tags: ['rateDistortion', 'Blahut–Arimoto', 'R(D)'],
    render: () => <RateDistortionSpecimen />,
  },
  {
    module: 'info',
    title: 'Huffman tree against the entropy',
    description:
      'huffmanSteps on a Zipf distribution, drawn as a tree merge by merge with codeword paths, then codeword lengths against −log₂ p and expected lengths of the Huffman, Shannon–Fano and Shannon codes against H.',
    tags: ['huffmanCode', 'huffmanSteps', 'Tree', 'shannonFanoCode', 'shannonCode', 'kraftSum', 'coding'],
    render: () => <HuffmanSpecimen />,
  },
  {
    module: 'info',
    title: 'Divergences between Bernoullis',
    description:
      'KL in both directions, Jensen–Shannon, total variation, Hellinger and Pearson χ² between Bernoulli(p) and Bernoulli(q).',
    tags: ['klDivergence', 'jensenShannonDivergence', 'totalVariation', 'hellingerDistance', 'fDivergence'],
    render: () => <DivergencesSpecimen />,
  },
  {
    module: 'info',
    title: 'Mutual information from samples (KSG)',
    description: 'ksgMutualInformation on correlated Gaussian samples against the closed form −½ log(1 − ρ²).',
    tags: ['ksgMutualInformation', 'mutual information', 'k-nearest neighbours'],
    render: () => <KsgSpecimen />,
  },
]
