import type { Specimen } from '../../../specimen'
import { DbnSpecimen, HopfieldSpecimen, RbmSpecimen } from './_boltzmann/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/generative/boltzmann',
    title: 'Boltzmann machines and Hopfield networks',
    description:
      'A restricted Boltzmann machine learning bars and stripes by contrastive divergence, with its exact log-likelihood, receptive fields and Gibbs samples; a deep belief network stacked one RBM at a time on noisy digits, sampled top-down and fine-tuned from one label per digit; classical and modern Hopfield networks recalling noisy digits, and their capacity.',
    tags: [
      'restricted Boltzmann machine',
      'deep belief network',
      'greedy layer-wise pretraining',
      'contrastive divergence',
      'Hopfield network',
      'modern Hopfield',
      'associative memory',
      'energy-based',
    ],
    render: () => (
      <>
        <RbmSpecimen />
        <DbnSpecimen />
        <HopfieldSpecimen />
      </>
    ),
  },
]
