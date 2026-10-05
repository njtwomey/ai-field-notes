import type { Specimen } from '@lab/specimen'
import { ModelSpecimen } from './_model/ModelSpecimen'

export const specimens: Specimen[] = [
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/exact-inference',
    title: 'A model described once, drawn twice',
    description:
      'Models written in the description language (LDA, the sprinkler network, a Gaussian mixture, the casino HMM), drawn in plate notation and as an expanded factor graph with a chosen Markov blanket.',
    tags: ['model description', 'plate notation', 'factor graph', 'Markov blanket', 'LDA'],
    render: () => <ModelSpecimen />,
  },
]
