import type { Specimen } from '@lab/specimen'
import { MlpTrainingSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/feedforward-networks',
    title: 'An MLP learning a decision surface',
    description:
      'A tanh MLP trained by Adam on two spirals, XOR or moons (a traceable training loop): the decision surface at any step, the loss trace, and every parameter tensor with its gradient.',
    tags: ['training', 'MLP', 'Adam', 'trace', 'ParamsView', 'decision surface'],
    render: () => <MlpTrainingSpecimen />,
  },
]
