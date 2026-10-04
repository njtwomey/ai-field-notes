import type { Specimen } from '@lab/specimen'
import { FullBatchShowcase } from './_full-batch/showcase'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/feedforward-networks',
    title: 'Showcase: training a neural network with L-BFGS',
    description:
      'A small MLP trained in the browser by full-batch L-BFGS against gradient descent, Adam and SGD from the same initial weights: the fit at any iteration, the loss against iterations and against full-data gradient evaluations, and the L-BFGS line search, step lengths and curvature pairs; a ReLU preset shows the kinks hurting the line search.',
    tags: [
      'showcase',
      'L-BFGS',
      'quasi-Newton',
      'full batch',
      'line search',
      'Adam',
      'SGD',
      'MLP',
      'fullBatchTraining',
      'fullBatchComparison',
    ],
    render: () => <FullBatchShowcase />,
  },
]
