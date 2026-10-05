import type { Specimen } from '@lab/specimen'
import { SmoSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/supervised-learning/kernel-methods',
    title: 'SMO step by step',
    description: 'smoSteps: the working pair, the dual variables, the decision function and the KKT gap at every step.',
    tags: ['smoSteps', 'SVM', 'dual', 'trace'],
    render: () => <SmoSpecimen />,
  },
]
