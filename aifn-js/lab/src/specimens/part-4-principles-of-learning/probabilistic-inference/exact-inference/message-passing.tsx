import type { Specimen } from '@lab/specimen'
import { IsingSpecimen } from './_message-passing/IsingSpecimen'
import { TreeBpSpecimen } from './_message-passing/TreeBpSpecimen'

export const specimens: Specimen[] = [
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/exact-inference',
    title: 'Belief propagation on a tree',
    description: 'Sum-product and max-product messages stepped one at a time on a tree, against exact (max-)marginals.',
    tags: ['belief propagation', 'sum-product', 'max-product', 'messages', 'tree'],
    render: () => <TreeBpSpecimen />,
  },
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/exact-inference',
    title: 'Loopy belief propagation on an Ising grid',
    description:
      'Loopy BP beliefs on a 5 × 5 Ising model against Gibbs-sampling estimates, with coupling, field and damping.',
    tags: ['loopy belief propagation', 'Ising model', 'Gibbs sampling', 'damping', 'Bethe'],
    render: () => <IsingSpecimen />,
  },
]
