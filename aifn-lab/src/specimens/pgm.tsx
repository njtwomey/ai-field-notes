import type { Specimen } from '../specimen'
import { CasinoSpecimen } from './_pgm/CasinoSpecimen'
import { IsingSpecimen } from './_pgm/IsingSpecimen'
import { LdaSpecimen } from './_pgm/LdaSpecimen'
import { ModelSpecimen } from './_pgm/ModelSpecimen'
import { TreeBpSpecimen } from './_pgm/TreeBpSpecimen'

export const specimens: Specimen[] = [
  {
    module: 'pgm',
    title: 'A model described once, drawn twice',
    description:
      'Models written in the description language (LDA, the sprinkler network, a Gaussian mixture, the casino HMM), drawn in plate notation and as an expanded factor graph with a chosen Markov blanket.',
    tags: ['model description', 'plate notation', 'factor graph', 'Markov blanket', 'LDA'],
    render: () => <ModelSpecimen />,
  },
  {
    module: 'pgm',
    title: 'Belief propagation on a tree',
    description: 'Sum-product and max-product messages stepped one at a time on a tree, against exact (max-)marginals.',
    tags: ['belief propagation', 'sum-product', 'max-product', 'messages', 'tree'],
    render: () => <TreeBpSpecimen />,
  },
  {
    module: 'pgm',
    title: 'Loopy belief propagation on an Ising grid',
    description:
      'Loopy BP beliefs on a 5 × 5 Ising model against Gibbs-sampling estimates, with coupling, field and damping.',
    tags: ['loopy belief propagation', 'Ising model', 'Gibbs sampling', 'damping', 'Bethe'],
    render: () => <IsingSpecimen />,
  },
  {
    module: 'pgm',
    title: 'The dishonest casino: forward–backward and Viterbi',
    description: 'Posterior and filtered probabilities of the loaded die, and the Viterbi path, on sampled rolls.',
    tags: ['hidden Markov model', 'forward–backward', 'Viterbi', 'casino'],
    render: () => <CasinoSpecimen />,
  },
  {
    module: 'pgm',
    title: 'LDA by collapsed Gibbs sampling',
    description: 'The registered LDA engine, picked by infer() from the model description, learning the bars topics.',
    tags: ['LDA', 'topic model', 'collapsed Gibbs', 'custom engine'],
    render: () => <LdaSpecimen />,
  },
]
