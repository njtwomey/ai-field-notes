import type { Specimen } from '@lab/specimen'
import { CasinoSpecimen } from './_exact/CasinoSpecimen'

export const specimens: Specimen[] = [
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/exact-inference',
    title: 'The dishonest casino: forward–backward and Viterbi',
    description: 'Posterior and filtered probabilities of the loaded die, and the Viterbi path, on sampled rolls.',
    tags: ['hidden Markov model', 'forward–backward', 'Viterbi', 'casino'],
    render: () => <CasinoSpecimen />,
  },
]
