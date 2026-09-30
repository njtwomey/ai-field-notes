import type { Specimen } from '../specimen'
import { RegretSpecimen, UcbStepsSpecimen } from './_bandits/figures'

export const specimens: Specimen[] = [
  {
    module: 'bandits',
    title: 'Regret curves',
    description:
      'Uniform, ε-greedy, EXP3, UCB1, KL-UCB and Thompson sampling on Bernoulli arms, with the Lai–Robbins bound.',
    tags: ['regret', 'UCB', 'Thompson sampling', 'KL-UCB', 'EXP3', 'replicate'],
    render: () => <RegretSpecimen />,
  },
  {
    module: 'bandits',
    title: 'A bandit run round by round',
    description: 'UCB indices or Thompson draws per arm, stepped through a traced run.',
    tags: ['UCB', 'Thompson sampling', 'trace'],
    render: () => <UcbStepsSpecimen />,
  },
]
