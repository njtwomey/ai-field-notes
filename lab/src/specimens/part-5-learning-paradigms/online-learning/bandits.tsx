import type { Specimen } from '@lab/specimen'
import { RegretSpecimen, UcbStepsSpecimen } from './_bandits/figures'
import { BanditTrainerSpecimen } from '@lab/specimens/part-5-learning-paradigms/reinforcement-learning/_trainer/trainers'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/online-learning',
    title: 'Regret curves',
    description:
      'Uniform, ε-greedy, EXP3, UCB1, KL-UCB and Thompson sampling on Bernoulli arms, with the Lai–Robbins bound.',
    tags: ['regret', 'UCB', 'Thompson sampling', 'KL-UCB', 'EXP3', 'replicate'],
    render: () => <RegretSpecimen />,
  },
  {
    module: 'part-5-learning-paradigms/online-learning',
    title: 'A bandit run round by round',
    description: 'UCB indices or Thompson draws per arm, stepped through a traced run.',
    tags: ['UCB', 'Thompson sampling', 'trace'],
    render: () => <UcbStepsSpecimen />,
  },
  {
    module: 'part-5-learning-paradigms/online-learning',
    title: 'Gym trainer: a bandit',
    description:
      'One training run of a bandit policy in the worker; pick a round on the curve to see the pulls up to it.',
    tags: ['training', 'regret', 'UCB', 'Thompson sampling', 'worker'],
    render: () => <BanditTrainerSpecimen />,
  },
]
