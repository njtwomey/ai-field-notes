import type { Specimen } from '@lab/specimen'
import { HedgeRegretSpecimen, LearningRateSpecimen } from './_online/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/online-learning',
    title: 'Learning with expert advice: Hedge and regret',
    description:
      'Hedge, fixed share and follow-the-leader played round by round on stochastic, switching and adversarial loss sequences, with the regret against the best expert drawn under its bound; a sweep of the learning rate.',
    tags: [
      'Hedge',
      'exponential weights',
      'regret',
      'fixed share',
      'follow the leader',
      'learning rate',
      'online learning',
    ],
    render: () => (
      <>
        <HedgeRegretSpecimen />
        <LearningRateSpecimen />
      </>
    ),
  },
]
