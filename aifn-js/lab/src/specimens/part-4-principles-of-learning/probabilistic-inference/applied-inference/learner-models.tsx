import type { Specimen } from '@lab/specimen'
import { LearnerEquitySweepSpecimen, ZeroInflatedAbilitySpecimen } from './_learner-models/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/applied-inference',
    title: "Zero-inflated learner models: ability without the zeros' bias",
    description:
      'IRT, a linear knowledge-tracing machine and IRT-ZILM (Twomey et al., 2022) fitted to simulated students whose neurodivergent conditions make some item formats unsuitable: ability estimates against the truth by group, the bias per group, and the posterior that each zero came from the context rather than low ability.',
    tags: ['zero-inflated', 'item response theory', 'learner model', 'equity', 'neurodiversity', 'IRT-ZILM'],
    render: () => <ZeroInflatedAbilitySpecimen />,
  },
  {
    module: 'part-4-principles-of-learning/probabilistic-inference/applied-inference',
    title: 'Learner-model equity as the zero-inflation gap grows',
    description:
      'The equity gap (ability bias of students with a condition minus those without), ability error and per-group bias of IRT, a linear KTM and IRT-ZILM as the zero-inflation rate on unsuitable formats rises.',
    tags: ['zero-inflated', 'item response theory', 'equity', 'fairness', 'IRT-ZILM'],
    render: () => <LearnerEquitySweepSpecimen />,
  },
]
