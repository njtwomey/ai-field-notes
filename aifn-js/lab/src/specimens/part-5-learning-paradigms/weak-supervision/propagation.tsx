import type { Specimen } from '@lab/specimen'
import { LabelPropagationFigure } from './_propagation/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/weak-supervision',
    title: 'Label propagation on a graph',
    description:
      'Two moons, a k-nearest-neighbour graph with heat-kernel weights and one or a few labels per moon: harmonic label propagation (Zhu and Ghahramani) or label spreading (Zhou et al.) played step by step from the given labels, with the accuracy on the unlabelled points and the closed-form limit.',
    tags: [
      'label propagation',
      'label spreading',
      'semi-supervised',
      'harmonic function',
      'k-NN graph',
      'labelPropagationSteps',
      'labelSpreadingSteps',
      'harmonicLabels',
    ],
    render: () => <LabelPropagationFigure />,
  },
]
