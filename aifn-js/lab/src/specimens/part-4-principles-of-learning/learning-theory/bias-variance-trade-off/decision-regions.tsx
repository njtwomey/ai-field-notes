import type { Specimen } from '@lab/specimen'
import { DecisionRegionsSpecimen } from '@lab/specimens/part-5-learning-paradigms/supervised-learning/_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-4-principles-of-learning/learning-theory/bias-variance-trade-off',
    title: 'Decision regions of every classifier',
    description:
      'k-NN, naive Bayes, LDA, QDA, the perceptron, SVMs, Crammer–Singer, trees, forests, boosting and multiclass reductions on 2-D data, in DecisionRegionView with a draggable query point.',
    tags: [
      'DecisionRegionView',
      'kNearestNeighbours',
      'supportVectorMachine',
      'decisionTree',
      'randomForest',
      'oneVersusOne',
    ],
    render: () => <DecisionRegionsSpecimen />,
  },
]
