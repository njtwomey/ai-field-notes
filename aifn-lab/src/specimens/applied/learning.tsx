import type { Specimen } from '../../specimen'
import { DecisionRegionsSpecimen } from './learning/_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning',
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
