import type { Specimen } from '../specimen'
import { DecisionRegionsSpecimen, OutputCodeSpecimen, SmoSpecimen } from './_classify/figures'
import { BoostingSpecimen, PruningSpecimen, TreeGrowthSpecimen } from './_classify/trees'

export const specimens: Specimen[] = [
  {
    module: 'classify',
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
  {
    module: 'classify',
    title: 'SMO step by step',
    description: 'smoSteps: the working pair, the dual variables, the decision function and the KKT gap at every step.',
    tags: ['smoSteps', 'SVM', 'dual', 'trace'],
    render: () => <SmoSpecimen />,
  },
  {
    module: 'classify',
    title: 'Growing and pruning trees',
    description:
      'treeGrowthSteps with the split search at each node, drawn with TreeView; costComplexityPath and pruneTree.',
    tags: ['CART', 'splitSearch', 'TreeView', 'costComplexityPath', 'pruneTree'],
    render: () => (
      <>
        <TreeGrowthSpecimen />
        <PruningSpecimen />
      </>
    ),
  },
  {
    module: 'classify',
    title: 'Boosting rounds',
    description:
      'adaBoost (SAMME) and gradientBoosting, stage by stage: the score, the boundary and the sample weights.',
    tags: ['adaBoost', 'gradientBoosting', 'ensembles'],
    render: () => <BoostingSpecimen />,
  },
  {
    module: 'classify',
    title: 'Output codes',
    description: 'Code matrices for one-versus-rest, one-versus-one, exhaustive and random codes, with codeDistance.',
    tags: ['outputCode', 'exhaustiveCode', 'randomCode', 'codeDistance', 'ECOC'],
    render: () => <OutputCodeSpecimen />,
  },
]
