import type { Specimen } from '@lab/specimen'
import { DecisionTreeShowcase } from './_trees/showcase'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/supervised-learning/tree-based-methods',
    title: 'Showcase: decision trees, grown, inspected and pruned',
    description:
      'decisionTree fitted step by step (depth-, breadth- or best-first, with a leaf budget), every split explained from the search the fitter recorded; click a node for its region and split criterion, or the scatter for a decision path; then costComplexityPath played cut by cut against training and validation error.',
    tags: [
      'showcase',
      'CART',
      'decision tree',
      'treeGrowthSteps',
      'nodeRegion',
      'decisionPath',
      'splitCurve',
      'costComplexityPath',
      'TreeView',
    ],
    render: () => <DecisionTreeShowcase />,
  },
]
