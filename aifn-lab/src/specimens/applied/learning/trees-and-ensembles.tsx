import type { Specimen } from '../../../specimen'
import { DecisionTreeShowcase } from './_trees/showcase'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/trees-and-ensembles',
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
