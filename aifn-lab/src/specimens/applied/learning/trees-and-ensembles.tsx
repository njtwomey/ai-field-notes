import type { Specimen } from '../../../specimen'
import { PruningSpecimen, TreeGrowthSpecimen } from './_shared/trees'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/trees-and-ensembles',
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
]
