import type { Specimen } from '@lab/specimen'
import { NodeClassificationShowcase } from './_graph/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/graph-neural-networks',
    title: 'Graph neural networks: message passing',
    description:
      "Showcase: a two-layer GCN, GAT or GraphSAGE trained in the browser on Zachary's karate club from one labelled member per club: the predicted clubs on the friendship graph, the GAT's attention on each friendship, the members' 2-D logits separating, and the accuracy curves, played checkpoint by checkpoint.",
    tags: [
      'showcase',
      'graph neural network',
      'GCN',
      'GAT',
      'GraphSAGE',
      'message passing',
      'karate club',
      'semi-supervised',
      'nodeClassificationRun',
    ],
    render: () => <NodeClassificationShowcase />,
  },
]
