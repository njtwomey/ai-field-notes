import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'

const ROWS: { k: string; y: number }[] = [
  { k: '1', y: 0 },
  { k: '2', y: 1.3 },
  { k: 'K', y: 3.3 },
]
const MID = 1.65

const nodes: DiagramNode[] = [
  { id: 'vx', x: 0, y: 2.35, shape: 'text', w: 0.6, h: 0.5, label: '$\\vdots$' },
  { id: 'vf', x: 2.2, y: 2.35, shape: 'text', w: 0.6, h: 0.5, label: '$\\vdots$' },
  {
    id: 'pool',
    x: 8.2,
    y: MID,
    w: 5,
    h: 4.2,
    label:
      'attention pooling\n$a_k = \\operatorname{softmax}_k \\, \\wvec^\\top \\tanh(\\Vmat\\hvec_k)$\n$\\zvec = \\sum_k a_k \\hvec_k$',
    tone: 2,
  },
  { id: 'g', x: 12.4, y: MID, w: 2, h: 0.9, label: 'classifier $g$', tone: 1 },
  { id: 'out', x: 14.6, y: MID, shape: 'text', w: 1.4, label: '$\\theta(X)$' },
]
const edges: DiagramEdge[] = [
  { from: 'pool', to: 'g', label: '$\\zvec$' },
  { from: 'g', to: 'out' },
]
for (const { k, y } of ROWS) {
  nodes.push(
    { id: `x${k}`, x: 0, y, shape: 'circle', w: 0.9, h: 0.9, label: `$\\xvec_${k}$`, tone: 'ink' },
    { id: `f${k}`, x: 2.2, y, w: 1.4, h: 0.8, label: '$f$', tone: 0 },
    // An entry point on the pooling block's left edge, so each embedding arrives at its own height.
    { id: `in${k}`, x: 5.7, y, shape: 'dot' },
  )
  edges.push({ from: `x${k}`, to: `f${k}` }, { from: `f${k}`, to: `in${k}`, label: `$\\hvec_${k}$` })
}

const spec: DiagramSpec = {
  unit: 38,
  nodes,
  edges,
  groups: [
    { id: 'bag', label: 'bag $X$', tone: 'ink', around: ['x1', 'xK'], pad: 0.3 },
    { id: 'shared', label: 'shared', tone: 0, dashed: true, around: ['f1', 'fK'], pad: 0.3 },
  ],
}

/** Attention-based MIL: embed every instance, pool with learned weights, classify the bag. */
export function AttentionMilDiagram() {
  return (
    <Interactive
      title="Attention-based MIL pooling"
      caption="Every instance in the bag passes through the same embedding network f. The attention block scores each embedding, normalises the scores over the bag with a softmax and returns the weighted average z. The classifier g maps z to the bag-label probability. The weights a_k show which instances drove the prediction, and reordering the instances changes nothing."
    >
      <Diagram
        spec={spec}
        ariaLabel="Instances x_1 to x_K of a bag each pass through a shared network f; the embeddings h_k enter an attention pooling block that computes weights a_k and the weighted sum z; a classifier maps z to the bag probability"
      />
    </Interactive>
  )
}
