import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'

const ROWS: { b: string; y: number }[] = [
  { b: '1', y: 0 },
  { b: '2', y: 1.4 },
  { b: 'B', y: 3.6 },
]
const MID = 1.8

const nodes: DiagramNode[] = [
  { id: 'D', x: 0, y: MID, w: 2, h: 1, label: 'training set\n$\\Dcal$, $n$ points', tone: 'neutral' },
  { id: 'fan', x: 1.6, y: MID, shape: 'dot' },
  { id: 'vd1', x: 3.6, y: 2.55, shape: 'text', w: 0.8, label: '$\\vdots$' },
  { id: 'vd2', x: 7.2, y: 2.55, shape: 'text', w: 0.8, label: '$\\vdots$' },
  { id: 'avg', x: 10.2, y: MID, shape: 'op', label: '$\\Sigma$' },
  { id: 'out', x: 12, y: MID, shape: 'text', w: 2.2, label: 'average or\nmajority vote' },
]
const edges: DiagramEdge[] = [
  { from: 'D', to: 'fan', arrow: 'none' },
  { from: 'avg', to: 'out' },
]
for (const { b, y } of ROWS) {
  nodes.push(
    { id: `s${b}`, x: 3.6, y, w: 2.2, h: 0.8, label: `bootstrap $\\Dcal^{*${b}}$`, tone: 0 },
    {
      id: `t${b}`,
      x: 7.2,
      y,
      w: 3.2,
      h: 0.9,
      label: `tree $\\hat f^{*${b}}$\nsplit on $m$ random features`,
      tone: 2,
    },
  )
  edges.push(
    { from: 'fan', to: `s${b}:w`, via: [[1.6, y]] },
    { from: `s${b}`, to: `t${b}` },
    { from: `t${b}:e`, to: 'avg', via: [[10.2, y]] },
  )
}

const spec: DiagramSpec = { unit: 38, spread: [1.2, 1], nodes, edges }

/** Random forest: bagging plus random feature subsets at every split. */
export function ForestDiagram() {
  return (
    <Interactive
      title="Growing a random forest"
      caption="Each tree is grown deep on its own bootstrap sample of the n training points. At every split the tree may choose only among m features drawn at random from the d available, which makes the trees less alike. The forest averages the trees' predictions for regression and takes a vote for classification. The points left out of a bootstrap sample give that tree's out-of-bag error."
    >
      <Diagram
        spec={spec}
        ariaLabel="The training set is resampled into B bootstrap samples; each grows a tree that splits on random feature subsets; the trees' predictions are averaged or voted"
      />
    </Interactive>
  )
}
