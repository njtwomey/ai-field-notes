import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'

const K = 5
const ROW = 0.9
const COL = 1.5
const X0 = 1.6
const SUM_X = X0 + K * COL + 1.6
const MID = ((K - 1) * ROW) / 2

const nodes: DiagramNode[] = []
const edges: DiagramEdge[] = []
for (let k = 0; k < K; k++) {
  const y = k * ROW
  nodes.push({ id: `r${k}`, x: 0, y, shape: 'text', small: true, w: 1.6, label: `fit $\\hat f^{-${k + 1}}$` })
  for (let j = 0; j < K; j++) {
    const held = j === k
    nodes.push({
      id: `c${k}${j}`,
      x: X0 + j * COL,
      y,
      w: 1.3,
      h: 0.6,
      small: true,
      label: held ? 'predict' : 'train',
      tone: held ? 1 : 'neutral',
    })
  }
  nodes.push({ id: `l${k}`, x: X0 + K * COL, y, shape: 'text', small: true, w: 1, label: `$\\text{loss}_${k + 1}$` })
  edges.push({
    from: `l${k}:e`,
    to: 'sum:w',
    via: [
      [SUM_X - 0.8, y],
      [SUM_X - 0.8, MID],
    ],
    arrow: k === 2 ? 'end' : 'none',
  })
}
nodes.push(
  { id: 'sum', x: SUM_X, y: MID, shape: 'op', label: '$\\Sigma$' },
  { id: 'cv', x: SUM_X + 2.4, y: MID, shape: 'text', w: 3.4, label: '$\\operatorname{CV}$ = mean over all $n$' },
)
for (let j = 0; j < K; j++)
  nodes.push({
    id: `f${j}`,
    x: X0 + j * COL,
    y: -0.65,
    shape: 'text',
    small: true,
    w: 1.2,
    h: 0.4,
    label: `fold ${j + 1}`,
  })
edges.push({ from: 'sum', to: 'cv' })

const spec: DiagramSpec = { unit: 38, nodes, edges }

/** Five-fold cross-validation: each fold is held out once. */
export function KFoldDiagram() {
  return (
    <Interactive
      title="Five-fold cross-validation"
      caption="Each row is one fit. The model is trained on four folds and predicts the fifth; every point is predicted exactly once, by a model that did not see it. The cross-validation error is the mean loss over all n held-out predictions."
    >
      <Diagram
        spec={spec}
        ariaLabel="A five-by-five grid: in row k, fold k is held out and predicted while the other four folds are used for training; the five fold losses are combined into the cross-validation error"
      />
    </Interactive>
  )
}
