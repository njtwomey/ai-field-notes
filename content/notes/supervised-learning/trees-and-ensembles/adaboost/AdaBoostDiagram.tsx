import { Interactive } from 'aifn-render'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'

const ROUNDS: { m: string; x: number }[] = [
  { m: '1', x: 0 },
  { m: '2', x: 5 },
  { m: 'M', x: 11.4 },
]
const COMBINE_X = 5.7

const nodes: DiagramNode[] = []
const edges: DiagramEdge[] = []
ROUNDS.forEach(({ m, x }, r) => {
  nodes.push(
    {
      id: `w${r}`,
      x,
      y: 0,
      w: 2.4,
      h: 0.7,
      label: r === 0 ? 'weights $w_i = 1/n$' : `weights $w_i^{(${m})}$`,
      tone: 'neutral',
    },
    { id: `g${r}`, x, y: 1.5, w: 2.4, h: 0.7, label: `fit $G_${m}$`, tone: 0 },
    {
      id: `a${r}`,
      x,
      y: 3,
      w: 2.4,
      h: 0.9,
      label: `$\\text{err}_${m}$, vote\n$\\alpha_${m} = \\log\\frac{1 - \\text{err}_${m}}{\\text{err}_${m}}$`,
      tone: 'neutral',
    },
  )
  edges.push(
    { from: `w${r}`, to: `g${r}` },
    { from: `g${r}`, to: `a${r}` },
    {
      from: `a${r}:s`,
      to: 'comb:n',
      via: [
        [x, 4.3],
        [COMBINE_X, 4.3],
      ],
      arrow: r === 1 ? 'end' : 'none',
    },
  )
})
nodes.push(
  { id: 'dots', x: 8.3, y: 0, shape: 'text', w: 0.8, label: '$\\cdots$' },
  {
    id: 'comb',
    x: COMBINE_X,
    y: 5.4,
    w: 4,
    h: 0.9,
    label: 'final classifier $\\sgn\\big(\\sum_{m} \\alpha_m G_m(\\xvec)\\big)$',
    tone: 1,
  },
)
edges.push(
  {
    from: 'a0:e',
    to: 'w1:w',
    via: [
      [2.5, 3],
      [2.5, 0],
    ],
    label: 'errors $\\times e^{\\alpha_1}$',
    labelSide: 'right',
    tone: 1,
  },
  {
    from: 'a1:e',
    to: 'dots:w',
    via: [
      [7.3, 3],
      [7.3, 0],
    ],
    tone: 1,
  },
  { from: 'dots:e', to: 'w2:w', tone: 1 },
)

const spec: DiagramSpec = { unit: 38, nodes, edges }

/** The rounds of AdaBoost: fit, score, reweight, and a weighted vote at the end. */
export function AdaBoostDiagram() {
  return (
    <Interactive
      title="The rounds of AdaBoost"
      caption="Each round fits a weak learner to the currently weighted data, scores it by its weighted error and gives it the vote α_m. The weights of the points it misclassifies are multiplied by e^α_m before the next round, so the next learner concentrates on them. The final classifier is the sign of the α-weighted vote."
    >
      <Diagram
        spec={spec}
        ariaLabel="Rounds 1, 2 to M: each round fits a weak learner to weighted data and computes its error and vote; misclassified points are up-weighted for the next round; all learners are combined by a weighted vote"
      />
    </Interactive>
  )
}
