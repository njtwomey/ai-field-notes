import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'

const CLIENTS: { k: string; x: number }[] = [
  { k: '1', x: 0 },
  { k: '2', x: 4 },
  { k: 'm', x: 10 },
]

const nodes: DiagramNode[] = [
  { id: 'srv', x: 5, y: 0, w: 3.6, h: 0.9, label: 'server holds $\\wvec_t$', tone: 0 },
  { id: 'bus1', x: 5, y: 1.2, shape: 'dot' },
  { id: 'dots', x: 7, y: 2.6, shape: 'text', w: 0.8, label: '$\\cdots$' },
  {
    id: 'agg',
    x: 5,
    y: 5.2,
    w: 4.6,
    h: 0.9,
    label: '$\\wvec_{t+1} = \\sum_{k \\in S} \\frac{n_k}{n_S}\\, \\wvec^k_{t+1}$',
    tone: 0,
  },
]
const edges: DiagramEdge[] = [
  { from: 'srv', to: 'bus1', arrow: 'none', label: 'send $\\wvec_t$', labelSide: 'right', labelRotate: false },
]
for (const { k, x } of CLIENTS) {
  nodes.push({
    id: `c${k}`,
    x,
    y: 2.6,
    w: 2.8,
    h: 1.1,
    label: `client $${k}$\n$E$ epochs of local SGD`,
    tone: 'neutral',
  })
  edges.push(
    // Shared segments overlap exactly, so the fan-out and fan-in read as one bus each.
    { from: 'bus1', to: `c${k}:n`, via: [[x, 1.2]] },
    {
      from: `c${k}:s`,
      to: 'agg:n',
      via: [
        [x, 4.1],
        [5, 4.1],
      ],
    },
  )
}
edges.push({
  from: 'agg:w',
  to: 'srv:w',
  via: [
    [-2.2, 5.2],
    [-2.2, 0],
  ],
  dashed: true,
  label: 'next round',
})

const spec: DiagramSpec = {
  unit: 38,
  nodes,
  edges,
}

/** One round of federated averaging. */
export function FedAvgRound() {
  return (
    <Interactive
      title="One round of federated averaging"
      caption="The server sends the current model to a sample S of clients. Each client runs E epochs of local SGD on its own data and returns its updated model; the data never leave the client. The server replaces its model by the average of the returned models, weighted by each client's number of examples n_k out of n_S in the sample."
    >
      <Diagram
        spec={spec}
        ariaLabel="The server broadcasts the model to sampled clients; each trains locally on its own data; the server averages the returned models weighted by data size and starts the next round"
      />
    </Interactive>
  )
}
