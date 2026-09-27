import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  unit: 40,
  spread: [1.15, 1],
  nodes: [
    { id: 'src', x: 0, y: 0, w: 2.6, h: 0.9, label: 'stream: $\\xvec_t \\sim \\Dcal$', tone: 'neutral' },
    { id: 'q', x: 4.4, y: 0, shape: 'pill', w: 3.4, h: 1, label: 'query rule\nuncertain or disputed?', tone: 2 },
    { id: 'ask', x: 8.8, y: 0, w: 2.8, h: 0.9, label: 'ask oracle for $y_t$', tone: 1 },
    { id: 'upd', x: 8.8, y: 2.2, w: 3.4, h: 0.9, label: 'update model / version space $V_{t+1}$', tone: 0 },
    { id: 'inf', x: 4.4, y: 2.2, w: 3, h: 0.9, label: 'predict $\\hat y_t$, discard $\\xvec_t$', tone: 'neutral' },
    { id: 'next', x: 0, y: 2.2, shape: 'pill', w: 2.6, h: 0.8, label: 'next round $t + 1$', tone: 'neutral' },
  ],
  edges: [
    { from: 'src', to: 'q' },
    { from: 'q', to: 'ask', label: 'yes' },
    { from: 'ask', to: 'upd' },
    { from: 'q', to: 'inf', label: 'no', labelSide: 'right', labelRotate: false },
    {
      from: 'upd:s',
      to: 'next:s',
      via: [
        [8.8, 3.3],
        [0, 3.3],
      ],
      dashed: true,
    },
    { from: 'inf', to: 'next', dashed: true },
    { from: 'next', to: 'src', dashed: true },
  ],
}

/** One round of stream-based active learning. */
export function StreamQueryDiagram() {
  return (
    <Interactive
      title="One round of stream-based active learning"
      caption="Each unlabelled example is seen once. The query rule decides on the spot whether its label is worth paying for: when the model is uncertain, a committee disagrees, or the example lies in the region of disagreement. A queried label updates the model; an unqueried example is labelled by the model and then discarded."
    >
      <Diagram
        spec={spec}
        ariaLabel="An example arrives from the stream; a query rule decides whether to ask the oracle for its label; if yes, the label updates the model; if no, the model predicts and the example is discarded; the next round begins"
      />
    </Interactive>
  )
}
