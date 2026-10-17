import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  nodes: [
    { id: 'x', x: 0, y: 2.3, shape: 'text', w: 0.8, label: 'image' },
    { id: 'j', x: 1.1, y: 2.3, shape: 'dot' },
    { id: 'f', x: 4, y: 1, w: 2.2, h: 0.8, label: 'encoder $f_{\\thetavec}$', tone: 0 },
    { id: 'g', x: 6.8, y: 1, w: 2.4, h: 0.8, label: 'projector $g_{\\thetavec}$', tone: 0 },
    { id: 'q', x: 9.7, y: 1, w: 2.4, h: 0.8, label: 'predictor $q_{\\thetavec}$', tone: 0 },
    { id: 'fx', x: 4, y: 3.6, w: 2.2, h: 0.8, label: 'encoder $f_{\\xivec}$', tone: 1 },
    { id: 'gx', x: 6.8, y: 3.6, w: 2.4, h: 0.8, label: 'projector $g_{\\xivec}$', tone: 1 },
    { id: 'sg', x: 9.7, y: 3.6, shape: 'pill', w: 2.4, h: 0.7, label: 'stop-gradient', tone: 'neutral', small: true },
    { id: 'L', x: 13.2, y: 2.3, w: 2.8, h: 1, label: 'loss\n$2 - 2\\cos(\\cdot, \\cdot)$', tone: 'neutral' },
  ],
  edges: [
    { from: 'x', to: 'j', arrow: 'none' },
    { from: 'j', to: 'f:w', via: [[1.1, 1]], label: 'view $A$' },
    { from: 'j', to: 'fx:w', via: [[1.1, 3.6]], label: 'view $B$', labelSide: 'right' },
    { from: 'f', to: 'g' },
    { from: 'g', to: 'q', label: '$\\zvec_A$' },
    { from: 'q:e', to: 'L:n', via: [[13.2, 1]] },
    { from: 'fx', to: 'gx' },
    { from: 'gx', to: 'sg', label: "$\\zvec'_B$", labelSide: 'right' },
    { from: 'sg:e', to: 'L:s', via: [[13.2, 3.6]] },
    {
      from: 'f',
      to: 'fx',
      dashed: true,
      label: 'moving average $\\xivec \\leftarrow \\beta\\xivec + (1 - \\beta)\\thetavec$',
      labelRotate: false,
      labelSide: 'left',
    },
  ],
  groups: [
    { id: 'on', label: 'online: trained by gradient', tone: 0, around: ['f', 'q'], pad: 0.3 },
    { id: 'tg', label: 'target: no gradient', tone: 1, around: ['fx', 'sg'], pad: 0.3, labelAt: 'bottom-left' },
  ],
}

/** BYOL's asymmetry: a predictor on one side, a slowly moving copy with no gradient on the other. */
export function ByolDiagram() {
  return (
    <Figure
      title="BYOL"
      caption="Two augmented views of one image go through two networks. The online network, with an extra predictor, is trained to predict the target network's projection of the other view. No gradient flows into the target; its weights follow the online weights as a moving average. There are no negative pairs."
    >
      <Diagram
        spec={spec}
        ariaLabel="BYOL: view A through the online encoder, projector and predictor; view B through the target encoder and projector with a stop-gradient; loss compares the two; target weights are a moving average of the online weights"
      />
    </Figure>
  )
}
