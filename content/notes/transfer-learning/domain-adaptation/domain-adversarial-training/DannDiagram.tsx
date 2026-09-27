import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  unit: 38,
  nodes: [
    { id: 'x', x: 0, y: 2, shape: 'circle', label: '$\\xvec$', tone: 'ink', filled: true },
    {
      id: 'gf',
      x: 2.6,
      y: 2,
      shape: 'encoder',
      w: 2,
      h: 2.4,
      label: 'feature\nextractor\n$G_f(\\cdot;\\thetavec_f)$',
      tone: 0,
    },
    { id: 'f', x: 5, y: 2, shape: 'circle', label: '$\\fvec$', tone: 'ink' },
    { id: 'gy', x: 8, y: 0.4, w: 2.4, h: 1.1, label: 'label predictor\n$G_y(\\cdot;\\thetavec_y)$', tone: 1 },
    { id: 'ly', x: 11, y: 0.4, shape: 'pill', w: 1.6, h: 0.8, label: '$L_y$', tone: 'neutral' },
    { id: 'grl', x: 6.6, y: 3.6, w: 1.4, h: 0.8, label: 'GRL', tone: 'ink' },
    { id: 'gd', x: 9.3, y: 3.6, w: 2.4, h: 1.1, label: 'domain classifier\n$G_d(\\cdot;\\thetavec_d)$', tone: 2 },
    { id: 'ld', x: 12.2, y: 3.6, shape: 'pill', w: 1.6, h: 0.8, label: '$L_d$', tone: 'neutral' },
    { id: 'n1', x: 8, y: -0.7, shape: 'text', small: true, label: 'source examples only' },
    { id: 'n2', x: 9.3, y: 4.8, shape: 'text', small: true, label: 'source and target examples' },
    {
      id: 'n3',
      x: 6.6,
      y: 4.7,
      shape: 'text',
      small: true,
      label: 'forward: identity\nbackward: $\\times(-\\lambda)$',
    },
  ],
  edges: [
    { from: 'x', to: 'gf' },
    { from: 'gf', to: 'f' },
    { from: 'f:n', to: 'gy:w', via: [[5, 0.4]] },
    { from: 'gy', to: 'ly', label: '$\\hat y$' },
    { from: 'f:s', to: 'grl:w', via: [[5, 3.6]] },
    { from: 'grl', to: 'gd' },
    { from: 'gd', to: 'ld', label: '$\\hat d$' },
  ],
}

/** The DANN architecture: a shared feature extractor feeding a label predictor and, through a gradient reversal layer, a domain classifier. */
export function DannDiagram() {
  return (
    <Interactive
      title="Domain-adversarial neural network"
      caption="The label predictor is trained on labelled source examples. The domain classifier is trained on features of source and target examples to tell them apart. The gradient reversal layer (GRL) passes features forward unchanged and multiplies the domain loss gradient by −λ on the way back, so the feature extractor is pushed to make the two domains indistinguishable while keeping the labels predictable."
    >
      <Diagram
        spec={spec}
        ariaLabel="Input x to feature extractor to features f; f to label predictor with loss L_y; f through gradient reversal layer to domain classifier with loss L_d"
      />
    </Interactive>
  )
}
