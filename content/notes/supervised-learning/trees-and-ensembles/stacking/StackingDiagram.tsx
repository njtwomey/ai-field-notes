import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  unit: 38,
  spread: [1.25, 1],
  nodes: [
    { id: 'D', x: 0, y: 1.2, w: 2.2, h: 1, label: 'training set\n$n$ points, $K$ folds', tone: 'neutral' },
    { id: 'fan', x: 1.7, y: 1.2, shape: 'dot' },
    { id: 'b1', x: 4.2, y: 0, w: 3, h: 0.9, label: 'base model $\\hat f_1$\nfit on $K - 1$ folds', tone: 0 },
    { id: 'vd', x: 4.2, y: 1.25, shape: 'text', w: 0.8, label: '$\\vdots$' },
    { id: 'bL', x: 4.2, y: 2.4, w: 3, h: 0.9, label: 'base model $\\hat f_L$\nfit on $K - 1$ folds', tone: 0 },
    { id: 'Z', x: 8.2, y: 1.2, w: 2.8, h: 1, label: 'out-of-fold predictions\n$n \\times L$ matrix', tone: 'neutral' },
    { id: 'meta', x: 11.6, y: 1.2, w: 2.2, h: 1, label: 'meta-model\n$g$', tone: 1 },
    { id: 'yh', x: 13.6, y: 1.2, shape: 'text', w: 0.6, label: '$\\hat y$' },
    { id: 'x', x: 0, y: 4.4, shape: 'text', w: 1.6, label: 'new $\\xvec$' },
    {
      id: 'refit',
      x: 4.2,
      y: 4.4,
      w: 3.6,
      h: 0.9,
      label: 'base models refitted on all $n$\n$\\hat f_1(\\xvec), \\dots, \\hat f_L(\\xvec)$',
      tone: 0,
    },
  ],
  edges: [
    { from: 'D', to: 'fan', arrow: 'none' },
    { from: 'fan', to: 'b1:w', via: [[1.7, 0]] },
    { from: 'fan', to: 'bL:w', via: [[1.7, 2.4]] },
    { from: 'b1:e', to: 'Z:n', via: [[8.2, 0]], label: 'predict held-out fold' },
    { from: 'bL:e', to: 'Z:s', via: [[8.2, 2.4]] },
    { from: 'Z', to: 'meta', label: 'fit', labelOffset: 0.15 },
    { from: 'meta', to: 'yh' },
    { from: 'x', to: 'refit' },
    { from: 'refit:e', to: 'meta:s', via: [[11.6, 4.4]], label: 'at test time', dashed: true },
  ],
}

/** Stacking: out-of-fold predictions train the meta-model; refitted base models feed it at test time. */
export function StackingDiagram() {
  return (
    <Interactive
      title="Stacking"
      caption="Every base model is cross-fitted: for each fold it is trained on the other K − 1 folds and predicts the held-out fold. The n × L matrix of out-of-fold predictions and the targets y train the meta-model. At test time the base models, refitted on all the data, feed their predictions for a new x to the meta-model."
    >
      <Diagram
        spec={spec}
        ariaLabel="Base models are cross-fitted on K folds to give out-of-fold predictions, which train a meta-model; at test time refitted base models feed the meta-model"
      />
    </Interactive>
  )
}
