import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 40,
  nodes: [
    { id: 'DL', x: 0, y: 0, w: 2.2, h: 0.9, label: 'labelled set $\\Dcal_L$', tone: 'neutral' },
    { id: 'train', x: 0, y: 1.8, w: 2.6, h: 0.9, label: 'training set', tone: 'neutral' },
    { id: 'fit', x: 3.8, y: 1.8, w: 2.2, h: 0.9, label: 'fit model', tone: 0 },
    { id: 'DU', x: 7.4, y: 0, w: 2.4, h: 0.9, label: 'unlabelled $\\xvec_u$', tone: 'neutral' },
    { id: 'pred', x: 7.4, y: 1.8, w: 2.8, h: 0.9, label: 'predict $\\hat p(y \\mid \\xvec_u)$', tone: 0 },
    {
      id: 'test',
      x: 7.4,
      y: 3.7,
      shape: 'pill',
      w: 3.2,
      h: 0.9,
      label: '$\\max_y \\hat p(y \\mid \\xvec_u) \\ge \\tau$?',
      tone: 2,
    },
    { id: 'add', x: 0, y: 3.7, w: 3.6, h: 0.9, label: 'add pseudo-labelled $(\\xvec_u, \\hat y_u)$', tone: 2 },
    { id: 'skip', x: 11.4, y: 3.7, shape: 'text', w: 2.4, label: 'stays unlabelled\nthis round' },
  ],
  edges: [
    { from: 'DL', to: 'train' },
    { from: 'train', to: 'fit' },
    { from: 'fit', to: 'pred' },
    { from: 'DU', to: 'pred' },
    { from: 'pred', to: 'test' },
    { from: 'test', to: 'add', label: 'yes' },
    { from: 'test', to: 'skip', label: 'no', dashed: true },
    { from: 'add:n', to: 'train:s' },
  ],
}

/** The self-training loop. */
export function SelfTrainingLoop() {
  return (
    <Figure
      title="The self-training loop"
      caption="The model is fitted to the training set, which starts as the labelled set. It predicts every unlabelled point, and the points whose largest class probability reaches the threshold τ join the training set with their predicted class as a pseudo-label. The model is refitted and the loop repeats until no new point qualifies."
    >
      <Diagram
        spec={spec}
        ariaLabel="The labelled set forms the training set; a model is fitted and predicts the unlabelled points; confident predictions are added to the training set as pseudo-labels and the model is refitted"
      />
    </Figure>
  )
}
