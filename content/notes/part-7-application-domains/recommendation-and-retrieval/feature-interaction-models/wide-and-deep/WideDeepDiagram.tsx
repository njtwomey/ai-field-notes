import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const spec: DiagramSpec = {
  unit: 36,
  nodes: [
    { id: 'sparse', x: 3.5, y: 8, w: 6.5, h: 0.7, label: 'sparse features (one-hot IDs, categories)', tone: 'neutral' },
    { id: 'dense', x: 9.6, y: 8, w: 2.8, h: 0.7, label: 'dense features', tone: 'neutral' },
    { id: 'cross', x: 1.6, y: 5.6, w: 2.8, h: 0.8, label: 'cross products\n$\\phi(\\xvec)$', tone: 1 },
    { id: 'emb', x: 5.6, y: 6.6, w: 3.4, h: 0.7, label: 'embeddings', tone: 0 },
    { id: 'cat', x: 7.6, y: 5.4, w: 4.4, h: 0.6, label: 'concatenate', tone: 'neutral' },
    { id: 'mlp', x: 7.6, y: 3.6, shape: 'stack', w: 3, h: 1.1, label: 'ReLU layers', tone: 0 },
    { id: 'wide', x: 1.6, y: 3.6, w: 2.8, h: 0.8, label: 'linear\n$\\wvec_{\\text{wide}}$', tone: 1 },
    { id: 'sum', x: 4.6, y: 1.8, shape: 'op', label: '$+$' },
    { id: 'out', x: 4.6, y: 0.6, shape: 'pill', w: 2, h: 0.7, label: '$\\sigma$: $P(y = 1)$', tone: 'ink' },
    { id: 'wl', x: 1.6, y: 2.4, shape: 'text', small: true, label: 'wide: memorise' },
    { id: 'dl', x: 7.6, y: 2.4, shape: 'text', small: true, label: 'deep: generalise' },
  ],
  edges: [
    { from: 'sparse:n', to: 'cross:s', via: [[1.6, 7.2]] },
    { from: 'sparse:n', to: 'emb:s', via: [[5.6, 7.2]] },
    { from: 'emb:n', to: 'cat:s', via: [[5.6, 6]] },
    { from: 'dense:n', to: 'cat:s', via: [[9.6, 6]] },
    { from: 'cat', to: 'mlp' },
    { from: 'cross', to: 'wide' },
    { from: 'wide:n', to: 'sum:w', via: [[1.6, 1.8]] },
    { from: 'mlp:n', to: 'sum:e', via: [[7.6, 1.8]] },
    { from: 'sum', to: 'out' },
  ],
}

/** Wide & Deep: a linear model on crossed sparse features and an MLP on embeddings, summed before one sigmoid. */
export function WideDeepDiagram() {
  return (
    <Interactive
      title="Wide & Deep"
      caption="The wide part is a linear model over raw and crossed sparse features; the deep part embeds the sparse features, concatenates them with dense features and passes them through ReLU layers. Their logits are added and trained jointly through one logistic loss."
    >
      <Diagram
        spec={spec}
        ariaLabel="Wide and Deep architecture: cross-product features into a linear model, embeddings into an MLP, summed into a sigmoid"
      />
    </Interactive>
  )
}
