import { Diagram, Figure } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const action = (id: string, x: number, label: string) => ({
  id,
  x,
  y: 9,
  w: 1.5,
  h: 0.6,
  label,
  tone: 1 as const,
})

const spec: DiagramSpec = {
  unit: 34,
  nodes: [
    action('a1', 1, '$\\avec_{T-M+1}$'),
    { id: 'adots', x: 2.6, y: 9, shape: 'text', label: '$\\cdots$' },
    action('a2', 4.2, '$\\avec_{T-1}$'),
    action('a3', 6.2, '$\\avec_T$'),
    { id: 'feat', x: 3.6, y: 10.1, shape: 'text', small: true, label: 'PinSage, action type, surface, duration, time' },
    {
      id: 'proj',
      x: 3.6,
      y: 7.7,
      w: 6.8,
      h: 0.7,
      label: 'project, add learned position embedding',
      tone: 'neutral',
      small: true,
    },
    { id: 'tf', x: 3.6, y: 6.1, w: 6.8, h: 1.2, shape: 'stack', label: 'pre-norm transformer, causal mask', tone: 0 },
    { id: 'mlp', x: 3.6, y: 4.5, w: 6.8, h: 0.7, label: 'MLP, divide by $\\norm{\\cdot}_2$', tone: 0, small: true },
    { id: 'e1', x: 1, y: 3.2, w: 1.5, h: 0.6, label: '$\\evec_{T-M+1}$', tone: 0 },
    { id: 'edots', x: 2.6, y: 3.2, shape: 'text', label: '$\\cdots$' },
    { id: 'e2', x: 4.2, y: 3.2, w: 1.5, h: 0.6, label: '$\\evec_{T-1}$', tone: 0 },
    { id: 'e3', x: 6.2, y: 3.2, w: 1.5, h: 0.6, label: '$\\evec_T$', tone: 0 },
    { id: 'loss', x: 3.6, y: 1.6, w: 4.6, h: 0.8, label: 'sampled softmax, logQ correction', tone: 'ink', small: true },
    { id: 'pos', x: 10.6, y: 3.2, w: 2.8, h: 0.9, label: 'positives in the\nnext 28 days', tone: 2, small: true },
    {
      id: 'neg',
      x: 10.6,
      y: 1.6,
      w: 2.8,
      h: 0.9,
      label: 'in-batch and\nrandom negatives',
      tone: 'neutral',
      small: true,
      dashed: true,
    },
    { id: 'pmlp', x: 10.6, y: 5.3, w: 2.8, h: 0.8, label: 'pin MLP, $\\norm{\\cdot}_2$', tone: 2, small: true },
    { id: 'ps', x: 10.6, y: 7, w: 2.8, h: 0.7, label: 'PinSage of pin', tone: 'neutral', small: true },
    { id: 'join', x: 8.4, y: 1.6, shape: 'dot' },
    {
      id: 'lab',
      x: 3.6,
      y: 0.5,
      shape: 'text',
      small: true,
      label: 'each sampled position predicts one random future positive',
    },
  ],
  edges: [
    { from: 'a1', to: 'proj:s', via: [[1, 8.2]] },
    { from: 'a2', to: 'proj:s', via: [[4.2, 8.2]] },
    { from: 'a3', to: 'proj:s', via: [[6.2, 8.2]] },
    { from: 'proj', to: 'tf' },
    { from: 'tf', to: 'mlp' },
    { from: 'mlp:n', to: 'e1:s', via: [[1, 4.1]] },
    { from: 'mlp:n', to: 'e2:s', via: [[4.2, 4.1]] },
    { from: 'mlp:n', to: 'e3:s', via: [[6.2, 4.1]] },
    { from: 'e1:n', to: 'loss:w', via: [[1, 1.6]], dashed: true },
    {
      from: 'e3:n',
      to: 'loss:s',
      via: [
        [6.2, 2.5],
        [3.6, 2.5],
      ],
      dashed: true,
    },
    { from: 'ps', to: 'pmlp' },
    { from: 'pmlp', to: 'pos' },
    { from: 'pos:w', to: 'join', via: [[8.4, 3.2]], arrow: 'none' },
    { from: 'neg:w', to: 'join', arrow: 'none' },
    { from: 'join', to: 'loss:e' },
  ],
}

/** PinnerFormer: a causal transformer over a user's actions, trained so each position predicts long-horizon positives. */
export function PinnerFormerDiagram() {
  return (
    <Figure
      title="PinnerFormer"
      caption="The user's most recent actions, each described by the pin's PinSage embedding and metadata, pass through a causally masked transformer that outputs a unit-length embedding at every position. Pins are embedded by a small MLP on their PinSage vector. The dense all-action loss pairs sampled positions with a random positive engagement from the following 28 days and contrasts it with negatives. At serving time only the last embedding is used, computed once a day."
    >
      <Diagram
        spec={spec}
        ariaLabel="PinnerFormer: action features projected and passed through a causal transformer to per-position user embeddings, trained with a sampled softmax against future positives and negatives embedded by a pin MLP"
      />
    </Figure>
  )
}
