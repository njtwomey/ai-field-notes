import { Interactive } from 'aifn-render'
import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

const field = (i: number) => ({
  id: `f${i}`,
  x: 1.4 + 2.2 * i,
  y: 8,
  w: 1.8,
  h: 0.6,
  label: `field ${i + 1}`,
  tone: 'neutral' as const,
})
const emb = (i: number) => ({
  id: `e${i}`,
  x: 1.4 + 2.2 * i,
  y: 6.6,
  w: 1.4,
  h: 0.6,
  label: `$\\vvec_{${i + 1}}$`,
  tone: 0,
})

const spec: DiagramSpec = {
  unit: 36,
  nodes: [
    ...[0, 1, 2, 3].map(field),
    ...[0, 1, 2, 3].map(emb),
    {
      id: 'fm',
      x: 2.5,
      y: 4.4,
      w: 3.6,
      h: 1,
      label: 'FM layer\n$\\sum_i w_ix_i + \\sum_{i<j}\\inner{\\vvec_i}{\\vvec_j}x_ix_j$',
      tone: 1,
      small: true,
    },
    { id: 'mlp', x: 6.9, y: 4.4, shape: 'stack', w: 3, h: 1, label: 'hidden layers', tone: 0 },
    { id: 'sum', x: 4.7, y: 2.4, shape: 'op', label: '$+$' },
    { id: 'out', x: 4.7, y: 1.2, shape: 'pill', w: 1.8, h: 0.7, label: '$\\sigma$', tone: 'ink' },
    { id: 'shared', x: 9.9, y: 6.6, shape: 'text', small: true, label: 'shared\nembeddings' },
  ],
  edges: [
    ...[0, 1, 2, 3].map((i) => ({ from: `f${i}`, to: `e${i}` })),
    ...[0, 1, 2, 3].map((i) => ({
      from: `e${i}:n`,
      to: 'fm:s',
      via: [[1.4 + 2.2 * i, 5.5] as [number, number], [2.5, 5.5] as [number, number]],
    })),
    ...[0, 1, 2, 3].map((i) => ({
      from: `e${i}:n`,
      to: 'mlp:s',
      via: [[1.4 + 2.2 * i, 5.5] as [number, number], [6.9, 5.5] as [number, number]],
    })),
    { from: 'fm:n', to: 'sum:w', via: [[2.5, 2.4]] },
    { from: 'mlp:n', to: 'sum:e', via: [[6.9, 2.4]] },
    { from: 'sum', to: 'out' },
  ],
}

/** DeepFM: one embedding table feeding both an FM component and a deep network. */
export function DeepFmDiagram() {
  return (
    <Interactive
      title="DeepFM"
      caption="Each field's one-hot value is looked up in one embedding table. The FM component uses the embeddings for its first- and second-order terms; the deep component concatenates the same embeddings and passes them through hidden layers for higher-order interactions. The two outputs are summed before the sigmoid."
    >
      <Diagram
        spec={spec}
        ariaLabel="DeepFM: fields to shared embeddings, feeding an FM layer and an MLP whose outputs are summed into a sigmoid"
      />
    </Interactive>
  )
}
