import { Interactive } from '@/components/viz'
import { Diagram } from '@/components/diagram/Diagram'
import { op } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'

const spec: DiagramSpec = {
  nodes: [
    { id: 'docs', x: 0, y: 1, shape: 'stack', w: 2.6, h: 0.9, label: 'document collection', tone: 'neutral' },
    { id: 'denc', x: 3.8, y: 1, w: 2.8, h: 0.9, label: 'passage encoder $\\dvec(z)$', tone: 0 },
    { id: 'index', x: 7.4, y: 1, shape: 'stack', w: 2.6, h: 0.9, label: 'vector index', tone: 'neutral' },
    { id: 'x', x: 0, y: 3.6, w: 1.6, h: 0.8, label: 'input $x$', tone: 'neutral' },
    { id: 'qenc', x: 3.8, y: 3.6, w: 2.8, h: 0.9, label: 'query encoder $\\qvec(x)$', tone: 0 },
    { id: 'topk', x: 7.4, y: 3.6, w: 2.6, h: 1, label: 'top $k$ by\n$\\dvec(z)\\transpose \\qvec(x)$', tone: 0 },
    { id: 'gen', x: 11.8, y: 3.6, w: 3, h: 1, label: 'generator\n$p_\\theta(y \\mid x, z)$', tone: 1 },
    op('sum', 14.4, 3.6, '$\\Sigma$'),
    { id: 'y', x: 15.5, y: 3.6, shape: 'text', w: 0.6, label: '$y$' },
    { id: 'suml', x: 14.4, y: 4.4, shape: 'text', small: true, label: 'sum over $z$' },
  ],
  edges: [
    { from: 'docs', to: 'denc' },
    { from: 'denc', to: 'index' },
    { from: 'x', to: 'qenc' },
    { from: 'qenc', to: 'topk' },
    { from: 'index', to: 'topk', label: 'search', labelRotate: false, labelSide: 'right' },
    { from: 'topk', to: 'gen', label: '$z$, $p_\\eta(z \\mid x)$' },
    { from: 'gen', to: 'sum' },
    { from: 'sum', to: 'y' },
    {
      from: 'x:s',
      to: 'gen:s',
      via: [
        [0, 5.2],
        [11.8, 5.2],
      ],
      label: 'the input itself',
      labelSide: 'right',
    },
  ],
  groups: [
    {
      id: 'off',
      label: 'offline, once per collection',
      tone: 'neutral',
      dashed: true,
      around: ['docs', 'index'],
      pad: 0.35,
    },
    { id: 'ret', label: 'retriever', tone: 0, around: ['qenc', 'topk'], pad: 0.3, labelAt: 'bottom-left' },
  ],
}

/** Retrieval then generation: the passages are fetched by embedding similarity and marginalised out of the output. */
export function RagDiagram() {
  return (
    <Interactive
      title="Retrieval-augmented generation"
      caption="Offline, every passage is embedded and stored in a vector index. Per request, the query encoder embeds the input and the index returns the k passages with the largest inner products. The generator reads the input together with each passage, and the output probability is summed over the passages, weighted by the retriever."
    >
      <Diagram
        spec={spec}
        ariaLabel="RAG pipeline: documents encoded offline into a vector index; the input is encoded, top-k passages retrieved, and a generator conditioned on input and passage produces y, marginalised over passages"
      />
    </Interactive>
  )
}
