import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import type { DiagramEdge, DiagramNode, DiagramSpec } from '@/components/diagram/types'
import { Interactive } from 'aifn-render'

const node = (id: string, x: number, y: number, label: string, tone: number | 'neutral'): DiagramNode => ({
  id,
  x,
  y,
  w: 3.2,
  h: 0.9,
  label,
  tone,
  small: true,
})
const edge = (from: string, to: string, docs: number): DiagramEdge => ({
  from: `${from}:s`,
  to: `${to}:n`,
  label: `${docs}`,
  labelRotate: false,
})

/** A three-level topic tree with the number of documents whose path passes through each edge. */
const spec: DiagramSpec = {
  nodes: [
    node('root', 7.4, 0, 'the, of, a, is', 'neutral'),
    node('bio', 1.8, 2.4, 'cell, gene, protein', 0),
    node('neuro', 7.4, 2.4, 'neuron, brain, cortex', 1),
    node('astro', 11.6, 2.4, 'star, galaxy, orbit', 2),
    node('dna', 0, 4.8, 'DNA, sequence, genome', 0),
    node('immune', 3.7, 4.8, 'antigen, T cell, immune', 0),
    node('vision', 7.4, 4.8, 'visual, retina, stimulus', 1),
    node('exo', 11.6, 4.8, 'planet, transit, host', 2),
    { id: 'newRoot', x: 14.6, y: 2.4, shape: 'text', small: true, label: 'new: $\\gamma = 1$' },
    { id: 'newBio', x: 1.8, y: 6.3, shape: 'text', small: true, label: 'new below cell/gene: $\\gamma = 1$' },
  ],
  edges: [
    edge('root', 'bio', 5),
    edge('root', 'neuro', 3),
    edge('root', 'astro', 1),
    edge('bio', 'dna', 4),
    edge('bio', 'immune', 1),
    edge('neuro', 'vision', 3),
    edge('astro', 'exo', 1),
  ],
}

export function TopicTree() {
  return (
    <Interactive
      title="A topic tree grown by the nested Chinese restaurant process"
      caption={
        <MathText text="Each box is a topic, shown by its most probable words. Edge labels count the documents whose root-to-leaf path uses that edge. The root topic holds words common to every document; deeper topics are more specific. A new document picks a child in proportion to these counts, or a new child in proportion to $\gamma$, at every level." />
      }
    >
      <Diagram spec={spec} ariaLabel="A three-level topic hierarchy with document counts on its edges" />
    </Interactive>
  )
}
