import { MathText } from '@/components/content/MathText'
import { GraphDiagram, Interactive } from '@/components/viz'
import { at, chainEnds, labelNodes } from '../_shared/chain-graph'

const ends = chainEnds(0, 'y0', 'y2', true)

/** The MEMM: directed edges from the previous label and from the whole observation sequence into each label. */
export function MemmGraph() {
  return (
    <Interactive
      title="The MEMM as a graph"
      caption={
        <MathText text="Each label $y_n$ has two parents: the previous label $y_{n-1}$ and the observation sequence $\xvec$ (shaded). Each node's conditional is one locally normalised classifier." />
      }
    >
      <GraphDiagram
        nodes={[...labelNodes(0), { id: 'x', label: '\\mathbf{x}', x: at(1), y: 1.2, kind: 'observed' }, ...ends.nodes]}
        edges={[
          { source: 'y0', target: 'y1' },
          { source: 'y1', target: 'y2' },
          ...[0, 1, 2].map((i) => ({ source: 'x', target: `y${i}` })),
          ...ends.edges,
        ]}
        height={190}
        ariaLabel="Maximum-entropy Markov model chain"
      />
    </Interactive>
  )
}
