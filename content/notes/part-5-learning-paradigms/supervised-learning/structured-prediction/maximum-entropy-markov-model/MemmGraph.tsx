import { Diagram, Figure, link, MathText, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { at, chainEnds, labelNodes } from '../_shared/chain-graph'

const ends = chainEnds(0, 'y0', 'y2', true)

const spec: DiagramSpec = {
  unit: 48,
  nodes: [...labelNodes(0), variable('x', at(1), 2, '$\\xvec$', { w: 1.05, h: 1.05, filled: true }), ...ends.nodes],
  edges: [link('y0', 'y1'), link('y1', 'y2'), ...[0, 1, 2].map((i) => link('x', `y${i}`)), ...ends.edges],
}

/** The MEMM: directed edges from the previous label and from the whole observation sequence into each label. */
export function MemmGraph() {
  return (
    <Figure
      title="The MEMM as a graph"
      caption={
        <MathText text="Each label $y_n$ has two parents: the previous label $y_{n-1}$ and the observation sequence $\xvec$ (shaded). Each node's conditional is one locally normalised classifier." />
      }
    >
      <Diagram spec={spec} ariaLabel="Maximum-entropy Markov model chain" />
    </Figure>
  )
}
