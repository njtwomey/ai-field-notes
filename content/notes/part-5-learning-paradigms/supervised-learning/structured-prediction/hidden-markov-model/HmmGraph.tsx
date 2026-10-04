import { Diagram, Figure, link, setting, useFigureState, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { POSITIONS, at, chainEnds, labelNodes, potentialChain } from '../_shared/chain-graph'

function directedChain(): DiagramSpec {
  const xs = POSITIONS.map((p, i) => variable(`x${i}`, at(i), 1.8, `$x_{${p}}$`, { w: 1.05, h: 1.05, filled: true }))
  const ends = chainEnds(0, 'y0', 'y2', true)
  return {
    nodes: [...labelNodes(0), ...xs, ...ends.nodes],
    edges: [link('y0', 'y1'), link('y1', 'y2'), ...[0, 1, 2].map((i) => link(`y${i}`, `x${i}`)), ...ends.edges],
  }
}

const DIRECTED = directedChain()
const POTENTIALS = potentialChain()

/** The HMM as a directed graph, and the same chain in potential form with the observations absorbed. */
export function HmmGraph() {
  const state = useFigureState({
    potentials: setting(false, 'potential form'),
  })
  const g = state.potentials ? POTENTIALS : DIRECTED
  return (
    <Figure
      title="The HMM as a graph"
      state={state}
      caption={
        state.potentials
          ? 'Potential form: each square is a factor. The node potential below each state absorbs its emission, and the edge potential between two states is the transition matrix.'
          : 'Each hidden state depends on the one before it and emits one observation. Shaded nodes are observed.'
      }
    >
      <Diagram spec={{ ...g, unit: 48 }} ariaLabel="Hidden Markov model chain" />
    </Figure>
  )
}
