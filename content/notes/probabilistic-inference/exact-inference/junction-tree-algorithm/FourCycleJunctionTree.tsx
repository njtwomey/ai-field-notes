import { useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { GraphDiagram, Interactive, ParamChoice, Readout, type GraphEdge, type GraphNode } from '@/components/viz'

type Stage = 'cycle' | 'chordal' | 'tree'

const CYCLE_NODES: GraphNode[] = [
  { id: 'a', x: 0, y: 0 },
  { id: 'b', x: 1.4, y: 0 },
  { id: 'c', x: 1.4, y: 1.4 },
  { id: 'd', x: 0, y: 1.4 },
]
const edge = (source: string, target: string, extra: Partial<GraphEdge> = {}): GraphEdge => ({
  source,
  target,
  directed: false,
  ...extra,
})
const CYCLE_EDGES = [edge('a', 'b'), edge('b', 'c'), edge('c', 'd'), edge('d', 'a')]

const STAGES: Record<Stage, { nodes: GraphNode[]; edges: GraphEdge[]; cliques: string }> = {
  cycle: {
    nodes: CYCLE_NODES,
    edges: CYCLE_EDGES,
    cliques: '$\\{a, b\\}$, $\\{b, c\\}$, $\\{c, d\\}$, $\\{d, a\\}$',
  },
  chordal: {
    nodes: CYCLE_NODES,
    edges: [...CYCLE_EDGES, edge('a', 'c', { dashed: true, highlight: true })],
    cliques: '$C_1 = \\{a, b, c\\}$, $C_2 = \\{a, c, d\\}$',
  },
  tree: {
    nodes: [
      { id: 'C1', label: 'a, b, c', x: 0, y: 0, kind: 'cluster' },
      { id: 'C2', label: 'a, c, d', x: 2, y: 0, kind: 'cluster' },
    ],
    edges: [edge('C1', 'C2', { label: 'a, c' })],
    cliques: '$C_1 = \\{a, b, c\\}$, $C_2 = \\{a, c, d\\}$, separator $\\{a, c\\}$',
  },
}

/** The worked example's three stages: the four-cycle, its triangulation by the chord a–c, and the junction tree. */
export function FourCycleJunctionTree() {
  const [stage, setStage] = useState<Stage>('cycle')
  const s = STAGES[stage]
  return (
    <Interactive
      title="From a four-cycle to a junction tree"
      caption={
        <MathText text="The four-cycle has no junction tree over its cliques. The chord $a - c$ (dashed) makes it chordal, with two maximal cliques. The junction tree joins them through their separator $\{a, c\}$, which labels the edge." />
      }
      controls={
        <ParamChoice
          label="stage"
          value={stage}
          onChange={setStage}
          options={[
            { value: 'cycle', label: '1. four-cycle' },
            { value: 'chordal', label: '2. triangulated' },
            { value: 'tree', label: '3. junction tree' },
          ]}
        />
      }
      readout={<Readout label="maximal cliques" value={<MathText text={s.cliques} />} />}
    >
      <GraphDiagram
        nodes={s.nodes}
        edges={s.edges}
        height={stage === 'tree' ? 110 : 200}
        ariaLabel={`Stage: ${stage}`}
      />
    </Interactive>
  )
}
