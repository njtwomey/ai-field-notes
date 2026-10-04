import { choice, Diagram, Figure, link, MathText, Readout, useFigureState, variable } from 'aifn-render'
import type { DiagramEdge, DiagramSpec } from 'aifn-render'

type Stage = 'cycle' | 'chordal' | 'tree'

const CYCLE_NODES = [
  variable('a', 0, 0, '$a$'),
  variable('b', 2, 0, '$b$'),
  variable('c', 2, 2, '$c$'),
  variable('d', 0, 2, '$d$'),
]
const edge = (from: string, to: string, extra: Partial<DiagramEdge> = {}) => link(from, to, false, extra)
const CYCLE_EDGES = [edge('a', 'b'), edge('b', 'c'), edge('c', 'd'), edge('d', 'a')]
const clique = (id: string, x: number, label: string) => ({ id, x, y: 0, w: 1.7, h: 0.8, tone: 'ink' as const, label })

const STAGES: Record<Stage, { spec: DiagramSpec; cliques: string }> = {
  cycle: {
    spec: { unit: 56, nodes: CYCLE_NODES, edges: CYCLE_EDGES },
    cliques: '$\\{a, b\\}$, $\\{b, c\\}$, $\\{c, d\\}$, $\\{d, a\\}$',
  },
  chordal: {
    spec: { unit: 56, nodes: CYCLE_NODES, edges: [...CYCLE_EDGES, edge('a', 'c', { dashed: true, highlight: true })] },
    cliques: '$C_1 = \\{a, b, c\\}$, $C_2 = \\{a, c, d\\}$',
  },
  tree: {
    spec: {
      unit: 56,
      nodes: [clique('C1', 0, '$a, b, c$'), clique('C2', 3.4, '$a, c, d$')],
      edges: [edge('C1', 'C2', { label: '$a, c$' })],
    },
    cliques: '$C_1 = \\{a, b, c\\}$, $C_2 = \\{a, c, d\\}$, separator $\\{a, c\\}$',
  },
}

/** The worked example's three stages: the four-cycle, its triangulation by the chord a–c, and the junction tree. */
export function FourCycleJunctionTree() {
  const state = useFigureState({
    stage: choice<Stage>(
      [
        { value: 'cycle', label: '1. four-cycle' },
        { value: 'chordal', label: '2. triangulated' },
        { value: 'tree', label: '3. junction tree' },
      ],
      'cycle',
      { label: 'stage' },
    ),
  })
  const s = STAGES[state.stage]
  return (
    <Figure
      title="From a four-cycle to a junction tree"
      state={state}
      caption={
        <MathText text="The four-cycle has no junction tree over its cliques. The chord $a - c$ (dashed) makes it chordal, with two maximal cliques. The junction tree joins them through their separator $\{a, c\}$, which labels the edge." />
      }

      readouts={<Readout label="maximal cliques" value={<MathText text={s.cliques} />} />}
    >
      <Diagram spec={s.spec} ariaLabel={`Stage: ${state.stage}`} />
    </Figure>
  )
}
