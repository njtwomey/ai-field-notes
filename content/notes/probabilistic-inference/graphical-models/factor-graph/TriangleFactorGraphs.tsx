import { MathText } from '@/components/content/MathText'
import { GraphDiagram, Interactive, type GraphEdge, type GraphNode } from '@/components/viz'

const VARIABLES: GraphNode[] = [
  { id: '1', label: 'x_1', x: 1, y: 0 },
  { id: '2', label: 'x_2', x: 0, y: 1.6 },
  { id: '3', label: 'x_3', x: 2, y: 1.6 },
]

const edge = (source: string, target: string): GraphEdge => ({ source, target, directed: false })

const PANELS: { caption: string; nodes: GraphNode[]; edges: GraphEdge[] }[] = [
  {
    caption: 'Markov random field',
    nodes: VARIABLES,
    edges: [edge('1', '2'), edge('2', '3'), edge('1', '3')],
  },
  {
    caption: 'three pairwise factors, $3k^2$ entries',
    nodes: [
      ...VARIABLES,
      { id: 'f12', label: 'f_{12}', x: 0.5, y: 0.8, kind: 'factor', labelPosition: 'left' },
      { id: 'f23', label: 'f_{23}', x: 1, y: 1.6, kind: 'factor', labelPosition: 'bottom' },
      { id: 'f13', label: 'f_{13}', x: 1.5, y: 0.8, kind: 'factor', labelPosition: 'right' },
    ],
    edges: [edge('1', 'f12'), edge('f12', '2'), edge('2', 'f23'), edge('f23', '3'), edge('1', 'f13'), edge('f13', '3')],
  },
  {
    caption: 'one factor on all three, $k^3$ entries',
    nodes: [...VARIABLES, { id: 'f123', label: 'f_{123}', x: 1, y: 1, kind: 'factor', labelPosition: 'bottom' }],
    edges: [edge('1', 'f123'), edge('2', 'f123'), edge('3', 'f123')],
  },
]

/** The same triangle as an undirected graph and as the factor graphs of two different factorisations. */
export function TriangleFactorGraphs() {
  return (
    <Interactive
      title="One undirected graph, two factor graphs"
      caption="Circles are variables and squares are factors. The triangle on the left fits both factorisations; the factor graphs tell them apart."
    >
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {PANELS.map((p) => (
          <div key={p.caption} className="text-center">
            <GraphDiagram nodes={p.nodes} edges={p.edges} height={170} ariaLabel={p.caption.replace(/\$/g, '')} />
            <p className="text-xs text-muted-foreground">
              <MathText text={p.caption} />
            </p>
          </div>
        ))}
      </div>
    </Interactive>
  )
}
