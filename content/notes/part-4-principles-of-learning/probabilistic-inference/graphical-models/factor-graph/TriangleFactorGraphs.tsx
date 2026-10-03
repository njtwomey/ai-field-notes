import { MathText } from 'aifn-render'
import { Diagram } from 'aifn-render'
import { factor, link, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { Interactive } from 'aifn-render'

const VARIABLES = [variable('1', 1.2, 0, '$x_1$'), variable('2', 0, 2, '$x_2$'), variable('3', 2.4, 2, '$x_3$')]

const edge = (from: string, to: string) => link(from, to, false)

const PANELS: { caption: string; spec: DiagramSpec }[] = [
  {
    caption: 'Markov random field',
    spec: { nodes: VARIABLES, edges: [edge('1', '2'), edge('2', '3'), edge('1', '3')] },
  },
  {
    caption: 'three pairwise factors, $3k^2$ entries',
    spec: {
      nodes: [
        ...VARIABLES,
        factor('f12', 0.6, 1, '$f_{12}$', 'w'),
        factor('f23', 1.2, 2, '$f_{23}$', 's'),
        factor('f13', 1.8, 1, '$f_{13}$', 'e'),
      ],
      edges: [
        edge('1', 'f12'),
        edge('f12', '2'),
        edge('2', 'f23'),
        edge('f23', '3'),
        edge('1', 'f13'),
        edge('f13', '3'),
      ],
    },
  },
  {
    caption: 'one factor on all three, $k^3$ entries',
    spec: {
      nodes: [...VARIABLES, factor('f123', 1.2, 1.25, '$f_{123}$', 's')],
      edges: [edge('1', 'f123'), edge('2', 'f123'), edge('3', 'f123')],
    },
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
          <div key={p.caption} className="flex flex-col justify-between gap-1 text-center">
            <Diagram spec={{ ...p.spec, unit: 48 }} ariaLabel={p.caption.replace(/\$/g, '')} />
            <p className="text-xs text-muted-foreground">
              <MathText text={p.caption} />
            </p>
          </div>
        ))}
      </div>
    </Interactive>
  )
}
