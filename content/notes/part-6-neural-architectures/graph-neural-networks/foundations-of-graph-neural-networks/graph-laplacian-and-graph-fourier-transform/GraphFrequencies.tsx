import { useContext, type ReactNode } from 'react'
import {
  Figure,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { EDGE_SEGMENTS, EDGES, EIGENVALUES, EIGENVECTORS, N, signChanges, XS, YS } from '../_shared/sensorGraph'

const INDEX = EIGENVALUES.map((_, k) => k)

function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** The eigenvectors of the Laplacian of a 30-node graph, drawn on the graph: graph frequencies ordered by smoothness. */
export function GraphFrequencies() {
  const state = useFigureState({
    k: int(1, { ge: 0, le: N - 1, label: 'eigenvector index k' }),
  })
  const { k } = state
  const u = EIGENVECTORS[k]
  const colour = useScaleColor('diverging')
  const bound = Math.max(...u.map(Math.abs)) || 1
  const colours = u.map((v) => colour((v / bound + 1) / 2))

  const gx = useAxis({ label: '', range: [-0.6, 5.6], nice: false })
  const gy = useAxis({ label: '', range: [-0.6, 4.6], nice: false, equal: gx })
  const sx = useAxis({ label: 'index k', range: [-1, N], nice: false, integer: true })
  const sy = useAxis({ label: 'eigenvalue λₖ', range: [0, 8] })

  return (
    <Figure
      title="Graph frequencies"
      state={state}
      hoverReadout={false}
      caption={`A graph of ${N} points, each joined to its 3 nearest neighbours (${EDGES.length} edges). Left: the eigenvector u_k of the Laplacian L = D − A, one value per node, coloured on a scale diverging about zero. Right: the eigenvalues λ₀ ≤ λ₁ ≤ … in ascending order. Drag the vertical line on the right, or type k, to choose the eigenvector. u₀ is constant with λ₀ = 0. u₁ and u₂ each split the graph into one positive and one negative region, like the slowest vibrations of a drum. Eigenvectors with large k alternate between neighbours. λ_k is the eigenvector's Dirichlet energy, the sum over edges of its squared differences, and the count of edges with a sign change rises from 0 at k = 0 to about 40 at the top of the spectrum, not quite monotonically.`}
      readouts={
        <>
          <Readout label="eigenvalue λₖ = uₖᵀ L uₖ" value={formatNumber(Number(EIGENVALUES[k].toPrecision(3)))} />
          <Readout label="edges with a sign change" value={`${signChanges(u)} of ${EDGES.length}`} />
        </>
      }
    >
      <FixedHeight height={280}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={gx} y={gy} title={`eigenvector u${k} on the graph`} bare>
            <Segments segments={EDGE_SEGMENTS} muted width={1.5} />
            <Points name="node" x={XS} y={YS} colors={colours} size={13} />
          </Plot>
          <Plot x={sx} y={sy} title="spectrum">
            <Points name="eigenvalue" x={INDEX} y={EIGENVALUES} muted size={6} />
            <Points name="chosen" x={[k]} y={[EIGENVALUES[k]]} emphasis size={10} />
            <Handle
              kind="x"
              at={k}
              label="k"
              onDrag={(x) => state.set('k', Math.max(0, Math.min(N - 1, Math.round(x))))}
            />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
