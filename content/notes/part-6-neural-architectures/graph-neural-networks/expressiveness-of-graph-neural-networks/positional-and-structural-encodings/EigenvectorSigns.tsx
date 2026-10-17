import { useMemo } from 'react'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import {
  choice,
  Figure,
  Plot,
  Points,
  Readout,
  Segments,
  setting,
  slider,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { laplacianEncoding } from 'aifn-compute/graph/matrices'
import { drawingRange } from '../_shared/graph-layers'
import { hexagon, path4, type Drawn } from '../_shared/graph-pairs'

const GRAPHS: Record<'path' | 'hexagon', Drawn> = { path: path4, hexagon }
const fmt = (v: number) => (Math.abs(v) < 5e-4 ? '0' : v.toFixed(3))

/** Eigenspaces of the normalised Laplacian (without the zero eigenvalue): eigenvalue, basis vectors. */
function eigenspaces(d: Drawn) {
  const n = d.graph.nodes
  const { encoding, values } = laplacianEncoding(d.graph, n - 1)
  const cols = toRows(encoding)
  const spaces: { value: number; basis: number[][] }[] = []
  toFlat(values).forEach((value, j) => {
    const column = cols.map((row) => row[j])
    const last = spaces[spaces.length - 1]
    if (last && Math.abs(last.value - value) < 1e-8) last.basis.push(column)
    else spaces.push({ value, basis: [column] })
  })
  return spaces
}

/** A Laplacian eigenvector as a positional encoding: its sign, and for a repeated eigenvalue its basis, are arbitrary. */
export function EigenvectorSigns() {
  const state = useFigureState({
    graph: choice(
      [
        { value: 'path', label: 'path on 4 nodes' },
        { value: 'hexagon', label: '6-cycle' },
      ],
      'path',
      { label: 'graph' },
    ),
    space: choice(
      [
        { value: 0, label: 'λ = 1/2' },
        { value: 1, label: 'λ = 3/2' },
        { value: 2, label: 'λ = 2' },
      ],
      0,
      { label: 'eigenvalue' },
    ),
    flip: setting(false, 'flip the sign'),
    angle: slider(0, 180, 0, { step: 5, label: 'rotation in a repeated eigenspace (degrees)' }),
  })
  const d = GRAPHS[state.graph]
  const spaces = useMemo(() => eigenspaces(d), [d])
  const space = spaces[Math.min(state.space, spaces.length - 1)]
  const theta = (state.angle * Math.PI) / 180
  const sign = state.flip ? -1 : 1
  const n = d.graph.nodes
  const vector = Array.from({ length: n }, (_, i) =>
    space.basis.length === 1
      ? sign * space.basis[0][i]
      : sign * (Math.cos(theta) * space.basis[0][i] + Math.sin(theta) * space.basis[1][i]),
  )
  const projector = Array.from({ length: n }, (_, i) => space.basis.reduce((s, u) => s + u[i] * u[i], 0))
  const colour = useScaleColor('diverging')
  const colors = vector.map((v) => colour(Math.max(0, Math.min(1, (v + 0.75) / 1.5))))
  const range = drawingRange([d], 0.6)
  const x = useAxis({ range: range.x, nice: false, zoom: false })
  const y = useAxis({ range: range.y, nice: false, zoom: false, equal: x })
  return (
    <Figure
      title="Laplacian eigenvectors are defined up to sign and basis"
      state={state}
      readouts={
        <>
          <Readout label="eigenvalue" value={space.value.toFixed(3)} />
          <Readout label="multiplicity" value={space.basis.length} />
          <Readout label="encoding, node by node" value={vector.map(fmt).join(', ')} />
          <Readout label="|encoding| (sign-invariant)" value={vector.map((v) => fmt(Math.abs(v))).join(', ')} />
          <Readout label="Σ over the eigenspace of u_i² (basis-invariant)" value={projector.map(fmt).join(', ')} />
        </>
      }
      caption="Nodes are coloured and labelled by one eigenvector of the normalised Laplacian I − D^(−1/2) A D^(−1/2), the positional encoding a node would receive. Flipping the sign gives an equally valid eigenvector, so the encoding of a node is not determined by the graph. On the 6-cycle every non-zero eigenvalue except the largest is repeated: rotating within the two-dimensional eigenspace gives a different valid encoding at every angle. The absolute value is unchanged by a flip but not by a rotation; the diagonal of the projector onto the eigenspace is unchanged by both."
    >
      <Plot x={x} y={y} bare legend={false}>
        <Segments name="edges" segments={d.edges.map(([a, b]) => ({ from: d.pos[a], to: d.pos[b] }))} width={1.5} />
        <Points
          name="nodes"
          x={d.pos.map((p) => p[0])}
          y={d.pos.map((p) => p[1])}
          colors={colors}
          labels={vector.map(fmt)}
          size={18}
        />
      </Plot>
    </Figure>
  )
}
