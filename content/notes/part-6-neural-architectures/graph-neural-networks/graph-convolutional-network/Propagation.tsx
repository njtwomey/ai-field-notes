import { useMemo } from 'react'
import { choice, Figure, formatNumber, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'

// Two 5-cliques (nodes 0–4 and 5–9) joined by one edge between nodes 4 and 5.
const N = 10
const EDGES: [number, number][] = []
for (const base of [0, 5]) for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) EDGES.push([base + i, base + j])
EDGES.push([4, 5])

// A fixed, arbitrary starting signal, one scalar feature per node.
const X0 = [1, -1, 0.5, 0.2, -0.8, 0.3, -0.5, 1, -0.2, 0.6]

type Operator = 'renormalised' | 'first-order'

/**
 * Renormalised: D̃^{-1/2} (A + I) D̃^{-1/2}, eigenvalues in (−1, 1].
 * First-order: I + D^{-1/2} A D^{-1/2}, eigenvalues in [0, 2].
 */
function operator(kind: Operator): number[][] {
  const a = Array.from({ length: N }, () => new Array<number>(N).fill(0))
  for (const [i, j] of EDGES) a[i][j] = a[j][i] = 1
  const loops = kind === 'renormalised' ? 1 : 0
  const deg = a.map((row) => row.reduce((s, v) => s + v, 0) + loops)
  return a.map((row, i) =>
    row.map(
      (v, j) => (v + (i === j ? loops : 0)) / Math.sqrt(deg[i] * deg[j]) + (kind === 'first-order' && i === j ? 1 : 0),
    ),
  )
}

const OPERATORS: Record<Operator, number[][]> = {
  renormalised: operator('renormalised'),
  'first-order': operator('first-order'),
}

const apply = (m: number[][], x: number[]) => m.map((row) => row.reduce((s, v, j) => s + v * x[j], 0))

/** Repeated graph-convolution propagation of one feature, without weights or nonlinearity. */
export function Propagation() {
  const state = useFigureState({
    kind: choice<Operator>(
      [
        { value: 'renormalised', label: 'renormalised Â' },
        { value: 'first-order', label: 'first-order' },
      ],
      'renormalised',
      { label: 'propagation matrix' },
    ),
    steps: int(12, { min: 1, max: 30, step: 1, label: 'steps', format: (v) => String(v) }),
  })

  const rows = useMemo(() => {
    const out = [X0]
    for (let k = 1; k <= state.steps; k++) out.push(apply(OPERATORS[state.kind], out[k - 1]))
    return out
  }, [state.kind, state.steps])

  const last = rows[rows.length - 1]
  const maxAbs = Math.max(...last.map(Math.abs))
  const bound = Math.max(...rows.flat().map(Math.abs))
  const mean = (xs: number[]) => xs.reduce((s, v) => s + v, 0) / xs.length
  const gap = Math.abs(mean(last.slice(0, 5)) - mean(last.slice(5)))

  const xAxis = useAxis({ label: 'node' })
  const yAxis = useAxis({ label: 'step k' })
  return (
    <Figure
      title="Repeated propagation on a graph"
      state={state}
      caption="Two 5-node cliques joined by one edge (nodes 4 and 5). Each row applies the propagation matrix once more to a scalar feature, starting from an arbitrary signal at step 0. With the renormalised matrix, one step makes each clique almost uniform, and further steps slowly pull the two cliques together: this is oversmoothing. The first-order matrix I + D^(−1/2) A D^(−1/2) has an eigenvalue of 2, so the feature grows by a factor approaching 2 at every step."

      readouts={
        <>
          <Readout label="max |feature| at last step" value={formatNumber(maxAbs)} />
          <Readout label="gap between clique means" value={formatNumber(gap)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={360}>
        <Raster
          x={Array.from({ length: N }, (_, i) => i)}
          y={rows.map((_, k) => k)}
          z={rows}
          scale={'diverging'}
          range={[-bound, bound]}
          valueLabel={'feature'}
        />
      </Plot>
    </Figure>
  )
}
