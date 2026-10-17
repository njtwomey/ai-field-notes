import { useContext, useMemo, type ReactNode } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  Rug,
  Segments,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { unweightedShortestPaths } from 'aifn-compute/graph/traversal'
import { rescaledLaplacian } from 'aifn-compute/graph/matrices'
import { chebyshevFilter } from 'aifn-compute/nn/graph'
import { chebyshevInterpolation, chebyshevSeries } from 'aifn-compute/numerics/interpolate'
import { EDGE_SEGMENTS, EIGENVALUES, EIGENVECTORS, GRAPH, N, POSITIONS, XS, YS } from '../_shared/sensorGraph'

const LAMBDA_MAX = EIGENVALUES[N - 1]
const DOMAIN: [number, number] = [0, LAMBDA_MAX]
const { lTilde } = rescaledLaplacian(GRAPH, { normalisation: 'unnormalised', lambdaMax: LAMBDA_MAX })
const GRID = Array.from({ length: 241 }, (_, i) => (i / 240) * LAMBDA_MAX)
const HOPS = Array.from({ length: N }, (_, s) => Array.from(toFlat(unweightedShortestPaths(GRAPH, s).distance)))

type Target = 'low-pass' | 'heat' | 'band-pass'
const TARGETS: Record<Target, (cut: number) => (lambda: number) => number> = {
  'low-pass': (cut) => (l) => (l <= cut ? 1 : 0),
  heat: (cut) => (l) => Math.exp(-l / cut),
  'band-pass': (cut) => (l) => Math.exp(-((l - cut) ** 2) / (2 * 0.6 ** 2)),
}

function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

const nearestNode = ([x, y]: [number, number]) =>
  POSITIONS.reduce((best, p, i) => {
    const d = (p[0] - x) ** 2 + (p[1] - y) ** 2
    const b = POSITIONS[best]
    return d < (b[0] - x) ** 2 + (b[1] - y) ** 2 ? i : best
  }, 0)

/**
 * A spectral filter and its degree-K Chebyshev approximation (`chebyshevInterpolation`, `chebyshevFilter`): the
 * frequency responses, and the two filters applied to an impulse at one node.
 */
export function ChebyshevFilter() {
  const state = useFigureState({
    target: choice<Target>(
      [
        { value: 'low-pass', label: 'ideal low-pass 1[λ ≤ λc]' },
        { value: 'heat', label: 'heat kernel exp(−λ/λc)' },
        { value: 'band-pass', label: 'band-pass around λc' },
      ],
      'low-pass',
      { label: 'target response g(λ)' },
    ),
    cutoff: float(2, { min: 0.1, max: 7.2, step: 0.05, label: 'λc' }),
    order: int(2, { ge: 0, le: 25, label: 'polynomial degree K', suggestions: [1, 2, 3, 5, 10, 20] }),
    source: int(14, { ge: 0, le: N - 1, label: 'impulse node' }),
  })
  const { target, cutoff, order, source } = state
  const g = useMemo(() => TARGETS[target](cutoff), [target, cutoff])
  const theta = useMemo(() => chebyshevInterpolation(g, order, DOMAIN), [g, order])
  const approx = GRID.map((l) => chebyshevSeries(theta, l, DOMAIN))
  const exactCurve = GRID.map(g)

  const impulse = useMemo(() => tensor(Array.from({ length: N }, (_, i) => (i === source ? 1 : 0))), [source])
  const cheb = useMemo(() => Array.from(toFlat(chebyshevFilter(lTilde, impulse, theta) as Tensor)), [impulse, theta])
  // The exact spectral filter U g(Λ) Uᵀ δ_s = Σ_k g(λ_k) u_k(s) u_k.
  const exact = useMemo(
    () =>
      Array.from({ length: N }, (_, i) =>
        EIGENVALUES.reduce((s, l, k) => s + g(l) * EIGENVECTORS[k][source] * EIGENVECTORS[k][i], 0),
      ),
    [g, source],
  )
  const hops = HOPS[source]
  const within = hops.filter((h) => h <= order).length
  const nonZero = (y: number[]) => y.filter((v) => Math.abs(v) > 1e-12).length
  const worst = Math.max(...EIGENVALUES.map((l) => Math.abs(chebyshevSeries(theta, l, DOMAIN) - g(l))))

  const colour = useScaleColor('diverging')
  const bound = Math.max(1e-9, ...exact.map(Math.abs), ...cheb.map(Math.abs))
  const reached = (y: number[], keep: (i: number) => boolean) => {
    const idx = y.map((_, i) => i).filter(keep)
    return {
      x: idx.map((i) => XS[i]),
      y: idx.map((i) => YS[i]),
      colors: idx.map((i) => colour((y[i] / bound + 1) / 2)),
    }
  }
  const chebOn = reached(cheb, (i) => Math.abs(cheb[i]) > 1e-12)
  const chebOff = reached(cheb, (i) => Math.abs(cheb[i]) <= 1e-12)
  const exactOn = reached(exact, () => true)

  const rx = useAxis({ label: 'eigenvalue λ', range: [0, LAMBDA_MAX], nice: false })
  const ry = useAxis({ label: 'gain', range: [-0.4, 1.4], nice: false })
  const gx = useAxis({ label: '', range: [-0.6, 5.6], nice: false })
  const gy = useAxis({ label: '', range: [-0.6, 4.6], nice: false, equal: gx })
  const moveSource = (p: [number, number]) => state.set('source', nearestNode(p))

  return (
    <Figure
      title="Chebyshev filters are local"
      state={state}
      hoverReadout={false}
      caption={`Top: a target frequency response g(λ) on the spectrum of the ${N}-node graph (dashed), its degree-K Chebyshev approximation (coloured), and the graph's eigenvalues (ticks). Drag the vertical line to move λc. Bottom: both filters applied to an impulse at one node; drag the marker to another node. The exact filter U g(Λ) Uᵀ needs every eigenvector and spreads over the whole graph. The Chebyshev filter needs only K products with the rescaled Laplacian, and its response is exactly zero (hollow grey) at every node more than K hops from the impulse. Raising K fits the target more closely and widens the neighbourhood; a sharp cut-off needs a high degree and rings near the jump.`}
      readouts={
        <>
          <Readout label="nodes within K hops" value={`${within} of ${N}`} />
          <Readout label="non-zero nodes: Chebyshev / exact" value={`${nonZero(cheb)} / ${nonZero(exact)}`} />
          <Readout label="max |g_K(λₖ) − g(λₖ)| on the spectrum" value={formatNumber(Number(worst.toPrecision(3)))} />
        </>
      }
    >
      <FixedHeight height={250}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="md:col-span-2">
            <Plot x={rx} y={ry} title="frequency response">
              <Curve name="target g(λ)" x={GRID} y={exactCurve} emphasis dashed />
              <Curve name={`Chebyshev, K = ${order}`} x={GRID} y={approx} slot={0} />
              <Rug values={EIGENVALUES} />
              <Handle
                kind="x"
                at={cutoff}
                label="λc"
                onDrag={(x) => state.set('cutoff', Math.max(0.1, Math.min(7.2, Math.round(x * 20) / 20)))}
              />
            </Plot>
          </div>
          <Plot x={gx} y={gy} title={`Chebyshev filter, K = ${order}`} bare>
            <Segments segments={EDGE_SEGMENTS} muted width={1.5} />
            {chebOff.x.length > 0 && <Points name="exactly zero" x={chebOff.x} y={chebOff.y} muted size={9} />}
            {chebOn.x.length > 0 && <Points name="node" x={chebOn.x} y={chebOn.y} colors={chebOn.colors} size={13} />}
            <Handle kind="point" at={POSITIONS[source]} label="impulse" onDrag={moveSource} />
          </Plot>
          <Plot x={gx} y={gy} title="exact spectral filter" bare>
            <Segments segments={EDGE_SEGMENTS} muted width={1.5} />
            <Points name="node" x={exactOn.x} y={exactOn.y} colors={exactOn.colors} size={13} />
            <Handle kind="point" at={POSITIONS[source]} label="impulse" onDrag={moveSource} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
