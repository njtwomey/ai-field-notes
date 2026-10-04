import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  barycentricEval,
  barycentricWeights,
  chebyshevNodes,
  equispacedNodes,
  lebesgueFunction,
  nodePolynomial,
} from '../_shared/splines'

type Nodes = 'equispaced' | 'chebyshev'
type Lower = 'omega' | 'lebesgue'

const runge = (x: number) => 1 / (1 + 25 * x * x)
const GRID = Array.from({ length: 801 }, (_, i) => -1 + i / 400)

/**
 * Runge's function 1 / (1 + 25 x^2) and its interpolating polynomial of degree n at equispaced or Chebyshev nodes,
 * with the node polynomial or the Lebesgue function below.
 */
export function RungeExplorer() {
  const state = useFigureState({
    n: int(10, { min: 2, max: 24, step: 1, label: 'Degree n (n + 1 nodes)', format: (v) => String(v) }),
    nodes: choice<Nodes>(
      [
        { value: 'equispaced', label: 'equispaced' },
        { value: 'chebyshev', label: 'Chebyshev' },
      ],
      'equispaced',
      { label: 'Nodes' },
    ),
    lower: choice<Lower>(
      [
        { value: 'omega', label: 'node polynomial' },
        { value: 'lebesgue', label: 'Lebesgue function' },
      ],
      'omega',
      { label: 'Lower panel' },
    ),
  })

  const r = useMemo(() => {
    const x = state.nodes === 'equispaced' ? equispacedNodes(state.n + 1, -1, 1) : chebyshevNodes(state.n + 1, -1, 1)
    const y = x.map(runge)
    const w = barycentricWeights(x)
    const p = GRID.map((u) => barycentricEval(x, y, w, u))
    const err = Math.max(...GRID.map((u, i) => Math.abs(p[i] - runge(u))))
    const omega = GRID.map((u) => Math.max(Math.abs(nodePolynomial(x, u)), 1e-12))
    const leb = GRID.map((u) => lebesgueFunction(x, w, u))
    const top = [
      { name: 'f(x) = 1 / (1 + 25x²)', x: GRID, y: GRID.map(runge), slot: 1, dashed: true },
      { name: `interpolant of degree ${state.n}`, x: GRID, y: p, slot: 0 },
      { name: 'nodes', x, y, slot: 0 },
    ] as const
    return { err, omega, leb, top, maxOmega: Math.max(...omega), lambda: Math.max(...leb) }
  }, [state.n, state.nodes])

  const lowerSeries: SeriesSpec[] = useMemo(
    () =>
      state.lower === 'omega'
        ? [{ name: '|ω(x)| = |∏(x − xⱼ)|', type: 'line', x: GRID, y: r.omega, slot: 0 }]
        : [{ name: 'Lebesgue function Σ|ℓⱼ(x)|', type: 'line', x: GRID, y: r.leb, slot: 0 }],
    [state.lower, r],
  )

  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-1, 2] })
  const xAxis2 = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis2 = useAxis({ range: state.lower === 'omega' ? [1e-9, 1] : [1, 2e4], log: true })
  return (
    <Figure
      title="Runge's phenomenon"
      state={state}
      caption="Raise the degree n with equispaced nodes: the interpolant converges in the middle and oscillates ever more wildly near ±1. Switch to Chebyshev nodes: the error falls at every x. The lower panel shows why. The node polynomial ω sets the size of the error term, and the Lebesgue function bounds how much the interpolant can amplify errors in the data."

      readouts={
        <>
          <Readout label="max |f − p|" value={formatNumber(r.err)} />
          <Readout label="max |ω|" value={r.maxOmega.toExponential(2)} />
          <Readout label="2⁻ⁿ (Chebyshev optimum)" value={(2 ** -state.n).toExponential(2)} />
          <Readout label="Lebesgue constant Λ" value={formatNumber(r.lambda)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...r.top[0]} />
        <Curve {...r.top[1]} />
        <Points {...r.top[2]} />
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={220}>
        {seriesLayers(lowerSeries)}
      </Plot>
    </Figure>
  )
}
