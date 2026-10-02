import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
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
  const n = useParam(10, { min: 2, max: 24, step: 1 })
  const [nodes, setNodes] = useState<Nodes>('equispaced')
  const [lower, setLower] = useState<Lower>('omega')

  const r = useMemo(() => {
    const x = nodes === 'equispaced' ? equispacedNodes(n.value + 1, -1, 1) : chebyshevNodes(n.value + 1, -1, 1)
    const y = x.map(runge)
    const w = barycentricWeights(x)
    const p = GRID.map((u) => barycentricEval(x, y, w, u))
    const err = Math.max(...GRID.map((u, i) => Math.abs(p[i] - runge(u))))
    const omega = GRID.map((u) => Math.max(Math.abs(nodePolynomial(x, u)), 1e-12))
    const leb = GRID.map((u) => lebesgueFunction(x, w, u))
    const top: XYSeries[] = [
      { name: 'f(x) = 1 / (1 + 25x²)', type: 'line', x: GRID, y: GRID.map(runge), slot: 1, dashed: true },
      { name: `interpolant of degree ${n.value}`, type: 'line', x: GRID, y: p, slot: 0 },
      { name: 'nodes', type: 'scatter', x, y, slot: 0 },
    ]
    return { err, omega, leb, top, maxOmega: Math.max(...omega), lambda: Math.max(...leb) }
  }, [n.value, nodes])

  const lowerSeries: XYSeries[] = useMemo(
    () =>
      lower === 'omega'
        ? [{ name: '|ω(x)| = |∏(x − xⱼ)|', type: 'line', x: GRID, y: r.omega, slot: 0 }]
        : [{ name: 'Lebesgue function Σ|ℓⱼ(x)|', type: 'line', x: GRID, y: r.leb, slot: 0 }],
    [lower, r],
  )

  return (
    <Interactive
      title="Runge's phenomenon"
      caption="Raise the degree n with equispaced nodes: the interpolant converges in the middle and oscillates ever more wildly near ±1. Switch to Chebyshev nodes: the error falls at every x. The lower panel shows why. The node polynomial ω sets the size of the error term, and the Lebesgue function bounds how much the interpolant can amplify errors in the data."
      controls={
        <>
          <ParamSlider label="Degree n (n + 1 nodes)" param={n} withArrows format={(v) => String(v)} />
          <ParamChoice
            label="Nodes"
            value={nodes}
            onChange={setNodes}
            options={[
              { value: 'equispaced', label: 'equispaced' },
              { value: 'chebyshev', label: 'Chebyshev' },
            ]}
          />
          <ParamChoice
            label="Lower panel"
            value={lower}
            onChange={setLower}
            options={[
              { value: 'omega', label: 'node polynomial' },
              { value: 'lebesgue', label: 'Lebesgue function' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="max |f − p|" value={formatNumber(r.err)} />
          <Readout label="max |ω|" value={r.maxOmega.toExponential(2)} />
          <Readout label="2⁻ⁿ (Chebyshev optimum)" value={(2 ** -n.value).toExponential(2)} />
          <Readout label="Lebesgue constant Λ" value={formatNumber(r.lambda)} />
        </>
      }
    >
      <XYChart series={r.top} xRange={[-1, 1]} yRange={[-1, 2]} xLabel="x" yLabel="y" height={300} />
      <XYChart
        series={lowerSeries}
        xRange={[-1, 1]}
        yRange={lower === 'omega' ? [1e-9, 1] : [1, 2e4]}
        yLog
        xLabel="x"
        height={220}
      />
    </Interactive>
  )
}
