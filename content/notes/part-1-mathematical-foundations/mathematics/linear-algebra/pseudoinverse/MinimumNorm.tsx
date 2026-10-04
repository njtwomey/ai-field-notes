import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'

type Vec = [number, number]
const R = 4
const snap = (v: number) => Math.round(Math.min(Math.max(v, -3), 3) * 20) / 20

/** One equation in two unknowns, a₁x₁ + a₂x₂ = b: a line of solutions, of which A⁺b is the closest to the origin. */
export function MinimumNorm() {
  const [a, setA] = useState<Vec>([1, 2])
  const state = useFigureState({
    b: float(3, { min: -6, max: 6, step: 0.1, label: 'right-hand side b' }),
  })

  const r = useMemo(() => {
    const n2 = a[0] * a[0] + a[1] * a[1]
    // A = [a₁ a₂] has A⁺ = aᵀ/‖a‖², unless a = 0, when A⁺ = 0.
    const pinv: Vec = n2 > 0 ? [a[0] / n2, a[1] / n2] : [0, 0]
    const x: Vec = [pinv[0] * state.b, pinv[1] * state.b]
    const series: SeriesSpec[] = []
    if (n2 > 0) {
      const d: Vec = [-a[1] / Math.sqrt(n2), a[0] / Math.sqrt(n2)]
      const T = 20
      series.push({
        name: 'solutions of a₁x₁ + a₂x₂ = b',
        type: 'line',
        x: [x[0] - T * d[0], x[0] + T * d[0]],
        y: [x[1] - T * d[1], x[1] + T * d[1]],
        slot: 0,
      })
    }
    series.push(
      { name: 'shortest path to the line', type: 'line', x: [0, x[0]], y: [0, x[1]], slot: 1, dashed: true },
      { name: 'minimum-norm solution A⁺b', type: 'scatter', x: [x[0]], y: [x[1]], emphasis: true },
    )
    return { pinv, x, series, solvable: n2 > 0 || state.b === 0 }
  }, [a, state.b])

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="The solution closest to the origin"
      state={state}
      caption="One equation in two unknowns has a line of solutions. Drag the tip of the arrow a to change the coefficients, and set b with the slider. The pseudoinverse picks the solution A⁺b nearest the origin; the dashed path to it meets the line at a right angle, so A⁺b is parallel to a. Shrink a towards zero and A⁺b runs off to infinity; at a = 0 exactly, A⁺ = 0 and A⁺b jumps back to the origin."

      readouts={
        <>
          <Readout label="A⁺" value={`(${formatNumber(r.pinv[0])}, ${formatNumber(r.pinv[1])})ᵀ`} />
          <Readout label="A⁺b" value={`(${formatNumber(r.x[0])}, ${formatNumber(r.x[1])})`} />
          <Readout label="‖A⁺b‖" value={formatNumber(Math.hypot(...r.x))} />
          <Readout label="exact solution exists" value={r.solvable ? 'yes' : 'no'} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(r.series)}
          <Vectors vectors={[{ from: [0, 0], to: a }]} />
          <Handle kind="point" at={a} label="a" onDrag={([p, q]) => setA([snap(p), snap(q)])} />
        </Plot>
      </div>
    </Figure>
  )
}
