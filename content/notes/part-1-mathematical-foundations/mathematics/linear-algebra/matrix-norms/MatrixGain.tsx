import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { svd2 } from 'aifn/numerics/linalg'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const R = 3.5
const entry = (initial: number, label: string) => slider(-2, 2, initial, { step: 0.05, label })

/** The spectral norm as the longest stretch of the unit circle, beside the Frobenius and nuclear norms. */
export function MatrixGain() {
  const state = useFigureState({
    a: entry(1.5, 'a'),
    b: entry(-0.5, 'b'),
    c: entry(0.8, 'c'),
    d: entry(1, 'd'),
  })
  const [a, b, c, d] = [state.a, state.b, state.c, state.d]

  const r = useMemo(() => {
    const t = toFlat(linspace(0, 2 * Math.PI, 121))
    const svd = svd2([
      [a, b],
      [c, d],
    ])
    const [s1, s2] = svd.s
    const v1 = svd.v[0]
    const top: [number, number] = [s1 * svd.u[0][0], s1 * svd.u[0][1]]
    const series = [
      { name: 'unit circle', x: t.map(Math.cos), y: t.map(Math.sin), slot: 0, dashed: true },
      {
        name: 'image A x of the circle',
        x: t.map((s) => a * Math.cos(s) + b * Math.sin(s)),
        y: t.map((s) => c * Math.cos(s) + d * Math.sin(s)),
        slot: 1,
      },
      { name: 'most-stretched input v₁', x: [v1[0]], y: [v1[1]], slot: 0 },
    ] as const
    return {
      series,
      top,
      spectral: s1,
      frobenius: Math.hypot(a, b, c, d),
      nuclear: s1 + s2,
      one: Math.max(Math.abs(a) + Math.abs(c), Math.abs(b) + Math.abs(d)),
      inf: Math.max(Math.abs(a) + Math.abs(b), Math.abs(c) + Math.abs(d)),
    }
  }, [a, b, c, d])

  // The columns of A are the images of e₁ and e₂; dragging one sets that column.

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="How far a matrix can stretch a vector"
      caption="A = [[a, b], [c, d]] maps the dashed unit circle to the solid ellipse. The arrow is the longest semi-axis: its length is the spectral norm ‖A‖₂, reached at the input v₁ marked on the circle. The Frobenius norm is never smaller than the spectral norm, and equals it only when the ellipse is flat, a matrix of rank 1. The round handles are the columns of A; drag them or use the sliders."
      state={state}
      readouts={
        <>
          <Readout label="spectral ‖A‖₂ = σ₁" value={formatNumber(r.spectral)} />
          <Readout label="Frobenius ‖A‖_F" value={formatNumber(r.frobenius)} />
          <Readout label="nuclear ‖A‖_* = σ₁ + σ₂" value={formatNumber(r.nuclear)} />
          <Readout label="‖A‖₁ (column sums)" value={formatNumber(r.one)} />
          <Readout label="‖A‖∞ (row sums)" value={formatNumber(r.inf)} />
        </>
      }
    >
      {/* Equal-aspect charts take their height from their width; keep square plots a readable size. */}
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...r.series[0]} />
          <Curve {...r.series[1]} />
          <Points {...r.series[2]} />
          <Vectors vectors={[{ from: [0, 0], to: r.top }]} />
          <Handle {...state.handle(['a', 'c'], { label: 'A e₁' })} />
          <Handle {...state.handle(['b', 'd'], { label: 'A e₂' })} />
        </Plot>
      </div>
    </Figure>
  )
}
