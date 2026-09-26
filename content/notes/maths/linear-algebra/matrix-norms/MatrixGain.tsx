import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type ParamSpec,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { svd2 } from '@/lib/math/mat2'

const R = 3.5
const ENTRY: ParamSpec = { min: -2, max: 2, step: 0.05 }

/** The spectral norm as the longest stretch of the unit circle, beside the Frobenius and nuclear norms. */
export function MatrixGain() {
  const pa = useParam(1.5, ENTRY)
  const pb = useParam(-0.5, ENTRY)
  const pc = useParam(0.8, ENTRY)
  const pd = useParam(1, ENTRY)
  const [a, b, c, d] = [pa.value, pb.value, pc.value, pd.value]

  const r = useMemo(() => {
    const t = linspace(0, 2 * Math.PI, 121)
    const svd = svd2(a, b, c, d)
    const [s1, s2] = svd.s
    const v1 = svd.v[0]
    const top: [number, number] = [s1 * svd.u[0][0], s1 * svd.u[0][1]]
    const series: XYSeries[] = [
      { name: 'unit circle', type: 'line', x: t.map(Math.cos), y: t.map(Math.sin), slot: 0, dashed: true },
      {
        name: 'image A x of the circle',
        type: 'line',
        x: t.map((s) => a * Math.cos(s) + b * Math.sin(s)),
        y: t.map((s) => c * Math.cos(s) + d * Math.sin(s)),
        slot: 1,
      },
      { name: 'most-stretched input v₁', type: 'scatter', x: [v1[0]], y: [v1[1]], slot: 0 },
    ]
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
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [a, c],
      label: 'A e₁',
      onDrag: ([x, y]) => {
        pa.set(x)
        pc.set(y)
      },
    },
    {
      kind: 'point',
      at: [b, d],
      label: 'A e₂',
      onDrag: ([x, y]) => {
        pb.set(x)
        pd.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="How far a matrix can stretch a vector"
      caption="A = [[a, b], [c, d]] maps the dashed unit circle to the solid ellipse. The arrow is the longest semi-axis: its length is the spectral norm ‖A‖₂, reached at the input v₁ marked on the circle. The Frobenius norm is never smaller than the spectral norm, and equals it only when the ellipse is flat, a matrix of rank 1. The round handles are the columns of A; drag them or use the sliders."
      controls={
        <>
          <ParamSlider label="a" param={pa} />
          <ParamSlider label="b" param={pb} />
          <ParamSlider label="c" param={pc} />
          <ParamSlider label="d" param={pd} />
        </>
      }
      readout={
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
        <XYChart
          equalAspect
          xRange={[-R, R]}
          yRange={[-R, R]}
          xLabel="x₁"
          yLabel="x₂"
          series={r.series}
          vectors={[{ from: [0, 0], to: r.top }]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
