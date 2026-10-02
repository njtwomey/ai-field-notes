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
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { svd2 } from '@/lib/math/mat2'

const R = 3.5
const ENTRY: ParamSpec = { min: -2, max: 2, step: 0.05 }

/** A 2×2 matrix maps the unit circle to an ellipse whose axes are σ₁u₁ and σ₂u₂. */
export function UnitCircle() {
  const pa = useParam(1.5, ENTRY)
  const pb = useParam(1, ENTRY)
  const pc = useParam(0.4, ENTRY)
  const pd = useParam(1.2, ENTRY)
  const [a, b, c, d] = [pa.value, pb.value, pc.value, pd.value]

  const r = useMemo(() => {
    const t = linspace(0, 2 * Math.PI, 121)
    const circle = { x: t.map(Math.cos), y: t.map(Math.sin) }
    const ellipse = {
      x: t.map((s) => a * Math.cos(s) + b * Math.sin(s)),
      y: t.map((s) => c * Math.cos(s) + d * Math.sin(s)),
    }
    const svd = svd2(a, b, c, d)
    const series: XYSeries[] = [
      { name: 'unit circle', type: 'line', ...circle, slot: 0, dashed: true },
      { name: 'image of the circle, A x', type: 'line', ...ellipse, slot: 1 },
      {
        name: 'right singular vectors vᵢ',
        type: 'scatter',
        x: svd.v.map((v) => v[0]),
        y: svd.v.map((v) => v[1]),
        slot: 0,
      },
    ]
    return { svd, series }
  }, [a, b, c, d])

  const { s, u } = r.svd
  // The columns of A are the images of the basis vectors: A e₁ = (a, c) and A e₂ = (b, d). Dragging one sets a column.
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
  const rank = s[1] > 1e-9 ? 2 : s[0] > 1e-9 ? 1 : 0
  return (
    <Interactive
      title="Every matrix turns circles into ellipses"
      caption="Set the entries of A = [[a, b], [c, d]]. The dashed unit circle maps to the solid ellipse. The points vᵢ on the circle are the right singular vectors; A sends them to the arrows σᵢuᵢ, the ellipse's semi-axes. Make the rows proportional and the ellipse collapses to a segment: the matrix loses rank. The two round handles are the columns of A, the images A e₁ = (a, c) and A e₂ = (b, d); drag them to set the matrix."
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
          <Readout label="σ₁" value={formatNumber(s[0])} />
          <Readout label="σ₂" value={formatNumber(s[1])} />
          <Readout label="|det A| = σ₁σ₂" value={formatNumber(Math.abs(a * d - b * c))} />
          <Readout label="condition number σ₁/σ₂" value={s[1] > 1e-9 ? formatNumber(s[0] / s[1]) : '∞'} />
          <Readout label="rank" value={rank} />
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
          vectors={[
            { from: [0, 0], to: [s[0] * u[0][0], s[0] * u[0][1]] },
            { from: [0, 0], to: [s[1] * u[1][0], s[1] * u[1][1]] },
          ]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
