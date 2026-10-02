import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'

const Z = Array.from({ length: 101 }, (_, i) => i * 4)
const SHORT = 40
const LONG = 320

/** The pivoted normaliser (1 − s)p + sZ against the old normaliser Z: a rotation of the identity line about (p, p). */
export function PivotedNormalisation() {
  const slope = useParam(0.3, { min: 0.05, max: 1, step: 0.05 })
  const pivot = useParam(120, { min: 20, max: 380, step: 1 })
  const s = slope.value
  const p = pivot.value

  const series = useMemo<XYSeries[]>(
    () => [
      { name: 'old normaliser Z', type: 'line', x: Z, y: Z, muted: true, dashed: true },
      { name: 'pivoted (1 − s)p + sZ', type: 'line', x: Z, y: Z.map((z) => (1 - s) * p + s * z), slot: 0 },
      { name: 'pivot (p, p)', type: 'scatter', x: [p], y: [p], emphasis: true },
    ],
    [s, p],
  )
  const handles = useMemo<Handle[]>(() => [{ kind: 'x', at: p, onDrag: pivot.set, label: 'pivot' }], [p, pivot.set])

  // A document's score is proportional to 1/normaliser, so the change in its score is Z / Z'.
  const boost = (z: number) => z / ((1 - s) * p + s * z)
  return (
    <Interactive
      title="Pivoted length normalisation"
      caption="The dashed line is the old normalisation factor Z (cosine length or number of distinct terms). The pivoted factor is a line through (p, p) with slope s. Documents longer than the pivot get a smaller divisor, so their scores rise; shorter documents get a larger one. Drag the vertical line to move the pivot."
      controls={
        <>
          <ParamSlider label="slope s" param={slope} />
          <ParamSlider label="pivot p" param={pivot} format={(v) => v.toFixed(0)} />
        </>
      }
      readout={
        <>
          <Readout label={`score change, Z = ${SHORT}`} value={`× ${formatNumber(boost(SHORT))}`} />
          <Readout label={`score change, Z = ${LONG}`} value={`× ${formatNumber(boost(LONG))}`} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="old normalisation factor Z (document length)"
        yLabel="normaliser"
        xRange={[0, 400]}
        yRange={[0, 400]}
        handles={handles}
        height={320}
      />
    </Interactive>
  )
}
