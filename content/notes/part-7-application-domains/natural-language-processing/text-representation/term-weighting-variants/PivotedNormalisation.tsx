import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'

const Z = Array.from({ length: 101 }, (_, i) => i * 4)
const SHORT = 40
const LONG = 320

/** The pivoted normaliser (1 − s)p + sZ against the old normaliser Z: a rotation of the identity line about (p, p). */
export function PivotedNormalisation() {
  const state = useFigureState({
    slope: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'slope s' }),
    pivot: int(120, { min: 20, max: 380, step: 1, label: 'pivot p', format: (v) => v.toFixed(0) }),
  })
  const s = state.slope
  const p = state.pivot

  const series = useMemo(
    () =>
      [
        { name: 'old normaliser Z', x: Z, y: Z, muted: true, dashed: true },
        { name: 'pivoted (1 − s)p + sZ', x: Z, y: Z.map((z) => (1 - s) * p + s * z), slot: 0 },
        { name: 'pivot (p, p)', x: [p], y: [p], emphasis: true },
      ] as const,
    [s, p],
  )
  const handles = useMemo<Handle[]>(
    () => [{ kind: 'x', at: p, onDrag: (v: number) => state.set('pivot', v), label: 'pivot' }],
    [p, (v: number) => state.set('pivot', v)],
  )

  // A document's score is proportional to 1/normaliser, so the change in its score is Z / Z'.
  const boost = (z: number) => z / ((1 - s) * p + s * z)
  const xAxis = useAxis({ label: 'old normalisation factor Z (document length)', range: [0, 400] })
  const yAxis = useAxis({ label: 'normaliser', range: [0, 400] })
  return (
    <Figure
      title="Pivoted length normalisation"
      state={state}
      caption="The dashed line is the old normalisation factor Z (cosine length or number of distinct terms). The pivoted factor is a line through (p, p) with slope s. Documents longer than the pivot get a smaller divisor, so their scores rise; shorter documents get a larger one. Drag the vertical line to move the pivot."

      readouts={
        <>
          <Readout label={`score change, Z = ${SHORT}`} value={`× ${formatNumber(boost(SHORT))}`} />
          <Readout label={`score change, Z = ${LONG}`} value={`× ${formatNumber(boost(LONG))}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Points {...series[2]} />
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
