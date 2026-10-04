import { useMemo } from 'react'
import {
  Bars,
  Button,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { DOMAIN, ise, sample, summary, trueDensity } from '../../_shared/density'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const GRID = toFlat(linspace(DOMAIN[0], DOMAIN[1], 400))
const TRUTH = GRID.map(trueDensity)

/** Density histogram with bins [origin + kh, origin + (k+1)h) covering the plotting domain. */
function histogram(data: number[], h: number, origin: number) {
  const first = origin + Math.floor((DOMAIN[0] - origin) / h) * h
  const bins = Math.ceil((DOMAIN[1] - first) / h)
  const counts = new Array<number>(bins).fill(0)
  for (const x of data) {
    const k = Math.floor((x - first) / h)
    if (k >= 0 && k < bins) counts[k]++
  }
  const heights = counts.map((c) => c / (data.length * h))
  const centres = counts.map((_, k) => first + (k + 0.5) * h)
  const at = (x: number) => heights[Math.floor((x - first) / h)] ?? 0
  return { centres, heights, at }
}

export function HistogramExplorer() {
  const state = useFigureState({
    width: float(0.5, { min: 0.05, max: 1.5, step: 0.01, label: 'bin width h' }),
    origin: float(0, { min: -1.5, max: 1.5, step: 0.01, label: 'bin edge position' }),
    n: int(200, { min: 20, max: 1000, step: 10, label: 'sample size n' }),
  })
  const data = useMemo(() => sample(state.n, 23), [state.n])
  const stats = useMemo(() => summary(data), [data])
  const hist = useMemo(() => histogram(data, state.width, state.origin), [data, state.width, state.origin])
  const error = ise(GRID, GRID.map(hist.at))
  const scott = 3.49 * stats.sd * state.n ** (-1 / 3)
  const fd = 2 * stats.iqr * state.n ** (-1 / 3)
  const sturges = (stats.max - stats.min) / (Math.ceil(Math.log2(state.n)) + 1)

  const xAxis = useAxis({ label: 'x', range: DOMAIN })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Bin width and bin origin"
      state={state}
      caption="A density histogram of a sample from a two-component Gaussian mixture (dashed). Drag the vertical line to move one bin edge, which shifts the whole grid of bins: with wide bins the shape changes noticeably, even though the data do not. The bin width trades bias (wide bins flatten the narrow mode) against variance (narrow bins are noisy). The buttons apply the Freedman–Diaconis and Scott rules."
      controls={
        <>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => state.set('width', fd)}>
              Freedman–Diaconis
            </Button>
            <Button variant="outline" size="sm" onClick={() => state.set('width', scott)}>
              Scott
            </Button>
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="integrated squared error" value={formatNumber(error)} />
          <Readout label="Freedman–Diaconis h" value={formatNumber(fd)} />
          <Readout label="Scott h" value={formatNumber(scott)} />
          <Readout label="Sturges h" value={formatNumber(sturges)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Bars name="histogram" x={hist.centres} y={hist.heights} slot={0} />
        <Curve name="true density" x={GRID} y={TRUTH} dashed slot={1} />
        <Handle {...state.handle('origin', { label: 'bin edge' })} />
      </Plot>
    </Figure>
  )
}
