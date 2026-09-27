import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { DOMAIN, ise, sample, summary, trueDensity } from '../../_shared/density'

const GRID = linspace(DOMAIN[0], DOMAIN[1], 400)
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
  const [n, setN] = useState(200)
  const width = useParam(0.5, { min: 0.05, max: 1.5, step: 0.01 })
  const origin = useParam(0, { min: -1.5, max: 1.5, step: 0.01 })
  const data = useMemo(() => sample(n, 23), [n])
  const stats = useMemo(() => summary(data), [data])
  const hist = useMemo(() => histogram(data, width.value, origin.value), [data, width.value, origin.value])
  const error = ise(GRID, GRID.map(hist.at))
  const scott = 3.49 * stats.sd * n ** (-1 / 3)
  const fd = 2 * stats.iqr * n ** (-1 / 3)
  const sturges = (stats.max - stats.min) / (Math.ceil(Math.log2(n)) + 1)
  const handles: Handle[] = [{ kind: 'x', at: origin.value, label: 'bin edge', onDrag: (x) => origin.set(x) }]

  return (
    <Interactive
      title="Bin width and bin origin"
      caption="A density histogram of a sample from a two-component Gaussian mixture (dashed). Drag the vertical line to move one bin edge, which shifts the whole grid of bins: with wide bins the shape changes noticeably, even though the data do not. The bin width trades bias (wide bins flatten the narrow mode) against variance (narrow bins are noisy). The buttons apply the Freedman–Diaconis and Scott rules."
      controls={
        <>
          <ParamSlider label="bin width h" param={width} />
          <ParamSlider label="bin edge position" param={origin} />
          <ParamSlider label="sample size n" value={n} onChange={setN} min={20} max={1000} step={10} />
          <div className="flex gap-2">
            <ParamButton onClick={() => width.set(fd)}>Freedman–Diaconis</ParamButton>
            <ParamButton onClick={() => width.set(scott)}>Scott</ParamButton>
          </div>
        </>
      }
      readout={
        <>
          <Readout label="integrated squared error" value={formatNumber(error)} />
          <Readout label="Freedman–Diaconis h" value={formatNumber(fd)} />
          <Readout label="Scott h" value={formatNumber(scott)} />
          <Readout label="Sturges h" value={formatNumber(sturges)} />
        </>
      }
    >
      <XYChart
        height={340}
        xLabel="x"
        yLabel="density"
        xRange={DOMAIN}
        yRange={[0, undefined]}
        handles={handles}
        series={[
          { name: 'histogram', type: 'bar', x: hist.centres, y: hist.heights, slot: 0 },
          { name: 'true density', type: 'line', x: GRID, y: TRUTH, dashed: true, slot: 1 },
        ]}
      />
    </Interactive>
  )
}
