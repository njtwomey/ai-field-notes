import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { DOMAIN, ise, sample, summary, trueDensity } from '../../_shared/density'
import { kde, KERNEL_OPTIONS, silverman, type Kernel } from './kde'

const GRID = linspace(DOMAIN[0], DOMAIN[1], 160)
const TRUTH = GRID.map(trueDensity)
const H_GRID = linspace(0.03, 1.2, 40)

export function BandwidthExplorer() {
  const [n, setN] = useState(200)
  const [kind, setKind] = useState<Kernel>('gaussian')
  const h = useParam(0.3, { min: 0.03, max: 1.2, step: 0.01 })
  const data = useMemo(() => sample(n, 17), [n])
  const stats = useMemo(() => summary(data), [data])
  const rule = silverman(stats.sd, stats.iqr, n)
  // Integrated squared error over a grid of bandwidths: the curve on the right, recomputed only when the data change.
  const curve = useMemo(() => H_GRID.map((b) => ise(GRID, kde(data, GRID, b, kind))), [data, kind])
  const best = H_GRID[curve.indexOf(Math.min(...curve))]
  const estimate = useMemo(() => kde(data, GRID, h.value, kind), [data, h.value, kind])
  const handles: Handle[] = [{ kind: 'x', at: h.value, label: 'h', onDrag: (x) => h.set(x) }]

  return (
    <Interactive
      title="Bandwidth controls the bias–variance trade-off"
      caption="Left: a kernel density estimate (solid) of a sample from a two-component Gaussian mixture (dashed), with the sample as a rug along the bottom. Right: the integrated squared error of the estimate for every bandwidth. Drag the line labelled h on the right, or use the slider. Small h gives a spiky, high-variance estimate; large h blurs the narrow mode. Silverman's rule assumes a single Gaussian and chooses too wide a bandwidth for the narrow mode."
      controls={
        <>
          <ParamSlider label="bandwidth h" param={h} />
          <ParamSlider label="sample size n" value={n} onChange={setN} min={20} max={500} step={10} />
          <ParamChoice label="kernel" value={kind} onChange={setKind} options={KERNEL_OPTIONS} />
          <ParamButton onClick={() => h.set(rule)}>Use Silverman&apos;s rule</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="integrated squared error" value={formatNumber(ise(GRID, estimate))} />
          <Readout label="Silverman's h" value={formatNumber(rule)} />
          <Readout label="best h on this sample" value={formatNumber(best)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={320}
          xLabel="x"
          yLabel="density"
          xRange={DOMAIN}
          yRange={[-0.02, undefined]}
          series={[
            { name: 'true density', type: 'line', x: GRID, y: TRUTH, dashed: true, slot: 1 },
            { name: 'estimate', type: 'line', x: GRID, y: estimate, slot: 0 },
            { name: 'sample', type: 'scatter', x: data, y: data.map(() => -0.01), muted: true },
          ]}
        />
        <XYChart
          height={320}
          xLabel="bandwidth h"
          yLabel="integrated squared error"
          yRange={[0, undefined]}
          handles={handles}
          series={[
            { name: 'ISE', type: 'line', x: H_GRID, y: curve, slot: 2 },
            { name: 'current h', type: 'scatter', x: [h.value], y: [ise(GRID, estimate)], emphasis: true },
          ]}
        />
      </div>
    </Interactive>
  )
}
