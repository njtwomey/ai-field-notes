import { useMemo } from 'react'
import {
  Button,
  choice,
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
import { DOMAIN, ise, sample, summary, trueDensity } from '../../_shared/density'
import { kde, KERNEL_OPTIONS, silverman, type Kernel } from './kde'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const GRID = toFlat(linspace(DOMAIN[0], DOMAIN[1], 160))
const TRUTH = GRID.map(trueDensity)
const H_GRID = toFlat(linspace(0.03, 1.2, 40))

export function BandwidthExplorer() {
  const state = useFigureState({
    h: float(0.3, { min: 0.03, max: 1.2, step: 0.01, label: 'bandwidth h' }),
    n: int(200, { min: 20, max: 500, step: 10, label: 'sample size n' }),
    kind: choice<Kernel>(KERNEL_OPTIONS, 'gaussian', { label: 'kernel' }),
  })
  const data = useMemo(() => sample(state.n, 17), [state.n])
  const stats = useMemo(() => summary(data), [data])
  const rule = silverman(stats.sd, stats.iqr, state.n)
  // Integrated squared error over a grid of bandwidths: the curve on the right, recomputed only when the data change.
  const curve = useMemo(() => H_GRID.map((b) => ise(GRID, kde(data, GRID, b, state.kind))), [data, state.kind])
  const best = H_GRID[curve.indexOf(Math.min(...curve))]
  const estimate = useMemo(() => kde(data, GRID, state.h, state.kind), [data, state.h, state.kind])

  const xAxis = useAxis({ label: 'x', range: DOMAIN })
  const yAxis = useAxis({ label: 'density', range: [-0.02, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'bandwidth h', hold: 'union' })
  const yAxis2 = useAxis({ label: 'integrated squared error', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Bandwidth controls the bias–variance trade-off"
      state={state}
      caption="Left: a kernel density estimate (solid) of a sample from a two-component Gaussian mixture (dashed), with the sample as a rug along the bottom. Right: the integrated squared error of the estimate for every bandwidth. Drag the line labelled h on the right, or use the slider. Small h gives a spiky, high-variance estimate; large h blurs the narrow mode. Silverman's rule assumes a single Gaussian and chooses too wide a bandwidth for the narrow mode."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => state.set('h', rule)}>
            Use Silverman&apos;s rule
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="integrated squared error" value={formatNumber(ise(GRID, estimate))} />
          <Readout label="Silverman's h" value={formatNumber(rule)} />
          <Readout label="best h on this sample" value={formatNumber(best)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve name="true density" x={GRID} y={TRUTH} dashed slot={1} />
          <Curve name="estimate" x={GRID} y={estimate} slot={0} />
          <Points name="sample" x={data} y={data.map(() => -0.01)} muted />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve name="ISE" x={H_GRID} y={curve} slot={2} />
          <Points name="current h" x={[state.h]} y={[ise(GRID, estimate)]} emphasis />
          <Handle {...state.handle('h', { label: 'h' })} />
        </Plot>
      </div>
    </Figure>
  )
}
