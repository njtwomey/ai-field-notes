import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'

const bump = (x: number, mu: number, sd: number) => Math.exp(-0.5 * ((x - mu) / sd) ** 2)
const unnormalised = (x: number) => 0.6 * bump(x, 0.3, 0.08) + 0.4 * bump(x, 0.72, 0.1)

// Normalise the target on [0, 1] once, by the trapezoid rule, so that the acceptance rate is exactly 1/M.
const GRID = toFlat(linspace(0, 1, 2001))
const Z = GRID.reduce((s, x, i) => s + (i === 0 || i === GRID.length - 1 ? 0.5 : 1) * unnormalised(x), 0) / 2000
const target = (x: number) => unnormalised(x) / Z
const PEAK = Math.max(...GRID.map(target))
const BINS = 40

/**
 * Rejection sampling from a two-bump density on [0, 1] with a uniform proposal. Each proposal x gets a height
 * u · M · q(x); it is accepted when the height falls under the target density.
 */
export function RejectionSampler() {
  const state = useFigureState({
    m: float(3, { min: 1, max: 6, step: 0.1, label: 'envelope constant M' }),
    n: int(1500, { min: 200, max: 4000, step: 100, label: 'proposals' }),
  })

  const { scatter, histogram, rate } = useMemo(() => {
    const r = stream(4)
    const xs: number[] = []
    const hs: number[] = []
    const accepted: number[] = []
    const groups: number[] = []
    for (let i = 0; i < state.n; i++) {
      const x = uniform(r)
      const h = uniform(r) * state.m
      const ok = h <= target(x)
      xs.push(x)
      hs.push(h)
      groups.push(ok ? 0 : 1)
      if (ok) accepted.push(x)
    }
    const counts = new Array(BINS).fill(0)
    for (const x of accepted) counts[Math.min(Math.floor(x * BINS), BINS - 1)]++
    const width = 1 / BINS
    return {
      scatter: {
        name: 'proposals',
        type: 'scatter',
        x: xs,
        y: hs,
        group: groups,
        groupNames: ['accepted', 'rejected'],
      } satisfies SeriesSpec,
      histogram: {
        name: 'accepted samples',
        type: 'bar',
        x: counts.map((_, i) => (i + 0.5) * width),
        y: counts.map((c) => (accepted.length ? c / (accepted.length * width) : 0)),
        slot: 0,
      } satisfies SeriesSpec,
      rate: accepted.length / state.n,
    }
  }, [state.m, state.n])

  const curves = useMemo((): SeriesSpec[] => {
    const xs = toFlat(linspace(0, 1, 301))
    return [
      { name: 'target p(x)', type: 'line', x: xs, y: xs.map(target), slot: 2 },
      { name: 'envelope M·q(x)', type: 'line', x: [0, 1], y: [state.m, state.m], slot: 3, dashed: true },
    ]
  }, [state.m])
  const density = curves[0]
  const valid = state.m >= PEAK

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'height u·M·q(x)', range: [0, 6] })
  const xAxis2 = useAxis({ label: 'x', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'density', range: [0, 4] })
  return (
    <Figure
      title="Accept what falls under the curve"
      state={state}
      caption="Each proposal x is drawn uniformly on [0, 1] and given a uniform height between 0 and M. It is kept when the height falls under the target density. Kept points are uniform under the curve, so their x values follow the target. A larger M wastes more proposals: the acceptance rate is 1/M. When M drops below the peak of the target, the envelope no longer covers it, the tops of the bumps are cut off, and the histogram on the right is wrong."

      readouts={
        <>
          <Readout label="acceptance rate" value={`${(100 * rate).toFixed(1)}%`} />
          <Readout label="1/M" value={`${(100 / state.m).toFixed(1)}%`} />
          <Readout label="peak of p" value={formatNumber(PEAK)} />
          <Readout label="envelope" value={valid ? 'covers p' : 'too low: biased'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers([scatter, ...curves])}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          {seriesLayers([histogram, density])}
        </Plot>
      </div>
    </Figure>
  )
}
