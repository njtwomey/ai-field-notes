import { useMemo } from 'react'
import {
  choice,
  Figure,
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
import { sigmoid } from 'aifn-compute/numerics/special'

type Target = 'sine' | 'bump' | 'kink'
type Units = 'sigmoid' | 'relu'

const TARGETS: Record<Target, (x: number) => number> = {
  sine: (x) => Math.sin(2 * Math.PI * x),
  bump: (x) => Math.exp(-((x - 0.4) ** 2) / 0.01),
  kink: (x) => Math.abs(x - 0.6) - 0.3 * x,
}

const XS = toFlat(linspace(0, 1, 801))
const X_RANGE: [number, number] = [0, 1]

/**
 * One hidden layer with n units approximating f on [0, 1] from its values at n + 1 evenly spaced knots t_j.
 * Sigmoid units: a staircase, one steep sigmoid at each midpoint between knots, of height f(t_j) − f(t_{j−1}).
 * ReLU units: the piecewise-linear interpolant, one ReLU per knot whose weight is the change of slope there.
 */
function network(f: (x: number) => number, n: number, units: Units, steepness: number) {
  const t = toFlat(linspace(0, 1, n + 1))
  const ft = t.map(f)
  if (units === 'sigmoid') {
    const k = steepness * n
    return (x: number) => {
      let y = ft[0]
      for (let j = 1; j <= n; j++) y += (ft[j] - ft[j - 1]) * sigmoid(k * (x - (t[j - 1] + t[j]) / 2))
      return y
    }
  }
  const slope = ft.slice(1).map((v, j) => (v - ft[j]) * n)
  return (x: number) => {
    let y = ft[0] + slope[0] * Math.max(0, x)
    for (let j = 1; j < n; j++) y += (slope[j] - slope[j - 1]) * Math.max(0, x - t[j])
    return y
  }
}

export function StepApproximation() {
  const state = useFigureState({
    target: choice<Target>(
      [
        { value: 'sine', label: 'sine' },
        { value: 'bump', label: 'bump' },
        { value: 'kink', label: 'kink' },
      ],
      'sine',
      { label: 'target f' },
    ),
    units: choice<Units>(
      [
        { value: 'sigmoid', label: 'sigmoid' },
        { value: 'relu', label: 'ReLU' },
      ],
      'sigmoid',
      { label: 'hidden units' },
    ),
    n: int(10, { min: 1, max: 60, step: 1, label: 'number of units n' }),
    steepness: int(20, { min: 1, max: 200, step: 1, label: 'steepness (× n)', when: (v) => v.units === 'sigmoid' }),
  })

  const { series, error } = useMemo(() => {
    const f = TARGETS[state.target]
    const g = network(f, state.n, state.units, state.steepness)
    const truth = XS.map(f)
    const approx = XS.map(g)
    const s: SeriesSpec[] = [
      { name: 'target f', type: 'line', x: XS, y: truth, slot: 0 },
      {
        name: `network, ${state.n} ${state.units === 'relu' ? 'ReLU' : 'sigmoid'} units`,
        type: 'line',
        x: XS,
        y: approx,
        slot: 1,
      },
    ]
    return { series: s, error: Math.max(...truth.map((v, i) => Math.abs(v - approx[i]))) }
  }, [state.target, state.units, state.n, state.steepness])

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'f(x)', hold: 'union' })
  return (
    <Figure
      title="One hidden layer approximating a continuous function"
      state={state}
      caption="The network has n hidden units and a linear output. Sigmoid units build a staircase: each is a steep step placed between two knots, with height equal to the change of f between them. ReLU units build the piecewise-linear interpolant: each adds a change of slope at a knot. Increase n and the largest error falls; for sigmoids it falls like 1/n, for ReLUs on a smooth target like 1/n². The steepness slider sets the sigmoid slope in units of n."

      readouts={<Readout label="largest error on [0, 1]" value={formatNumber(error)} />}
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
