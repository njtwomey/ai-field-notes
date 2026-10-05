import { useMemo } from 'react'
import {
  Curve,
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
import { normal, stream } from 'aifn-compute/foundation/random'

const T = 300

export function EcmSimulator() {
  const state = useFigureState({
    alphaY: float(-0.2, { min: -1, max: 0, step: 0.05, label: 'adjustment of y, α_y', format: (v) => v.toFixed(2) }),
    alphaX: float(0, { min: 0, max: 1, step: 0.05, label: 'adjustment of x, α_x', format: (v) => v.toFixed(2) }),
    count: int(10, { min: 1, max: 50, step: 1, label: 'draws', format: (v) => String(v) }),
    seed: int(4, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  // Draw k has its own stream, so adding draws leaves the existing ones unchanged. Draw 0 is the pair plotted above.
  const sims = useMemo(
    () =>
      Array.from({ length: state.count }, (_, k) => {
        const g = stream(state.seed * 1000 + k)
        const x = [0]
        const y = [0]
        for (let t = 1; t < T; t++) {
          // Equilibrium y = x. A positive gap pulls y down (alpha_y < 0) and pushes x up (alpha_x > 0).
          const gap = y[t - 1] - x[t - 1]
          x.push(x[t - 1] + state.alphaX * gap + normal(g))
          y.push(y[t - 1] + state.alphaY * gap + normal(g))
        }
        return { x, y, gap: y.map((v, i) => v - x[i]) }
      }),
    [state.alphaY, state.alphaX, state.seed, state.count],
  )

  const charts = useMemo(() => {
    const t = Array.from({ length: T }, (_, i) => i)
    const [sim, ...others] = sims
    const levels = [
      { name: 'x_t', x: t, y: sim.x, slot: 0 },
      { name: 'y_t', x: t, y: sim.y, slot: 1 },
    ] as const
    // The gap is an AR(1) with coefficient ρ and shock variance 2, from 0: var_t = ρ² var_{t−1} + 2.
    const rho = 1 + state.alphaY - state.alphaX
    const band: number[] = [0]
    for (let i = 1; i < T; i++) band.push(Math.sqrt(rho * rho * band[i - 1] ** 2 + 2))
    const gap: SeriesSpec[] = [
      ...others.map((o): SeriesSpec => ({
        name: 'gaps of further draws',
        type: 'line',
        x: t,
        y: o.gap,
        slot: 2,
        thin: true,
      })),
      { name: 'gap y_t − x_t', type: 'line', x: t, y: sim.gap, slot: 2 },
      { name: '±2 sd of the gap', type: 'line', x: t, y: band.map((v) => 2 * v), slot: 3, dashed: true },
      { name: '±2 sd of the gap', type: 'line', x: t, y: band.map((v) => -2 * v), slot: 3, dashed: true },
      { name: 'equilibrium', type: 'line', x: [0, T - 1], y: [0, 0], muted: true, dashed: true },
    ]
    return { levels, gap }
  }, [sims, state.alphaY, state.alphaX])

  // The gap follows an AR(1): z_t = (1 + alpha_y - alpha_x) z_{t-1} + noise.
  const root = 1 + state.alphaY - state.alphaX
  const stable = Math.abs(root) < 1
  const halfLife = !stable
    ? 'never'
    : root <= 0
      ? 'under 1 step'
      : `${formatNumber(Math.log(0.5) / Math.log(root))} steps`

  const xAxis = useAxis({ label: 't', range: [0, T - 1] })
  const yAxis = useAxis({ label: 'level', hold: 'union' })
  const xAxis2 = useAxis({ label: 't', range: [0, T - 1] })
  const yAxis2 = useAxis({ label: 'gap', hold: 'union' })
  return (
    <Figure
      title="Error correction between two random walks"
      state={state}
      caption="Both series take a Gaussian step each period. In addition, y corrects by α_y times the previous gap y − x, and x by α_x times the same gap. With α_y = α_x = 0 the two are independent random walks and drift apart. With any correction the gap is stationary, so the series are cointegrated with β = 1: they wander together. The gap follows an AR(1) with coefficient 1 + α_y − α_x, so either series, or both, can do the correcting. The lower panel repeats the simulation: the bold line is the gap of the pair above, the light lines are the gaps of further independent draws (the draws slider sets how many), and the dashed curves are ±2 standard deviations of the gap. When |1 + α_y − α_x| < 1 the band levels off; otherwise it widens as √t."

      readouts={
        <>
          <Readout label="AR coefficient of the gap" value={formatNumber(root)} />
          <Readout label="half-life of a deviation" value={halfLife} />
          <Readout label="cointegrated" value={stable ? 'yes' : 'no'} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={240}>
          <Curve {...charts.levels[0]} />
          <Curve {...charts.levels[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={160}>
          {seriesLayers(charts.gap)}
        </Plot>
      </div>
    </Figure>
  )
}
