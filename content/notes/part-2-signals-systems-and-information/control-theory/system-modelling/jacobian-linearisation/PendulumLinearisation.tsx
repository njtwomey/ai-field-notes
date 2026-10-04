import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { rk4 } from '../../_shared/control'

const G_OVER_L = 9.81
const DAMPING = 0.5
const DT = 0.01
const T_END = 8

function simulate(theta0: number, linear: boolean) {
  const f = ([th, om]: number[]) => [om, -G_OVER_L * (linear ? th : Math.sin(th)) - DAMPING * om]
  const t: number[] = []
  const y: number[] = []
  let x = [theta0, 0]
  for (let k = 0; k <= T_END / DT; k++) {
    t.push(k * DT)
    y.push(x[0])
    x = rk4(f, x, DT)
  }
  return { t, y }
}

export function PendulumLinearisation() {
  const state = useFigureState({
    theta0: slider(2, 175, 30, { step: 1, label: 'initial angle θ₀ (degrees)', format: (v) => `${v}°` }),
  })

  const { series, maxError } = useMemo(() => {
    const rad = (state.theta0 * Math.PI) / 180
    const nl = simulate(rad, false)
    const lin = simulate(rad, true)
    const deg = (v: number[]) => v.map((r) => (r * 180) / Math.PI)
    const s: SeriesSpec[] = [
      { name: 'nonlinear pendulum', type: 'line', x: nl.t, y: deg(nl.y), slot: 0 },
      { name: 'linearisation', type: 'line', x: lin.t, y: deg(lin.y), slot: 1, dashed: true },
    ]
    const err = Math.max(...nl.y.map((v, i) => Math.abs(v - lin.y[i])))
    return { series: s, maxError: (err * 180) / Math.PI }
  }, [state.theta0])

  const xAxis = useAxis({ label: 'time t (s)', range: [0, T_END] })
  const yAxis = useAxis({ label: 'angle θ (degrees)', range: [-180, 180] })
  return (
    <Figure
      title="Where the linear pendulum stops being accurate"
      state={state}
      caption="The damped pendulum released from rest at angle θ₀, simulated with the full sin θ (solid) and with its linearisation about the hanging equilibrium, sin θ ≈ θ (dashed). Drag the starting point on the vertical axis or use the slider. Below about 20° the curves are indistinguishable. At larger angles the true pendulum swings more slowly than the linear model predicts, so the two drift out of phase; the linear period 2π/3.12 ≈ 2.0 s does not depend on amplitude, the true one does."

      readouts={<Readout label="largest gap over 8 s" value={`${formatNumber(maxError)}°`} />}
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
        <Handle kind="point" at={[0, state.theta0]} onDrag={([, y]) => state.set('theta0', y)} label="θ₀" />
      </Plot>
    </Figure>
  )
}
