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
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { averageRunLength } from './arl'
import { normal, stream } from 'aifn-compute/foundation/random'

const T = 300
const TIMES = Array.from({ length: T }, (_, t) => t + 1)

/** A stream whose mean shifts from 0 to δ, and the one-sided CUSUM statistic tuned to that shift. */
export function CusumChart() {
  const state = useFigureState({
    shiftAt: slider(30, 270, 150, { step: 1, label: 'shift time', format: (v) => String(v) }),
    delta: float(1, { min: 0.25, max: 3, step: 0.05, label: 'shift size δ (in σ)' }),
    h: slider(0.5, 15, 4, { step: 0.1, label: 'threshold h' }),
    seed: int(3, { min: 1, max: 40, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  // Seed 3 shows a clean detection 5 steps after the shift; seed 2 shows a false alarm before it.

  const r = useMemo(() => {
    const g = stream(state.seed)
    const x = TIMES.map((t) => (t >= state.shiftAt ? state.delta : 0) + normal(g))
    // Log-likelihood ratio of N(δ, 1) against N(0, 1) is δ(x − δ/2); dividing by δ gives increments x − k, k = δ/2.
    const k = state.delta / 2
    const s: number[] = []
    let acc = 0
    for (const v of x) {
      acc = Math.max(0, acc + v - k)
      s.push(acc)
    }
    return { x, s, k }
  }, [state.shiftAt, state.delta, state.seed])

  const alarm = r.s.findIndex((v) => v > state.h)
  const alarmAt = alarm === -1 ? null : alarm + 1
  const outcome =
    alarmAt === null ? 'no alarm' : alarmAt < state.shiftAt ? `false alarm at ${alarmAt}` : `alarm at ${alarmAt}`
  const arl0 = averageRunLength(0, r.k, state.h)
  const arl1 = averageRunLength(state.delta, r.k, state.h)

  const dataSeries = [
    { name: 'observations', x: TIMES, y: r.x, muted: true },
    {
      name: 'true mean',
      x: [1, state.shiftAt, state.shiftAt, T],
      y: [0, 0, state.delta, state.delta],
      slot: 0,
      dashed: true,
    },
  ] as const
  const cusumSeries: SeriesSpec[] = [
    { name: 'CUSUM statistic Sₙ', type: 'line', x: TIMES, y: r.s, slot: 1 },
    ...(alarmAt === null
      ? []
      : [{ name: 'alarm', type: 'scatter' as const, x: [alarmAt], y: [r.s[alarmAt - 1]], emphasis: true }]),
  ]
  // The threshold is a level on the statistic's axis, so it is dragged directly.
  const top = Math.max(state.h * 1.4, ...r.s.slice(0, Math.min(T, (alarmAt ?? T) + 40)))

  const xAxis = useAxis({ label: 'n', range: [1, T] })
  const yAxis = useAxis({ label: 'xₙ', hold: 'union' })
  const xAxis2 = useAxis({ label: 'n', range: [1, T] })
  const yAxis2 = useAxis({ label: 'Sₙ', range: [0, Math.ceil(top)] })
  return (
    <Figure
      title="CUSUM on a mean shift"
      state={state}
      caption="Top: a stream whose mean jumps from 0 to δ at the dashed step. Bottom: the one-sided CUSUM statistic Sₙ = max(0, Sₙ₋₁ + xₙ − δ/2). Before the shift it hovers near zero, pulled down by the −δ/2 drift; after it, it climbs at about δ/2 per step. Drag the threshold h: a lower h detects the shift sooner but raises false alarms before it, as the average run lengths in the readout show."

      readouts={
        <>
          <Readout label="outcome" value={outcome} />
          <Readout
            label="detection delay"
            value={alarmAt !== null && alarmAt >= state.shiftAt ? `${alarmAt - state.shiftAt} steps` : '—'}
          />
          <Readout label="ARL before the shift (in control)" value={formatNumber(arl0)} />
          <Readout label="ARL after the shift" value={formatNumber(arl1)} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={200}>
          <Points {...dataSeries[0]} />
          <Curve {...dataSeries[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={240}>
          {seriesLayers(cusumSeries)}
          <Handle {...state.handle('h', { axis: 'y', label: 'h' })} />
        </Plot>
      </div>
    </Figure>
  )
}
