import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { bode, feedbackStep, logspace } from '../../_shared/control'

const LOG_W: [number, number] = [-2, 2]
const W = logspace(LOG_W[0], LOG_W[1], 600)
const LW = W.map(Math.log10)
/** Open loop G(s) = 1 / (s(s+1)(0.2s+1)); the slider sets the gain K and the delay τ. */
const DEN = [0.2, 1.2, 1, 0]

/** First frequency (log10) at which `f` crosses `level` going downwards, by linear interpolation. */
function crossing(f: number[], level: number): number | null {
  for (let i = 1; i < f.length; i++) {
    if (f[i - 1] >= level && f[i] < level) {
      const a = (f[i - 1] - level) / (f[i - 1] - f[i])
      return LW[i - 1] + a * (LW[i] - LW[i - 1])
    }
  }
  return null
}
const at = (f: number[], lw: number) => {
  const i = Math.min(
    Math.max(
      1,
      LW.findIndex((v) => v >= lw),
    ),
    LW.length - 1,
  )
  const a = (lw - LW[i - 1]) / (LW[i] - LW[i - 1])
  return f[i - 1] + a * (f[i] - f[i - 1])
}

export function BodeMargins() {
  const state = useFigureState({
    logK: float(0, {
      min: -1,
      max: 1.2,
      step: 0.01,
      label: 'gain K',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    delay: float(0, { min: 0, max: 1.5, step: 0.05, label: 'delay τ (s)' }),
  })
  const k = 10 ** state.logK

  const view = useMemo(() => {
    const tf = { num: [k], den: DEN, delay: state.delay }
    const { mag, phase } = bode(tf, W)
    const lc = crossing(mag, 0)
    const l180 = crossing(phase, -180)
    const pm = lc === null ? null : 180 + at(phase, lc)
    const gm = l180 === null ? null : -at(mag, l180)
    const step = feedbackStep(tf, 1, 30, 0.01)
    return { mag, phase, lc, l180, pm, gm, step }
  }, [k, state.delay])

  const guide = (name: string, lw: number | null, y: [number, number], slot: number): SeriesSpec[] =>
    lw === null ? [] : [{ name, type: 'line', x: [lw, lw], y, slot, dashed: true }]
  const magSeries: SeriesSpec[] = [
    { name: '|L(iω)| (dB)', type: 'line', x: LW, y: view.mag, slot: 0 },
    { name: '0 dB', type: 'line', x: LOG_W, y: [0, 0], muted: true },
    ...guide('gain crossover ωc', view.lc, [-80, 60], 1),
    ...guide('phase crossover ω180', view.l180, [-80, 60], 2),
    ...(view.l180 !== null && view.gm !== null
      ? [{ name: 'gain margin', type: 'line' as const, x: [view.l180, view.l180], y: [-view.gm, 0], emphasis: true }]
      : []),
  ]
  const phaseSeries: SeriesSpec[] = [
    { name: '∠L(iω) (degrees)', type: 'line', x: LW, y: view.phase, slot: 0 },
    { name: '−180°', type: 'line', x: LOG_W, y: [-180, -180], muted: true },
    ...guide('gain crossover ωc', view.lc, [-360, -45], 1),
    ...guide('phase crossover ω180', view.l180, [-360, -45], 2),
    ...(view.lc !== null && view.pm !== null
      ? [
          {
            name: 'phase margin',
            type: 'line' as const,
            x: [view.lc, view.lc],
            y: [-180, view.pm - 180],
            emphasis: true,
          },
        ]
      : []),
  ]
  const stable = view.pm !== null && view.pm > 0 && (view.gm === null || view.gm > 0)
  const stepSeries = [
    { name: 'reference', x: [0, 30], y: [1, 1], muted: true },
    { name: 'closed-loop output', x: view.step.t, y: view.step.y, slot: 0 },
  ] as const
  const fmtW = (lw: number | null) => (lw === null ? '—' : `${formatNumber(10 ** lw)} rad/s`)

  const xAxis = useAxis({ label: 'log₁₀ ω', range: LOG_W })
  const yAxis = useAxis({ label: 'magnitude (dB)', range: [-80, 60] })
  const xAxis2 = useAxis({ label: 'log₁₀ ω', range: LOG_W })
  const yAxis2 = useAxis({ label: 'phase (degrees)', range: [-360, -45] })
  const xAxis3 = useAxis({ label: 'time t (s)', range: [0, 30] })
  const yAxis3 = useAxis({ label: 'output y', range: [-0.5, 2.5] })
  return (
    <Figure
      title="Gain and phase margins"
      state={state}
      caption="Bode plot of L(s) = K e^(−sτ) / (s(s+1)(0.2s+1)) and the closed-loop step response. The gain margin (bar on the magnitude plot) is how far |L| is below 0 dB where the phase reaches −180°. The phase margin (bar on the phase plot) is how far the phase is above −180° where |L| crosses 0 dB. Raising K lifts the magnitude curve without changing the phase, so the gain crossover moves right, towards the phase crossover, and both margins shrink; at K = 6 they vanish. A delay leaves the magnitude alone and subtracts ωτ from the phase, eating phase margin. Smaller margins show up as more overshoot and ringing."

      readouts={
        <>
          <Readout label="gain crossover ωc" value={fmtW(view.lc)} />
          <Readout label="phase margin" value={view.pm === null ? '—' : `${formatNumber(view.pm)}°`} />
          <Readout label="phase crossover ω180" value={fmtW(view.l180)} />
          <Readout label="gain margin" value={view.gm === null ? '∞' : `${formatNumber(view.gm)} dB`} />
          <Readout label="closed loop" value={stable ? 'stable' : 'unstable'} />
        </>
      }
    >
      <div className="space-y-2">
        <Plot x={xAxis} y={yAxis} height={220}>
          {seriesLayers(magSeries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={220}>
          {seriesLayers(phaseSeries)}
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={200}>
          <Curve {...stepSeries[0]} />
          <Curve {...stepSeries[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
