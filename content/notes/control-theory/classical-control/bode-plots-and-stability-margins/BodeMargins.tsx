import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
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
  const logK = useParam(0, { min: -1, max: 1.2, step: 0.01 })
  const delay = useParam(0, { min: 0, max: 1.5, step: 0.05 })
  const k = 10 ** logK.value

  const view = useMemo(() => {
    const tf = { num: [k], den: DEN, delay: delay.value }
    const { mag, phase } = bode(tf, W)
    const lc = crossing(mag, 0)
    const l180 = crossing(phase, -180)
    const pm = lc === null ? null : 180 + at(phase, lc)
    const gm = l180 === null ? null : -at(mag, l180)
    const step = feedbackStep(tf, 1, 30, 0.01)
    return { mag, phase, lc, l180, pm, gm, step }
  }, [k, delay.value])

  const guide = (name: string, lw: number | null, y: [number, number], slot: number): XYSeries[] =>
    lw === null ? [] : [{ name, type: 'line', x: [lw, lw], y, slot, dashed: true }]
  const magSeries: XYSeries[] = [
    { name: '|L(iω)| (dB)', type: 'line', x: LW, y: view.mag, slot: 0 },
    { name: '0 dB', type: 'line', x: LOG_W, y: [0, 0], muted: true },
    ...guide('gain crossover ωc', view.lc, [-80, 60], 1),
    ...guide('phase crossover ω180', view.l180, [-80, 60], 2),
    ...(view.l180 !== null && view.gm !== null
      ? [{ name: 'gain margin', type: 'line' as const, x: [view.l180, view.l180], y: [-view.gm, 0], emphasis: true }]
      : []),
  ]
  const phaseSeries: XYSeries[] = [
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
  const stepSeries: XYSeries[] = [
    { name: 'reference', type: 'line', x: [0, 30], y: [1, 1], muted: true },
    { name: 'closed-loop output', type: 'line', x: view.step.t, y: view.step.y, slot: 0 },
  ]
  const fmtW = (lw: number | null) => (lw === null ? '—' : `${formatNumber(10 ** lw)} rad/s`)

  return (
    <Interactive
      title="Gain and phase margins"
      caption="Bode plot of L(s) = K e^(−sτ) / (s(s+1)(0.2s+1)) and the closed-loop step response. The gain margin (bar on the magnitude plot) is how far |L| is below 0 dB where the phase reaches −180°. The phase margin (bar on the phase plot) is how far the phase is above −180° where |L| crosses 0 dB. Raising K lifts the magnitude curve without changing the phase, so the gain crossover moves right, towards the phase crossover, and both margins shrink; at K = 6 they vanish. A delay leaves the magnitude alone and subtracts ωτ from the phase, eating phase margin. Smaller margins show up as more overshoot and ringing."
      controls={
        <>
          <ParamSlider label="gain K" param={logK} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="delay τ (s)" param={delay} />
        </>
      }
      readout={
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
        <XYChart
          series={magSeries}
          xLabel="log₁₀ ω"
          yLabel="magnitude (dB)"
          xRange={LOG_W}
          yRange={[-80, 60]}
          height={220}
        />
        <XYChart
          series={phaseSeries}
          xLabel="log₁₀ ω"
          yLabel="phase (degrees)"
          xRange={LOG_W}
          yRange={[-360, -45]}
          height={220}
        />
        <XYChart
          series={stepSeries}
          xLabel="time t (s)"
          yLabel="output y"
          xRange={[0, 30]}
          yRange={[-0.5, 2.5]}
          height={200}
        />
      </div>
    </Interactive>
  )
}
