import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { averageRunLength } from './arl'

const T = 300
const TIMES = Array.from({ length: T }, (_, t) => t + 1)

/** A stream whose mean shifts from 0 to δ, and the one-sided CUSUM statistic tuned to that shift. */
export function CusumChart() {
  const shiftAt = useParam(150, { min: 30, max: 270, step: 1 })
  const delta = useParam(1, { min: 0.25, max: 3, step: 0.05 })
  const h = useParam(4, { min: 0.5, max: 15, step: 0.1 })
  // Seed 3 shows a clean detection 5 steps after the shift; seed 2 shows a false alarm before it.
  const seed = useParam(3, { min: 1, max: 40, step: 1 })

  const r = useMemo(() => {
    const g = rng(seed.value)
    const x = TIMES.map((t) => (t >= shiftAt.value ? delta.value : 0) + g.normal())
    // Log-likelihood ratio of N(δ, 1) against N(0, 1) is δ(x − δ/2); dividing by δ gives increments x − k, k = δ/2.
    const k = delta.value / 2
    const s: number[] = []
    let acc = 0
    for (const v of x) {
      acc = Math.max(0, acc + v - k)
      s.push(acc)
    }
    return { x, s, k }
  }, [shiftAt.value, delta.value, seed.value])

  const alarm = r.s.findIndex((v) => v > h.value)
  const alarmAt = alarm === -1 ? null : alarm + 1
  const outcome =
    alarmAt === null ? 'no alarm' : alarmAt < shiftAt.value ? `false alarm at ${alarmAt}` : `alarm at ${alarmAt}`
  const arl0 = averageRunLength(0, r.k, h.value)
  const arl1 = averageRunLength(delta.value, r.k, h.value)

  const dataSeries: XYSeries[] = [
    { name: 'observations', type: 'scatter', x: TIMES, y: r.x, muted: true },
    {
      name: 'true mean',
      type: 'line',
      x: [1, shiftAt.value, shiftAt.value, T],
      y: [0, 0, delta.value, delta.value],
      slot: 0,
      dashed: true,
    },
  ]
  const cusumSeries: XYSeries[] = [
    { name: 'CUSUM statistic Sₙ', type: 'line', x: TIMES, y: r.s, slot: 1 },
    ...(alarmAt === null
      ? []
      : [{ name: 'alarm', type: 'scatter' as const, x: [alarmAt], y: [r.s[alarmAt - 1]], emphasis: true }]),
  ]
  // The threshold is a level on the statistic's axis, so it is dragged directly.
  const handles: Handle[] = [{ kind: 'y', at: h.value, label: 'h', onDrag: h.set }]
  const top = Math.max(h.value * 1.4, ...r.s.slice(0, Math.min(T, (alarmAt ?? T) + 40)))

  return (
    <Interactive
      title="CUSUM on a mean shift"
      caption="Top: a stream whose mean jumps from 0 to δ at the dashed step. Bottom: the one-sided CUSUM statistic Sₙ = max(0, Sₙ₋₁ + xₙ − δ/2). Before the shift it hovers near zero, pulled down by the −δ/2 drift; after it, it climbs at about δ/2 per step. Drag the threshold h: a lower h detects the shift sooner but raises false alarms before it, as the average run lengths in the readout show."
      controls={
        <>
          <ParamSlider label="shift time" param={shiftAt} format={(v) => String(v)} />
          <ParamSlider label="shift size δ (in σ)" param={delta} />
          <ParamSlider label="threshold h" param={h} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="outcome" value={outcome} />
          <Readout
            label="detection delay"
            value={alarmAt !== null && alarmAt >= shiftAt.value ? `${alarmAt - shiftAt.value} steps` : '—'}
          />
          <Readout label="ARL before the shift (in control)" value={formatNumber(arl0)} />
          <Readout label="ARL after the shift" value={formatNumber(arl1)} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={dataSeries} xLabel="n" yLabel="xₙ" xRange={[1, T]} height={200} />
        <XYChart
          series={cusumSeries}
          xLabel="n"
          yLabel="Sₙ"
          xRange={[1, T]}
          yRange={[0, Math.ceil(top)]}
          handles={handles}
          height={240}
        />
      </div>
    </Interactive>
  )
}
