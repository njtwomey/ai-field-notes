import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type Handle } from '@/components/viz'

const TOTAL = 10_000
const PEAK = 1
const POINTS = 501
const Y_RANGE: [number, number] = [0, 1.05]
const X_RANGE: [number, number] = [0, TOTAL]

/** Multiply by 0.1 at 50% and at 75% of training. */
const stepDecay = (t: number) => PEAK * 0.1 ** ((t >= 0.5 * TOTAL ? 1 : 0) + (t >= 0.75 * TOTAL ? 1 : 0))

const cosine = (t: number, from: number, to: number) =>
  t <= from ? PEAK : 0.5 * PEAK * (1 + Math.cos((Math.PI * (t - from)) / (to - from)))

/** Linear warmup over `w` steps, then cosine decay to zero at the end of training. */
const warmupCosine = (t: number, w: number) => (w > 0 && t < w ? (PEAK * t) / w : cosine(t, w, TOTAL))

/** The transformer schedule: linear warmup, then decay as 1/√t, scaled to reach the peak at t = w. */
const inverseSqrt = (t: number, w: number) =>
  PEAK * Math.min(t / Math.max(w, 1), Math.sqrt(Math.max(w, 1) / Math.max(t, 1)))

export function ScheduleExplorer() {
  const warmup = useParam(1000, { min: 0, max: 3000, step: 50 })
  const w = warmup.value

  const series = useMemo(() => {
    const ts = Array.from({ length: POINTS }, (_, i) => (i * TOTAL) / (POINTS - 1))
    return [
      { name: 'step decay', type: 'line' as const, x: ts, y: ts.map(stepDecay), slot: 0 },
      { name: 'cosine', type: 'line' as const, x: ts, y: ts.map((t) => cosine(t, 0, TOTAL)), slot: 1 },
      { name: 'warmup + cosine', type: 'line' as const, x: ts, y: ts.map((t) => warmupCosine(t, w)), slot: 2 },
      { name: 'warmup + 1/√t', type: 'line' as const, x: ts, y: ts.map((t) => inverseSqrt(t, w)), slot: 3 },
    ]
  }, [w])

  const handles: Handle[] = [{ kind: 'x', at: w, label: 'end of warmup', onDrag: warmup.set }]
  const mean = (f: (t: number) => number) => {
    let s = 0
    for (let t = 0; t < TOTAL; t++) s += f(t)
    return s / TOTAL
  }

  return (
    <Interactive
      title="Learning-rate schedules"
      caption={`Step size as a fraction of its peak over ${TOTAL.toLocaleString('en')} steps. Step decay divides by 10 at 50% and 75% of training. Cosine decays from the peak to zero along half a cosine period. The warmup schedules rise linearly from zero to the peak; drag the vertical line or use the slider to set the warmup length. After warmup, one follows a cosine to zero and the other decays as 1/√t, as in the original transformer.`}
      controls={<ParamSlider label="warmup steps" param={warmup} />}
      readout={
        <>
          <Readout label="mean step, step decay" value={formatNumber(mean(stepDecay))} />
          <Readout label="mean step, cosine" value={formatNumber(mean((t) => cosine(t, 0, TOTAL)))} />
          <Readout label="mean step, warmup + cosine" value={formatNumber(mean((t) => warmupCosine(t, w)))} />
          <Readout label="mean step, warmup + 1/√t" value={formatNumber(mean((t) => inverseSqrt(t, w)))} />
        </>
      }
    >
      <XYChart
        series={series}
        xRange={X_RANGE}
        yRange={Y_RANGE}
        xLabel="step t"
        yLabel="η_t / η_max"
        handles={handles}
        height={320}
      />
    </Interactive>
  )
}
