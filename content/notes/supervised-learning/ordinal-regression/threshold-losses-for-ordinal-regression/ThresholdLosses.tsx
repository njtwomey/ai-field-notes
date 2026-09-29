import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { intervalOf, softplus } from '../_shared/ordinal'

type Surrogate = 'hinge' | 'logistic'
const SURROGATES = [
  { value: 'hinge' as const, label: 'hinge' },
  { value: 'logistic' as const, label: 'logistic' },
]
const S = linspace(-6, 6, 481)
const GAP = 0.2

/** The surrogate penalty for a margin z: the score should sit on the correct side of a threshold by z. */
const penalty = (kind: Surrogate, z: number) => (kind === 'hinge' ? Math.max(0, 1 - z) : softplus(-z))

/** Immediate-threshold and all-threshold losses (Rennie & Srebro) for true class y (0-based), as functions of s. */
function losses(kind: Surrogate, th: number[], y: number, s: number) {
  let immediate = 0
  let all = 0
  th.forEach((t, j) => {
    // Thresholds below the true class should lie below s; the rest above it.
    const z = j < y ? s - t : t - s
    const l = penalty(kind, z)
    all += l
    if (j === y - 1 || j === y) immediate += l
  })
  return { immediate, all, error: Math.abs(intervalOf(s, th) - y) }
}

/**
 * The two threshold losses for a five-class problem as a function of the score, against the absolute error of the
 * class the score falls in. All-threshold steepens at every threshold crossed; immediate-threshold keeps one slope
 * past the neighbouring thresholds.
 */
export function ThresholdLosses() {
  const [kind, setKind] = useState<Surrogate>('hinge')
  const [y, setY] = useState(2)
  const [theta, setTheta] = useState([-3, -1, 1, 3])

  const setThreshold = (j: number) => (v: number) =>
    setTheta((th) => {
      const lo = j > 0 ? th[j - 1] + GAP : -5.5
      const hi = j < th.length - 1 ? th[j + 1] - GAP : 5.5
      const next = th.slice()
      next[j] = Math.round(Math.min(hi, Math.max(lo, v)) * 100) / 100
      return next
    })

  const [t0, t1, t2, t3] = theta
  const series = useMemo<XYSeries[]>(() => {
    const th = [t0, t1, t2, t3]
    const rows = S.map((s) => losses(kind, th, y - 1, s))
    return [
      { name: 'all threshold', type: 'line', x: S, y: rows.map((r) => r.all), slot: 0 },
      { name: 'immediate threshold', type: 'line', x: S, y: rows.map((r) => r.immediate), slot: 1 },
      { name: 'absolute error', type: 'line', x: S, y: rows.map((r) => r.error), slot: 2, dashed: true },
    ]
  }, [kind, y, t0, t1, t2, t3])

  const handles: Handle[] = theta.map((t, j) => ({ kind: 'x', at: t, label: `θ${j + 1}`, onDrag: setThreshold(j) }))
  // A score far into the worst class, to compare how the two losses grow with distance.
  const far = losses(kind, theta, y - 1, y <= 3 ? 5.5 : -5.5)

  return (
    <Interactive
      title="Immediate-threshold and all-threshold losses"
      caption={
        <MathText text="Five classes and thresholds $\theta_1 < \dots < \theta_4$. For the chosen true class, the chart plots each loss against the score $s$; the dashed step is the absolute error of the class $s$ falls in. Immediate-threshold only penalises the two thresholds that bound the true class, so its slope stays the same however many thresholds $s$ crosses. All-threshold adds a penalty for every threshold on the wrong side, so its slope rises by one at each threshold crossed. With the hinge it never falls below the absolute error; with the logistic surrogate it never falls below $\log 2$ times the absolute error. Drag the thresholds." />
      }
      controls={
        <>
          <ParamSlider label="true class y" value={y} onChange={setY} min={1} max={5} step={1} />
          <ParamChoice label="surrogate" value={kind} onChange={setKind} options={SURROGATES} />
        </>
      }
      readout={
        <>
          <Readout label="at the far end: absolute error" value={far.error} />
          <Readout label="all threshold" value={formatNumber(far.all)} />
          <Readout label="immediate threshold" value={formatNumber(far.immediate)} />
        </>
      }
    >
      <XYChart
        series={series}
        handles={handles}
        xRange={[-6, 6]}
        yRange={[0, undefined]}
        xLabel="score s"
        yLabel="loss"
        height={320}
        ariaLabel="Threshold losses against the score"
      />
    </Interactive>
  )
}
