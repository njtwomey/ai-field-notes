import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { metricsAt, negativeDensity, positiveDensity, posterior, scoreForPosterior } from '../_shared/binormal'

const S = linspace(-4, 7, 301)

/**
 * Two class-conditional score densities, optionally scaled by the class priors, with a draggable decision threshold.
 * Shaded areas are the false positives (negatives right of the threshold) and false negatives (positives left of it).
 */
export function ThresholdExplorer() {
  const pi = useParam(0.05, { min: 0.005, max: 0.5, step: 0.005 })
  const d = useParam(2, { min: 0.5, max: 4, step: 0.1 })
  const costRatio = useParam(20, { min: 1, max: 100, step: 1 })
  const t = useParam(scoreForPosterior(0.5, 0.05, 2), { min: -4, max: 7, step: 0.01 })
  const [joint, setJoint] = useState(true)

  const pStar = 1 / (1 + costRatio.value)
  const sStar = scoreForPosterior(pStar, pi.value, d.value)
  const sHalf = scoreForPosterior(0.5, pi.value, d.value)

  const series = useMemo<XYSeries[]>(() => {
    const wNeg = joint ? 1 - pi.value : 1
    const wPos = joint ? pi.value : 1
    const neg = S.map((s) => wNeg * negativeDensity(s))
    const pos = S.map((s) => wPos * positiveDensity(s, d.value))
    const right = S.filter((s) => s >= t.value)
    const left = S.filter((s) => s <= t.value)
    return [
      { name: 'negatives', type: 'line', x: S, y: neg, slot: 0 },
      { name: 'positives', type: 'line', x: S, y: pos, slot: 1 },
      {
        name: 'false positives',
        type: 'line',
        x: right,
        y: right.map((s) => wNeg * negativeDensity(s)),
        slot: 0,
        area: true,
      },
      {
        name: 'false negatives',
        type: 'line',
        x: left,
        y: left.map((s) => wPos * positiveDensity(s, d.value)),
        slot: 1,
        area: true,
      },
    ]
  }, [joint, pi.value, d.value, t.value])

  const top = joint ? 0.42 * Math.max(1 - pi.value, pi.value) : 0.42
  const guides = useMemo<XYSeries[]>(
    () => [
      { name: 'cost-optimal threshold', type: 'line', x: [sStar, sStar], y: [0, top], dashed: true, slot: 2 },
      { name: 'posterior 0.5', type: 'line', x: [sHalf, sHalf], y: [0, top], dashed: true, muted: true },
    ],
    [sStar, sHalf, top],
  )

  const all = useMemo(() => [...series, ...guides], [series, guides])

  const m = metricsAt(t.value, pi.value, d.value)
  const cost = m.fpRate + costRatio.value * m.fnRate
  const best = metricsAt(sStar, pi.value, d.value)
  const bestCost = best.fpRate + costRatio.value * best.fnRate
  const handles: Handle[] = [{ kind: 'x', at: t.value, label: 'threshold', onDrag: (x) => t.set(x) }]

  return (
    <Interactive
      title="Moving the threshold"
      caption="Scores are N(0, 1) for negatives and N(d, 1) for positives. With the switch on, each density is scaled by its class prior, so the areas are the joint probabilities that the metrics count. Drag the threshold. The dashed coloured line is the cost-optimal threshold for a false negative costing c times a false positive; the grey dashed line is where the posterior is 0.5. Lower the prevalence and watch the positive hump shrink and the posterior-0.5 line run off to the right."
      controls={
        <>
          <ParamSlider label="prevalence π" param={pi} />
          <ParamSlider label="separation d" param={d} />
          <ParamSlider label="cost ratio c = c_FN / c_FP" param={costRatio} />
          <ParamSlider label="threshold on the score" param={t} />
          <ParamSwitch label="scale densities by prior" checked={joint} onChange={setJoint} />
          <ParamButton onClick={() => t.set(sStar)}>Snap to cost-optimal</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="posterior at threshold" value={formatNumber(posterior(t.value, pi.value, d.value))} />
          <Readout label="optimal posterior threshold" value={formatNumber(pStar)} />
          <Readout label="precision" value={formatNumber(m.precision)} />
          <Readout label="recall" value={formatNumber(m.recall)} />
          <Readout label="F₁" value={formatNumber(m.f1)} />
          <Readout label="balanced accuracy" value={formatNumber(m.balancedAccuracy)} />
          <Readout label="expected cost (c_FP = 1)" value={formatNumber(cost)} />
          <Readout label="minimum expected cost" value={formatNumber(bestCost)} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="score s"
        yLabel={joint ? 'prior × density' : 'density'}
        xRange={[-4, 7]}
        yRange={[0, top]}
        handles={handles}
        series={all}
      />
    </Interactive>
  )
}
