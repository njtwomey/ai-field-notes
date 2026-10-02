import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { majorityTable, ordinalScores, shiftedTable } from './metrics'

type Prior = 'uniform' | 'skewed'
const PRIORS: Record<Prior, number[]> = {
  uniform: [0.2, 0.2, 0.2, 0.2, 0.2],
  skewed: [0.05, 0.15, 0.5, 0.2, 0.1],
}
const PRIOR_OPTIONS = [
  { value: 'uniform' as const, label: 'uniform' },
  { value: 'skewed' as const, label: 'skewed' },
]
const CLASSES = [1, 2, 3, 4, 5]
const DISTANCES = [1, 2, 3, 4]
const show = (v: number) => (Number.isFinite(v) ? formatNumber(v) : 'undefined')

/**
 * Five ordered classes. A fraction of every class is predicted a fixed distance away; the chart shows how each metric
 * responds to that distance at a fixed error rate, and the heatmap shows the expected confusion matrix.
 */
export function MetricResponse() {
  const [prior, setPrior] = useState<Prior>('uniform')
  const [rate, setRate] = useState(0.3)
  const [majority, setMajority] = useState(false)
  const distance = useParam(1, { min: 1, max: 4, step: 1 })

  const series = useMemo<XYSeries[]>(() => {
    const rows = DISTANCES.map((d) => ordinalScores(shiftedTable(PRIORS[prior], rate, d)))
    return [
      { name: 'accuracy', type: 'line', x: DISTANCES, y: rows.map((r) => r.accuracy), slot: 0 },
      { name: 'quadratic weighted kappa', type: 'line', x: DISTANCES, y: rows.map((r) => r.quadraticKappa), slot: 1 },
      { name: "Kendall's τ_b", type: 'line', x: DISTANCES, y: rows.map((r) => r.kendallTauB), slot: 2 },
      { name: 'MAE', type: 'line', x: DISTANCES, y: rows.map((r) => r.mae), slot: 3, dashed: true },
    ]
  }, [prior, rate])

  const table = majority ? majorityTable(PRIORS[prior]) : shiftedTable(PRIORS[prior], rate, distance.value)
  const scores = ordinalScores(table)
  const shares = table.map((row) => row.map((c) => c / 1000))
  const handles: Handle[] = [{ kind: 'x', at: distance.value, label: 'd', onDrag: distance.set }]

  return (
    <Interactive
      title="How ordinal metrics respond to the size of an error"
      caption={
        <MathText text="Five ordered classes. A fraction of each class is predicted $d$ classes away (upwards if the scale allows, otherwise downwards). The chart holds that fraction fixed and varies $d$: accuracy does not move, MAE grows about linearly, and quadratic weighted kappa and Kendall's $\tau_b$ fall. The heatmap is the expected confusion matrix as shares of all items. The majority-class switch replaces the predictions with the most frequent class, which keeps accuracy respectable on skewed data and sets both kappas to zero. Drag $d$ along the chart." />
      }
      controls={
        <>
          <ParamChoice label="class shares" value={prior} onChange={setPrior} options={PRIOR_OPTIONS} />
          <ParamSlider label="error rate" value={rate} onChange={setRate} min={0} max={1} step={0.01} />
          <ParamSlider label="error distance d" param={distance} withArrows />
          <ParamSwitch label="predict the majority class" checked={majority} onChange={setMajority} />
        </>
      }
      readout={
        <>
          <Readout label="accuracy" value={show(scores.accuracy)} />
          <Readout label="MAE" value={show(scores.mae)} />
          <Readout label="macro-MAE" value={show(scores.macroMae)} />
          <Readout label="linear κ_w" value={show(scores.linearKappa)} />
          <Readout label="QWK" value={show(scores.quadraticKappa)} />
          <Readout label="Kendall τ_b" value={show(scores.kendallTauB)} />
          <Readout label="Spearman ρ" value={show(scores.spearman)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <XYChart
          series={series}
          handles={handles}
          xRange={[1, 4]}
          xLabel="error distance d"
          yLabel="metric"
          height={300}
          ariaLabel="Metrics against the distance of the errors"
        />
        <Heatmap
          x={CLASSES}
          y={CLASSES}
          z={shares}
          range={[0, 0.5]}
          xLabel="predicted class"
          yLabel="true class"
          valueLabel="share of items"
          height={300}
          ariaLabel="Expected confusion matrix"
        />
      </div>
    </Interactive>
  )
}
