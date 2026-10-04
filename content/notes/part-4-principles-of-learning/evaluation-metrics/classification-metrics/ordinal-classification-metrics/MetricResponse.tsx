import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  MathText,
  Plot,
  Raster,
  Readout,
  setting,
  useAxis,
  useFigureState,
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
  const state = useFigureState({
    prior: choice<Prior>(PRIOR_OPTIONS, 'uniform', { label: 'class shares' }),
    rate: float(0.3, { min: 0, max: 1, step: 0.01, label: 'error rate' }),
    distance: int(1, { min: 1, max: 4, step: 1, label: 'error distance d' }),
    majority: setting(false, 'predict the majority class'),
  })

  const series = useMemo(() => {
    const rows = DISTANCES.map((d) => ordinalScores(shiftedTable(PRIORS[state.prior], state.rate, d)))
    return [
      { name: 'accuracy', x: DISTANCES, y: rows.map((r) => r.accuracy), slot: 0 },
      { name: 'quadratic weighted kappa', x: DISTANCES, y: rows.map((r) => r.quadraticKappa), slot: 1 },
      { name: "Kendall's τ_b", x: DISTANCES, y: rows.map((r) => r.kendallTauB), slot: 2 },
      { name: 'MAE', x: DISTANCES, y: rows.map((r) => r.mae), slot: 3, dashed: true },
    ] as const
  }, [state.prior, state.rate])

  const table = state.majority
    ? majorityTable(PRIORS[state.prior])
    : shiftedTable(PRIORS[state.prior], state.rate, state.distance)
  const scores = ordinalScores(table)
  const shares = table.map((row) => row.map((c) => c / 1000))

  const xAxis = useAxis({ label: 'error distance d', range: [1, 4] })
  const yAxis = useAxis({ label: 'metric', hold: 'union' })
  const xAxis2 = useAxis({ label: 'predicted class' })
  const yAxis2 = useAxis({ label: 'true class' })
  return (
    <Figure
      title="How ordinal metrics respond to the size of an error"
      state={state}
      caption={
        <MathText text="Five ordered classes. A fraction of each class is predicted $d$ classes away (upwards if the scale allows, otherwise downwards). The chart holds that fraction fixed and varies $d$: accuracy does not move, MAE grows about linearly, and quadratic weighted kappa and Kendall's $\tau_b$ fall. The heatmap is the expected confusion matrix as shares of all items. The majority-class switch replaces the predictions with the most frequent class, which keeps accuracy respectable on skewed data and sets both kappas to zero. Drag $d$ along the chart." />
      }

      readouts={
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
        <Plot x={xAxis} y={yAxis} height={300} ariaLabel={'Metrics against the distance of the errors'}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Curve {...series[2]} />
          <Curve {...series[3]} />
          <Handle {...state.handle('distance', { label: 'd' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300} ariaLabel={'Expected confusion matrix'}>
          <Raster x={CLASSES} y={CLASSES} z={shares} range={[0, 0.5]} valueLabel={'share of items'} />
        </Plot>
      </div>
    </Figure>
  )
}
