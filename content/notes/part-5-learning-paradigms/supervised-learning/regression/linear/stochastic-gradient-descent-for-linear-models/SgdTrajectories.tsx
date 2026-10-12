import { useMemo, useState } from 'react'
import {
  Choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { SGD_ALPHA, SGD_L1_RATIO, sgdData } from '../_shared/diabetes'
import { HIGHLIGHT, featureStyle } from '../_shared/features'

/**
 * SGD weights per epoch on the diabetes data against the exact solution of the same objective, for several penalties
 * and schedules, from the engine's `linearSgdSteps` and `elasticNet`.
 */
export function SgdTrajectories() {
  const data = useComputed(sgdData, [], { mode: 'release' }).value
  // The runs and their labels come with the computed data, so the run is chosen by a plain control.
  const [runIndex, setRunIndex] = useState('1')
  const state = useFigureState({ epoch: slider(0, 60, 20, { step: 1, label: 'epoch' }) })
  const xAxis = useAxis({ label: 'epoch', hold: 'union' })
  const yAxis = useAxis({ label: 'coefficient', hold: 'union' })
  const xAxis2 = useAxis({ label: 'epoch', hold: 'union' })
  const yAxis2 = useAxis({ label: 'objective − optimum', hold: 'union', log: true })
  const run = data.runs[Number(runIndex)]
  const epochs = useMemo(() => run.coef[0].map((_, e) => e), [run])

  const paths = useMemo((): SeriesSpec[] => {
    const last = epochs.length - 1
    const weights: SeriesSpec[] = data.features.map((name, j) => ({
      type: 'line',
      x: epochs,
      y: run.coef[j],
      ...featureStyle(name),
    }))
    // The exact solution of the same objective, as a dashed level for each highlighted feature.
    const exact: SeriesSpec[] = data.features
      .filter((name) => name in HIGHLIGHT)
      .map((name) => ({
        name: 'exact solution',
        type: 'line',
        x: [0, last],
        y: [run.exact[data.features.indexOf(name)], run.exact[data.features.indexOf(name)]],
        slot: HIGHLIGHT[name],
        dashed: true,
      }))
    return [...weights, ...exact]
  }, [data, run, epochs])

  const gaps = useMemo(
    (): SeriesSpec[] =>
      data.runs.map((r, k) => ({ name: r.label, type: 'line', x: r.gap.map((_, e) => e), y: r.gap, slot: k })),
    [data],
  )

  const e = Math.min(state.epoch, epochs.length - 1)
  const deviation = Math.max(...run.coef.map((row, j) => Math.abs(row[e] - run.exact[j])))
  return (
    <Figure
      title="SGD against the exact solution"
      state={state}
      caption={`SGD on the standardised diabetes data with α = ${SGD_ALPHA} (elastic net: l1 ratio ${SGD_L1_RATIO}), one epoch per point, against the exact minimiser of the same objective (dashed). Pick a run and drag the epoch line. With the decaying invscaling schedule the weights settle near the exact values but never reach them; with a constant step they keep jittering. Watch the count of exact zeros: the truncated (cumulative-penalty) L1 update lands on the lasso's zero set only after about 59 epochs, and plain subgradient descent never produces an exact zero. The right-hand chart compares the objective gap of every run on a log scale.`}
      controls={
        <Choice
          label="run"
          value={runIndex}
          onChange={setRunIndex}
          options={data.runs.map((r, k) => ({ value: String(k), label: r.label }))}
        />
      }
      readouts={
        <>
          <Readout label="objective − optimum" value={formatNumber(run.gap[e])} />
          <Readout label="largest |w − w*|" value={formatNumber(deviation)} />
          <Readout label="exact zeros" value={`${run.zeros[e]} (exact solution: ${run.exactZeros})`} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          {seriesLayers(paths)}
          <Handle kind="x" at={e} label="epoch" onDrag={(x) => state.set('epoch', Math.round(x))} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(gaps)}
        </Plot>
      </div>
    </Figure>
  )
}
