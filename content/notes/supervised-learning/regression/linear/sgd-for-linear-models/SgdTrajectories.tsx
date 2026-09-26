import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import type { SgdTrajectories as Trajectories } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { HIGHLIGHT, featureStyle } from '../_shared/features'

/**
 * SGDRegressor weights per epoch on the diabetes data against the exact solution of the same objective, for several
 * penalties and schedules. Precomputed by python/mlc/figures/linear_models.py.
 */
export function SgdTrajectories() {
  const { data } = useFigure<Trajectories>('sgd-for-linear-models/trajectories')
  const [choice, setChoice] = useState('1')
  const epoch = useParam(20, { min: 0, max: 60, step: 1 })

  const run = data?.runs[Number(choice)]
  const epochs = useMemo(() => (run ? run.coef[0].map((_, e) => e) : []), [run])

  const paths = useMemo((): XYSeries[] => {
    if (!data || !run) return []
    const last = epochs.length - 1
    const weights: XYSeries[] = data.features.map((name, j) => ({
      type: 'line',
      x: epochs,
      y: run.coef[j],
      ...featureStyle(name),
    }))
    // The exact solution of the same objective, as a dashed level for each highlighted feature.
    const exact: XYSeries[] = data.features
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
    (): XYSeries[] =>
      data
        ? data.runs.map((r, k) => ({ name: r.label, type: 'line', x: r.gap.map((_, e) => e), y: r.gap, slot: k }))
        : [],
    [data],
  )

  if (!data || !run) return null
  const e = Math.min(epoch.value, epochs.length - 1)
  const deviation = Math.max(...run.coef.map((row, j) => Math.abs(row[e] - run.exact[j])))
  const handles: Handle[] = [{ kind: 'x', at: e, label: 'epoch', onDrag: (x) => epoch.set(Math.round(x)) }]

  return (
    <Interactive
      title="SGD against the exact solution"
      caption={`SGDRegressor on the standardised diabetes data with α = ${data.alpha} (elastic net: l1 ratio ${data.l1_ratio}), one epoch per point, against the exact minimiser of the same objective (dashed). Pick a run and drag the epoch line. With the decaying invscaling schedule the weights settle near the exact values but never reach them; with a constant step they keep jittering. Watch the count of exact zeros: scikit-learn's truncated L1 update lands on the lasso's zero set only after about 56 epochs, and plain subgradient descent never produces an exact zero. The right-hand chart compares the objective gap of every run on a log scale.`}
      controls={
        <>
          <ParamChoice
            label="run"
            value={choice}
            onChange={setChoice}
            options={data.runs.map((r, k) => ({ value: String(k), label: r.label }))}
          />
          <ParamSlider label="epoch" param={epoch} />
        </>
      }
      readout={
        <>
          <Readout label="objective − optimum" value={formatNumber(run.gap[e])} />
          <Readout label="largest |w − w*|" value={formatNumber(deviation)} />
          <Readout label="exact zeros" value={`${run.zeros[e]} (exact solution: ${run.exact_zeros})`} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <XYChart height={340} series={paths} xLabel="epoch" yLabel="coefficient" handles={handles} />
        <XYChart height={340} series={gaps} xLabel="epoch" yLabel="objective − optimum" yLog />
      </div>
    </Interactive>
  )
}
