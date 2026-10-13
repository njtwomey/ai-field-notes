import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Figure,
  float,
  formatNumber,
  Histogram,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { errorRate } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold, stratifiedKFold } from 'aifn-compute/learning/validate'
import { standardDeviation } from 'aifn-compute/probability/stats'
import { gaussians } from 'aifn-methods/data/synthetic'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { FoldGrid } from '../_shared/FoldGrid'
import { recallOfPositives, supervised } from '../_shared/problems'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const PARTITIONS = 100
const SEPARATION = 2.5
const METRICS = [
  { value: 'recall', label: 'recall of the rare class' },
  { value: 'error', label: 'error rate' },
] as const
const SHOWN = [
  { value: 'plain', label: 'shuffled k-fold' },
  { value: 'stratified', label: 'stratified k-fold' },
] as const

/**
 * Plain and stratified K-fold cross-validation of a logistic regression on data with a rare class: the folds of one
 * partition, and the estimates over many partitions beside the true value.
 */
export function StratifiedRareClass() {
  const state = useFigureState({
    prevalence: float(0.1, { min: 0.02, max: 0.5, step: 0.01, label: 'share of the rare class' }),
    n: int(100, { ge: 40, le: 400, suggestions: [60, 100, 200], label: 'data points n' }),
    k: int(5, { ge: 2, le: 10, suggestions: [3, 5, 10], label: 'folds K' }),
    metric: choice(METRICS, 'recall', { label: 'metric' }),
    shown: choice(SHOWN, 'plain', { label: 'folds drawn' }),
  })
  const { prevalence, n, k, metric, shown } = state
  const data = useMemo(
    () => supervised(gaussians(stream('stratified/data'), { n, prevalence, separation: SEPARATION })),
    [n, prevalence],
  )
  const labels = useMemo(() => Array.from(toFlat(data.y)), [data])
  const run = useComputed(
    () => {
      const model = logisticRegression({ l2: 0.1 })
      const fresh = supervised(gaussians(stream('stratified/fresh'), { n: 5000, prevalence, separation: SEPARATION }))
      const fitted = model.fit(data)
      const predicted = fitted.decide(fresh.x)
      const truth = { recall: recallOfPositives(fresh.y, predicted), error: errorRate(fresh.y, predicted) }
      const estimates = (stratify: boolean) => {
        const out = { recall: [] as number[], error: [] as number[] }
        for (let p = 0; p < PARTITIONS; p++) {
          const splitter = stratify ? stratifiedKFold({ k, shuffle: true }) : kFold({ k, shuffle: true })
          const cv = crossValidate(model, data, splitter, [errorRate, recallOfPositives], {
            stream: child(stream('stratified/partitions'), p),
          })
          out.recall.push(cv.mean.recall)
          out.error.push(cv.mean.errorRate)
        }
        return out
      }
      return { truth, plain: estimates(false), stratified: estimates(true) }
    },
    [data, prevalence, k],
    { mode: 'release' },
  )
  const v = run.value
  const plain = v.plain[metric].filter(Number.isFinite)
  const strat = v.stratified[metric].filter(Number.isFinite)
  const both = [...plain, ...strat]
  const binRange: [number, number] = both.length > 0 ? [Math.min(...both), Math.max(...both) + 1e-9] : [0, 1]
  const undefinedShare = 1 - plain.length / PARTITIONS
  const splits = useMemo(() => {
    const splitter = shown === 'stratified' ? stratifiedKFold({ k, shuffle: true }) : kFold({ k, shuffle: true })
    return splitter.split(data, stream('stratified/drawn'))
  }, [shown, k, data])
  const positives = splits.map((s) => Array.from(toFlat(s.test)).filter((i) => labels[i] === 1).length)
  const sx = useAxis({
    label: metric === 'recall' ? 'estimated recall' : 'estimated error rate',
    hold: 'union',
    key: `${metric}/${n}/${prevalence}/${k}`,
  })
  const sy = useAxis({ label: 'partitions (density)' })
  const sd = (a: number[]) => (a.length > 1 ? standardDeviation(a, { sample: true }) : NaN)
  return (
    <Figure
      title="Stratified folds and a rare class"
      state={state}
      hoverReadout={false}
      caption={`n points from two Gaussian classes, the rarer one (label 1) making up the chosen share, classified by logistic regression. Top: the folds of one partition, plain or stratified, with each row's label underneath; stratified folds deal the rare class out evenly. Bottom: the cross-validation estimate (the mean of the K per-fold values) over ${PARTITIONS} random partitions of these same data, plain and stratified, beside the true value for the model fitted to all n points (ink line, measured on 5,000 fresh points). With a rare class, plain folds often hold no rare points at all; recall is then undefined for that fold, and the whole estimate with it, so those partitions are missing from the plain histogram. Stratification also narrows the spread of the estimate.`}
      readouts={
        <>
          <Readout label="rare points per test fold (drawn partition)" value={positives.join(', ')} />
          <Readout label="plain partitions with an undefined estimate" value={`${Math.round(100 * undefinedShare)}%`} />
          <Readout label="spread, plain / stratified (sd)" value={`${fmt(sd(plain))} / ${fmt(sd(strat))}`} />
          <Readout label="true value" value={fmt(v.truth[metric])} />
        </>
      }
    >
      <FoldGrid
        splits={splits}
        n={n}
        title={shown === 'stratified' ? 'stratified folds' : 'shuffled folds'}
        rowLabel="fold"
        strip={{ values: labels, names: ['common class', 'rare class'], title: 'label' }}
      />
      <Plot x={sx} y={sy} title={`estimates over ${PARTITIONS} partitions of the same data`}>
        {plain.length > 0 && (
          <Histogram name="plain k-fold" values={plain} bins={20} range={binRange} slot={0} stale={run.stale} />
        )}
        {strat.length > 0 && (
          <Histogram name="stratified k-fold" values={strat} bins={20} range={binRange} slot={1} stale={run.stale} />
        )}
        <Annotation x={v.truth[metric]} text="true value" />
      </Plot>
    </Figure>
  )
}
