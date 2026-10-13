import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, mean } from 'aifn-compute/foundation/tensor'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold } from 'aifn-compute/learning/validate'
import { standardDeviation } from 'aifn-compute/probability/stats'
import { FoldGrid } from '../_shared/FoldGrid'
import { freshSample, polynomial, sineData, testError } from '../_shared/problems'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const avg = (v: number[]) => mean(fromData(Float64Array.from(v)))
/** The fold models whose true error is measured, at most this many per cross-validation (the first ones). */
const SCORED_FOLDS = 10

/**
 * K-fold cross-validation of a polynomial fit for several K, repeated over datasets: the estimate's mean and spread
 * beside the true error of the fold models it averages and of the model fitted to all n points.
 */
export function KFoldBias() {
  const state = useFigureState({
    n: int(40, { ge: 12, le: 120, suggestions: [20, 40, 80], label: 'data points n' }),
    degree: int(5, { ge: 1, le: 9, suggestions: [1, 3, 5, 7], label: 'polynomial degree' }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ' }),
    replicates: int(60, { ge: 10, le: 400, suggestions: [30, 60, 200], label: 'datasets drawn' }),
    k: int(5, { ge: 2, le: 120, label: 'folds K shown' }),
  })
  const { n, degree, noise, replicates } = state
  const ks = useMemo(() => [2, 3, 5, 10, 20, n].filter((k, i, a) => k <= n && a.indexOf(k) === i), [n])
  const run = useComputed(
    () => {
      const learner = polynomial(degree)
      const fresh = freshSample(stream('k-fold/fresh'), noise, 500)
      const full: number[] = []
      const est = ks.map(() => [] as number[])
      const folds = ks.map(() => [] as number[])
      for (let r = 0; r < replicates; r++) {
        const s = child(stream('k-fold'), r)
        const { data } = sineData(child(s, 'data'), n, noise)
        full.push(testError(learner.fit(data), fresh))
        ks.forEach((k, j) => {
          const cv = crossValidate(learner, data, kFold({ k, shuffle: true }), [meanSquaredError], {
            stream: child(s, 'cv', k),
          })
          est[j].push(cv.mean.meanSquaredError)
          folds[j].push(avg(cv.folds.slice(0, SCORED_FOLDS).map((f) => testError(f.model, fresh))))
        })
      }
      return {
        full: avg(full),
        mean: est.map(avg),
        sd: est.map((e) => standardDeviation(e, { sample: true })),
        folds: folds.map(avg),
      }
    },
    [n, degree, noise, replicates, ks],
    { mode: 'release' },
  )
  const v = run.value
  const nearest = ks.reduce(
    (b, k) => (Math.abs(Math.log(k / state.k)) < Math.abs(Math.log(b / state.k)) ? k : b),
    ks[0],
  )
  const j = ks.indexOf(nearest)
  const splits = useMemo(() => kFold({ k: nearest, shuffle: true }).split({ n }, stream('k-fold/grid')), [nearest, n])
  const band = useMemo(() => ({ lo: v.mean.map((m, i) => m - v.sd[i]), hi: v.mean.map((m, i) => m + v.sd[i]) }), [v])
  const kx = useAxis({ label: 'folds K', categories: ks.map((k) => (k === n ? `${k} (LOO)` : String(k))) })
  const at = ks.map((_, i) => i)
  const ey = useAxis({
    label: 'mean squared error',
    range: [0, undefined],
    hold: 'union',
    key: `${n}/${degree}/${noise}`,
  })
  return (
    <Figure
      title="The pessimistic bias of K-fold cross-validation"
      state={state}
      hoverReadout={false}
      caption={`Top: the folds of one shuffled ${nearest}-fold split of n = ${n} rows; each row of the grid is one fit, training on the blue cells and predicting the orange ones. Bottom: a degree-${degree} polynomial with a small ridge penalty, fitted to n points of y = sin(πx) + σε and cross-validated with K folds, repeated on ${replicates} datasets. The mean estimate (band: ±1 standard deviation over datasets) lies on the mean true error of the fold models it averages, each fitted to n(K − 1)/K points and measured on 500 fresh points. Both lie above the true error of the model fitted to all n points (dashed) when K is small, because each fold model sees less data; the gap closes as K grows towards n, leave-one-out (LOO). The spread over datasets is larger than the bias. Drag the line labelled K, or raise the degree to steepen the learning curve and widen the gap.`}
      readouts={
        <>
          <Readout
            label="training points per fold"
            value={`${n - Math.ceil(n / nearest)}–${n - Math.floor(n / nearest)}`}
          />
          <Readout label={`${nearest}-fold estimate (mean ± sd)`} value={`${fmt(v.mean[j])} ± ${fmt(v.sd[j])}`} />
          <Readout label="true error, fold models" value={fmt(v.folds[j])} />
          <Readout label="true error, model on all n" value={fmt(v.full)} />
        </>
      }
    >
      <FoldGrid
        splits={splits}
        n={n}
        title={`${nearest}-fold assignment`}
        rowLabel="fit"
        rowHeight={Math.max(4, Math.min(22, 300 / nearest))}
      />
      <Plot x={kx} y={ey} title="estimate and truth">
        <Area name="±1 sd over datasets" x={at} y={band.hi} base={band.lo} slot={0} stale={run.stale} />
        <Curve name="K-fold estimate (mean)" x={at} y={v.mean} slot={0} showPoints stale={run.stale} />
        <Curve name="true error of the fold models" x={at} y={v.folds} slot={1} showPoints stale={run.stale} />
        <Curve
          name="true error of the model on all n"
          x={[-0.5, ks.length - 0.5]}
          y={[v.full, v.full]}
          emphasis
          dashed
        />
        <Handle
          kind="x"
          at={j}
          label="K"
          onDrag={(x) => state.set('k', ks[Math.max(0, Math.min(ks.length - 1, Math.round(x)))])}
        />
      </Plot>
    </Figure>
  )
}
