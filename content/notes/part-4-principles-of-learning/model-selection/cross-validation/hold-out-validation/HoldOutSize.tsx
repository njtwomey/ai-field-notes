import { useMemo } from 'react'
import {
  Annotation,
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
import { meanSquaredError, rootMeanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, shuffleSplit } from 'aifn-compute/learning/validate'
import { standardDeviation } from 'aifn-compute/probability/stats'
import { freshSample, polynomial, sineData, testError } from '../_shared/problems'

const FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const avg = (v: number[]) => mean(fromData(Float64Array.from(v)))

/**
 * Hold-out validation of a polynomial fit, repeated over many datasets: the estimate's mean and spread against the
 * validation fraction, beside the true error of the model it scores and of the model refitted on all the data.
 */
export function HoldOutSize() {
  const state = useFigureState({
    n: int(60, { ge: 20, le: 400, suggestions: [30, 60, 120, 240], label: 'data points n' }),
    degree: int(5, { ge: 1, le: 9, suggestions: [1, 3, 5, 7], label: 'polynomial degree' }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ' }),
    replicates: int(200, { ge: 20, le: 1000, suggestions: [100, 200, 500], label: 'datasets drawn' }),
    fraction: float(0.3, { min: 0.1, max: 0.9, step: 0.1, label: 'validation fraction' }),
  })
  const { n, degree, noise, replicates } = state
  const run = useComputed(
    () => {
      const learner = polynomial(degree)
      const fresh = freshSample(stream('hold-out/fresh'), noise, 1000)
      const full: number[] = []
      const est = FRACTIONS.map(() => [] as number[])
      const part = FRACTIONS.map(() => [] as number[])
      for (let r = 0; r < replicates; r++) {
        const s = child(stream('hold-out'), r)
        const { data } = sineData(child(s, 'data'), n, noise)
        full.push(testError(learner.fit(data), fresh))
        FRACTIONS.forEach((f, j) => {
          const cv = crossValidate(learner, data, shuffleSplit({ splits: 1, testSize: f }), [meanSquaredError], {
            stream: child(s, 'split', j),
          })
          est[j].push(cv.mean.meanSquaredError)
          part[j].push(testError(cv.folds[0].model, fresh))
        })
      }
      const fullMean = avg(full)
      return {
        fullMean,
        mean: est.map(avg),
        sd: est.map((e) => standardDeviation(e, { sample: true })),
        part: part.map(avg),
        rmse: est.map((e) => rootMeanSquaredError(full, e)),
      }
    },
    [n, degree, noise, replicates],
    { mode: 'release' },
  )
  const v = run.value
  const j = Math.min(FRACTIONS.length - 1, Math.max(0, Math.round(state.fraction * 10) - 1))
  const band = useMemo(() => ({ lo: v.mean.map((m, i) => m - v.sd[i]), hi: v.mean.map((m, i) => m + v.sd[i]) }), [v])
  const nVal = Math.ceil(FRACTIONS[j] * n)
  const fx = useAxis({ label: 'fraction of the data held out', range: [0.05, 0.95], nice: false })
  const ey = useAxis({
    label: 'mean squared error',
    range: [0, undefined],
    hold: 'union',
    key: `${n}/${degree}/${noise}`,
  })
  const gx = useAxis({ label: 'fraction of the data held out', range: [0.05, 0.95], nice: false })
  const gy = useAxis({ label: 'RMS error about the final model’s error', range: [0, undefined] })
  return (
    <Figure
      title="How much data to hold out"
      state={state}
      caption={`A polynomial with a small ridge penalty (0.1, which keeps fits to a few points stable) is fitted to n points of y = sin(πx) + σε, x uniform on [−1, 1], and scored on a held-out fraction of them. The experiment is repeated on ${replicates} independent datasets. Left: the mean hold-out estimate (with a band of ±1 standard deviation over datasets), the true error of the model it scores, fitted on the remaining points, and the true error of the model refitted on all n points (dashed), each true error measured on 1,000 fresh points. The estimate's mean lies on the error of the model it scores: it is unbiased for that model. That model is fitted on fewer points than the final one, so the estimate is pessimistic, more so as the held-out fraction grows; its spread grows as the fraction shrinks. Right: the root-mean-square difference between the estimate and the final model's true error, which combines the two. Drag the vertical line to read the numbers at one fraction; raise the degree to steepen the learning curve.`}
      readouts={
        <>
          <Readout label="training / validation points" value={`${n - nVal} / ${nVal}`} />
          <Readout label="hold-out estimate (mean ± sd)" value={`${fmt(v.mean[j])} ± ${fmt(v.sd[j])}`} />
          <Readout label="true error, model scored" value={fmt(v.part[j])} />
          <Readout label="true error, model on all n" value={fmt(v.fullMean)} />
          <Readout label="RMS error about the final model’s error" value={fmt(v.rmse[j])} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={fx} y={ey} title="estimate and truth">
          <Area name="±1 sd over datasets" x={FRACTIONS} y={band.hi} base={band.lo} slot={0} stale={run.stale} />
          <Curve name="hold-out estimate (mean)" x={FRACTIONS} y={v.mean} slot={0} showPoints stale={run.stale} />
          <Curve name="true error of the model scored" x={FRACTIONS} y={v.part} slot={1} showPoints stale={run.stale} />
          <Curve
            name="true error of the model on all n"
            x={[0.05, 0.95]}
            y={[v.fullMean, v.fullMean]}
            emphasis
            dashed
          />
          <Handle
            kind="x"
            at={FRACTIONS[j]}
            label="held out"
            onDrag={(x) => state.set('fraction', Math.min(0.9, Math.max(0.1, Math.round(x * 10) / 10)))}
          />
        </Plot>
        <Plot x={gx} y={gy} title="total error of the estimate">
          <Curve name="RMS error" x={FRACTIONS} y={v.rmse} slot={2} showPoints stale={run.stale} />
          <Annotation x={FRACTIONS[j]} dashed />
        </Plot>
      </div>
    </Figure>
  )
}
