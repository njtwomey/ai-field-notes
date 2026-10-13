import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  formatPower,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { type Params } from 'aifn-compute/foundation/pytree'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { heNormal } from 'aifn-compute/nn/init'
import { Mlp } from 'aifn-compute/nn/layers'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule, sgdRule } from 'aifn-compute/optim/first-order'
import { moons } from 'aifn-methods/data/synthetic'

/** Learning rates on a half-decade grid, as log10 values. */
const LOG_RATES = [-3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5]
const RATES = LOG_RATES.map((e) => 10 ** e)
/** Losses above this are off the chart (chance level is ln 2 = 0.69). */
const CEILING = 1

const column = (y: Tensor) => fromData(Float64Array.from(toFlat(y)), [y.shape[0], 1])
const TRAIN = moons(stream('lr-sweep/train'), { n: 200, noise: 0.25 })
const VALID = moons(stream('lr-sweep/validation'), { n: 1000, noise: 0.25 })
const DATA = { x: TRAIN.x, y: column(TRAIN.y!) }
const VALID_Y = column(VALID.y!)
const NET = Mlp([2, 16, 16, 1], { activation: 'relu', init: heNormal() })

/** The validation loss of every (learning rate, seed) run, NaN for a run that diverged. */
function sweep(optimiser: 'Adam' | 'SGD with momentum', batch: number, steps: number, seeds: number) {
  return RATES.map((rate) =>
    Array.from({ length: seeds }, (_, k) => {
      const rule = optimiser === 'Adam' ? adamRule({ stepSize: rate }) : sgdRule({ stepSize: rate, momentum: 0.9 })
      const algorithm = trainingLoop({
        loss: (w: Params[], b: { x: Tensor; y: Tensor }) => binaryCrossEntropyWithLogits(NET.apply(w, b.x), b.y),
        data: DATA,
        batchSize: batch,
        optimizer: rule,
      })
      const final = run(algorithm, { params: NET.init(stream(`lr-sweep/init/${k + 1}`)) }, steps, {
        stream: stream(`lr-sweep/order/${k + 1}`),
      })
      if (final.diverged) return NaN
      return Number(unwrap(binaryCrossEntropyWithLogits(NET.apply(final.params, VALID.x), VALID_Y)))
    }),
  )
}

/** A learning-rate sweep of a small network: validation loss against learning rate, one point per seed. */
export function LearningRateSweep() {
  const state = useFigureState({
    optimiser: choice(['Adam', 'SGD with momentum'] as const, 'Adam', { label: 'optimiser' }),
    batch: int(32, { ge: 4, le: 128, suggestions: [8, 32, 128], label: 'batch size' }),
    steps: int(200, { ge: 50, le: 400, suggestions: [100, 200, 400], label: 'training steps' }),
    seeds: int(3, { ge: 1, le: 5, suggestions: [1, 3, 5], label: 'seeds per learning rate' }),
  })
  const { optimiser, batch, steps, seeds } = state
  const computed = useComputed(() => sweep(optimiser, batch, steps, seeds), [optimiser, batch, steps, seeds], {
    mode: 'release',
  })
  const losses = computed.value

  const summary = useMemo(() => {
    const means = losses.map((row) => row.reduce((a, b) => a + b, 0) / row.length)
    const finite = means.map((m, i) => (Number.isFinite(m) ? i : -1)).filter((i) => i >= 0)
    const best = finite.reduce((a, i) => (means[i] < means[a] ? i : a), finite[0])
    const row = losses[best]
    const sd = row.length > 1 ? Math.sqrt(row.reduce((a, v) => a + (v - means[best]) ** 2, 0) / (row.length - 1)) : NaN
    const points = losses.flatMap((r, i) => r.map((v) => [RATES[i], v] as const))
    const shown = points.filter(([, v]) => Number.isFinite(v) && v <= CEILING)
    return { means, best, sd, shown, off: points.length - shown.length }
  }, [losses])

  const curve = useMemo(() => {
    const keep = summary.means.map((m, i) => (Number.isFinite(m) && m <= CEILING ? i : -1)).filter((i) => i >= 0)
    return { x: keep.map((i) => RATES[i]), y: keep.map((i) => summary.means[i]) }
  }, [summary])

  const x = useAxis({ label: 'learning rate', log: true, range: [1e-4, 10], format: formatPower })
  const y = useAxis({ label: 'validation cross-entropy', range: [0, CEILING] })
  const bestRate = RATES[summary.best]
  return (
    <Figure
      title="A learning-rate sweep"
      state={state}
      caption={`A network with two hidden layers of 16 ReLU units, trained on 200 points of the two-moons data and scored by its cross-entropy on 1000 fresh points from the same distribution. Each learning rate on a half-decade grid is trained once per seed (the seed sets the initial weights and the order of the minibatches); the points are single runs and the curve their mean. The learning rate is constant, the number of training steps is fixed, and nothing else is tuned. Runs whose loss is above ${CEILING}, or that diverge, are left off the chart and counted in the readouts; a constant prediction of one half scores ln 2 = 0.69. Change the batch size or the optimiser and watch the best learning rate move.`}
      readouts={{
        sweep: (
          <>
            <Readout label="best learning rate" value={formatNumber(bestRate)} />
            <Readout label="mean loss there" value={formatNumber(summary.means[summary.best])} />
            <Readout
              label="spread over seeds (sd)"
              value={Number.isFinite(summary.sd) ? formatNumber(summary.sd) : '—'}
            />
            <Readout label="runs off the chart" value={String(summary.off)} />
          </>
        ),
      }}
    >
      <Plot x={x} y={y}>
        {summary.shown.length > 0 && (
          <Points
            name="one run"
            x={summary.shown.map((p) => p[0])}
            y={summary.shown.map((p) => p[1])}
            slot={0}
            size={5}
            stale={computed.stale}
          />
        )}
        {curve.x.length > 0 && (
          <Curve name="mean over seeds" x={curve.x} y={curve.y} slot={0} showPoints stale={computed.stale} />
        )}
        <Annotation x={bestRate} text="best mean" dashed emphasis />
        <Annotation y={Math.log(2)} text="constant prediction" dashed muted />
      </Plot>
    </Figure>
  )
}
