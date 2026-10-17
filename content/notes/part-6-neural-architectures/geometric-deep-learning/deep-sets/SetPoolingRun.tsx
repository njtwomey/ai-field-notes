import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import type { SetTarget } from 'aifn-methods/data/synthetic'
import { FixedHeight, useSlicedRun } from '../_shared/figure-helpers'
import {
  deepSetsRun,
  POOLINGS,
  TEST,
  TEST_MAX,
  TEST_PER_SIZE,
  TRAIN_MAX,
  TRAIN_MIN,
  TRAIN_SETS,
  type Checkpoint,
  type Setup,
} from './deep-sets-run'

const PANEL = 280
const SIZES = Array.from({ length: TEST_MAX }, (_, i) => i + 1)
const TARGETS: { value: SetTarget; label: string }[] = [
  { value: 'sum', label: 'sum of the set' },
  { value: 'max', label: 'maximum of the set' },
  { value: 'variance', label: 'variance of the set' },
]

/** One φ–ρ network trained with sum, mean and max pooling on sets of 1 to 10 numbers, tested on sets of 1 to 30. */
export function SetPoolingRun() {
  const state = useFigureState({
    target: choice(TARGETS, 'sum', { label: 'target' }),
    steps: int(300, { ge: 50, le: 3000, suggestions: [150, 300, 1000], label: 'Adam steps' }),
    rate: float(0.01, { gt: 0, le: 0.2, suggestions: [0.003, 0.01, 0.03], label: 'learning rate' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    size: int(20, { ge: 1, le: TEST_MAX, label: 'test set size n' }),
  })
  const { target, steps, rate, seed, size } = state
  const setup = useMemo<Setup>(() => ({ target, steps, rate, seed }), [target, steps, rate, seed])
  const run = useSlicedRun(setup, deepSetsRun)
  const last: Checkpoint | undefined = run.checkpoints.at(-1)
  const truth = TEST[size - 1].targets[target]

  const nx = useAxis({ label: 'set size n', range: [0.5, TEST_MAX + 0.5], nice: false, integer: true })
  const ey = useAxis({ label: 'test RMSE', log: true, hold: 'union', key: target })
  const tx = useAxis({ label: 'true value', hold: 'union', key: target })
  const ty = useAxis({ label: 'prediction', hold: 'union', key: target, equal: tx })
  const lo = Math.min(...truth)
  const hi = Math.max(...truth)

  return (
    <Figure
      title="Which pooling extrapolates to larger sets"
      state={state}
      defaultSize="L"
      caption={`One network ρ(pool φ(x)) is trained three times from the same start, once with each pooling, on ${TRAIN_SETS} sets of ${TRAIN_MIN} to ${TRAIN_MAX} numbers drawn uniformly on [0, 1]. φ maps each number to 16 features through one hidden layer of 16 ReLU units; ρ maps the pooled 16 features to the prediction through one hidden layer of 16. Targets are standardised for training; errors are in the target's units. Left: the error on fresh sets of each size from 1 to ${TEST_MAX}; the dashed line marks the largest training size. Right: predictions against true values on ${TEST_PER_SIZE} test sets of the chosen size. Drag the vertical handle on the left plot to choose the test size. The sum is learned by sum pooling and the maximum by max pooling, and both keep working on sets three times larger than any seen in training. The variance is an average, so mean pooling learns it at every size; sum pooling has to divide by n inside ρ and fails beyond the training sizes. The curves fill in as training runs.`}
      readouts={{
        [`test RMSE at n = ${size}`]: (
          <>
            {POOLINGS.map((q) => (
              <Readout key={q} label={`${q} pooling`} value={last ? formatNumber(last.rmse[q][size - 1]) : '…'} />
            ))}
          </>
        ),
        training: (
          <>
            {POOLINGS.map((q) => (
              <Readout key={q} label={`${q}: loss (standardised)`} value={last ? formatNumber(last.loss[q]) : '…'} />
            ))}
            <Readout label="step" value={last ? `${last.step}${run.done ? '' : ', training…'}` : '…'} />
          </>
        ),
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={nx} y={ey} title="test error by set size">
            {POOLINGS.map((q, k) => (
              <Curve
                key={q}
                name={`${q} pooling`}
                slot={k}
                x={SIZES}
                y={last ? last.rmse[q].map((v) => Math.max(v, 1e-5)) : []}
                showPoints
              />
            ))}
            <Annotation x={TRAIN_MAX} text="largest training set" dashed />
            <Handle {...state.handle('size', { label: `n = ${size}` })} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={tx} y={ty} title={`predictions on sets of ${size}`}>
            <Curve name="prediction = truth" x={[lo, hi]} y={[lo, hi]} dashed muted />
            {last &&
              POOLINGS.map((q, k) => (
                <Points key={q} name={`${q} pooling`} slot={k} x={truth} y={last.predictions[q][size - 1]} size={6} />
              ))}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
