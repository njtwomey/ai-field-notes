import { drawTrainingSet } from 'aifn-methods/theory'
import { fitAndPredict, type ResampledModel } from 'aifn-methods/theory/bias-variance'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { child, stream } from 'aifn-compute/foundation/random'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const CANDIDATES: ResampledModel[] = Array.from({ length: 9 }, (_, degree) => ({ kind: 'polynomial', degree }))
const TRAIN_SIZE = 40
const TRUTH_SIZE = 2000
const NOISE = 0.3
const LIMITS: [number, number] = [0.02, 0.3]

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const mean = (v: readonly number[]) => v.reduce((a, b) => a + b, 0) / v.length
const clip = (v: number) => Math.max(LIMITS[0], Math.min(LIMITS[1], v))

/**
 * Many replicate problems, each with its own training, validation and test sets. The nine polynomial candidates are
 * fitted on the training set, the one with the lowest validation error is chosen, and its validation error and test
 * error are set against its true error on 2000 fresh points.
 */
export function SelectionAndAssessment() {
  const state = useFigureState({
    nVal: int(10, { min: 5, max: 1000, scale: 'log10', suggestions: [10, 20, 50, 200], label: 'validation points' }),
    nTest: int(100, { min: 5, max: 1000, scale: 'log10', suggestions: [20, 100, 500], label: 'test points' }),
    replicates: slider(20, 300, 150, { step: 10, label: 'replicate problems' }),
  })
  const runs = useComputed(
    () =>
      Array.from({ length: state.replicates }, (_, r) => {
        const s = child(stream('selection-and-assessment'), 'replicate', r)
        const train = drawTrainingSet(child(s, 'train'), 'sine', TRAIN_SIZE, NOISE)
        const val = drawTrainingSet(child(s, 'validation'), 'sine', state.nVal, NOISE)
        const test = drawTrainingSet(child(s, 'test'), 'sine', state.nTest, NOISE)
        const fresh = drawTrainingSet(child(s, 'truth'), 'sine', TRUTH_SIZE, NOISE)
        const scored = CANDIDATES.map((model) => ({
          model,
          validation: meanSquaredError(val.y, fitAndPredict(model, train.x, train.y, val.x)),
          truth: meanSquaredError(fresh.y, fitAndPredict(model, train.x, train.y, fresh.x)),
        }))
        const winner = scored.reduce((a, b) => (b.validation < a.validation ? b : a))
        return {
          validation: winner.validation,
          test: meanSquaredError(test.y, fitAndPredict(winner.model, train.x, train.y, test.x)),
          truth: winner.truth,
          oracle: Math.min(...scored.map((c) => c.truth)),
        }
      }),
    [state.nVal, state.nTest, state.replicates],
    { mode: 'release' },
  )
  const rows = runs.value
  const truths = rows.map((r) => clip(r.truth))

  const x = useAxis({ label: 'true error of the chosen model', range: LIMITS })
  const y = useAxis({ label: 'estimate of that error', range: LIMITS, equal: x })
  return (
    <Figure
      title="Selection on validation data, assessment on test data"
      state={state}
      caption={`Each point is one replicate problem: ${TRAIN_SIZE} training points from y = sin(πx) + ${NOISE}ε, polynomials of degree 0 to 8 fitted to them, and the degree with the lowest validation error chosen. Its validation error and its error on a separate test set are plotted against its true error, measured on ${TRUTH_SIZE} fresh points. Points on the dashed diagonal are exact. The validation points sit mostly below it, because the winner was chosen partly for luck on the validation set; the test points scatter evenly about it. Enlarge the validation set to shrink the optimism, and the test set to shrink the test scatter. Values are clipped to [${LIMITS[0]}, ${LIMITS[1]}].`}
      readouts={
        <>
          <Readout label="mean validation error of the winner" value={fmt(mean(rows.map((r) => r.validation)))} />
          <Readout label="mean test error of the winner" value={fmt(mean(rows.map((r) => r.test)))} />
          <Readout label="mean true error of the winner" value={fmt(mean(rows.map((r) => r.truth)))} />
          <Readout label="mean true error of the best candidate" value={fmt(mean(rows.map((r) => r.oracle)))} />
        </>
      }
    >
      <Plot x={x} y={y}>
        <Curve name="estimate = truth" x={LIMITS} y={LIMITS} muted dashed />
        <Points
          name="validation error of the winner"
          x={truths}
          y={rows.map((r) => clip(r.validation))}
          slot={1}
          stale={runs.stale}
        />
        <Points
          name="test error of the winner"
          x={truths}
          y={rows.map((r) => clip(r.test))}
          slot={0}
          stale={runs.stale}
        />
      </Plot>
    </Figure>
  )
}
