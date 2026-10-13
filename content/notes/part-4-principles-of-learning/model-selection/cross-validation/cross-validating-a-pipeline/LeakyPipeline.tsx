import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { pipeline } from 'aifn-compute/learning/compose'
import { dataset, type Decides } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold, type Fittable } from 'aifn-compute/learning/validate'
import { linearRegression } from 'aifn-methods/learning/linear'
import { oneHotEncoder, targetEncoder } from 'aifn-methods/learning/preprocessing'

const N = 100
const TEST = 20000
const STEPS = ['target encoding', 'one-hot encoding'] as const
type Step = (typeof STEPS)[number]

const encoder = (step: Step) =>
  step === 'target encoding' ? targetEncoder({ smooth: 0 }) : oneHotEncoder({ handleUnknown: 'ignore' })

/** A categorical input whose categories carry no information about y, encoded outside or inside the folds. */
export function LeakyPipeline() {
  const state = useFigureState({
    step: choice(STEPS, 'target encoding', { label: 'preprocessing step' }),
    perCategory: int(5, { ge: 2, le: 25, suggestions: [2, 5, 10, 20], label: 'rows per category' }),
    replicates: int(30, { ge: 5, le: 100, suggestions: [10, 30, 60], label: 'replicate data sets' }),
  })
  const { step, perCategory, replicates } = state
  const categories = Math.max(1, Math.floor(N / perCategory))
  const computed = useComputed(
    () => {
      const codes = (rows: number) =>
        fromData(
          Float64Array.from({ length: rows }, (_, i) => i % categories),
          [rows, 1],
        )
      const x = codes(N)
      const testX = codes(TEST)
      return Array.from({ length: replicates }, (_, r) => {
        const s = child(stream('leaky-pipeline'), r)
        const y = fromData(Float64Array.from(toFlat(normals(child(s, 'y'), [N]))))
        const testY = fromData(Float64Array.from(toFlat(normals(child(s, 'test'), [TEST]))))
        const folds = kFold({ k: 10, shuffle: true })
        // Outside: the encoder is fitted once on all rows; only the regression is cross-validated.
        const encoded = encoder(step as Step)
          .fit({ x, y })
          .transform(x)
        const outside = crossValidate(linearRegression(), dataset(encoded, y), folds, [meanSquaredError], {
          stream: child(s, 'cv'),
        })
        // Inside: the encoder is a step of the pipeline, refitted on each training fold.
        const whole = pipeline(encoder(step as Step), linearRegression()) as unknown as Fittable<
          Tensor,
          Decides<Tensor, Tensor>
        >
        const inside = crossValidate(whole, dataset(x, y), folds, [meanSquaredError], { stream: child(s, 'cv') })
        const fitted = whole.fit(dataset(x, y))
        return {
          outside: outside.mean.meanSquaredError,
          inside: inside.mean.meanSquaredError,
          truth: meanSquaredError(testY, fitted.decide(testX)),
        }
      })
    },
    [step, categories, replicates],
    { mode: 'release' },
  )
  const rows = computed.value
  const view = useMemo(() => {
    const avg = (k: 'outside' | 'inside' | 'truth') => rows.reduce((a, r) => a + r[k], 0) / rows.length
    const all = rows.flatMap((r) => [r.outside, r.inside, r.truth, 1])
    return {
      diag: [Math.min(...all), Math.max(...all)],
      truth: rows.map((r) => r.truth),
      outside: rows.map((r) => r.outside),
      inside: rows.map((r) => r.inside),
      mean: { outside: avg('outside'), inside: avg('inside'), truth: avg('truth') },
    }
  }, [rows])
  const tx = useAxis({ label: 'true error of the fitted pipeline' })
  const ey = useAxis({ label: '10-fold CV estimate', equal: tx })
  return (
    <Figure
      title="A preprocessing step fitted outside the folds"
      state={state}
      caption={`${N} rows, one categorical input with ${categories} categories of ${perCategory} rows each, and a target of pure noise with variance 1, so no model can beat a mean squared error of 1 (the dashed horizontal line). The input is encoded and then fed to least squares. Each replicate data set gives two points: the 10-fold cross-validation estimate with the encoder fitted once on all rows (outside the folds) and with the encoder refitted inside each training fold, both against the true error of the pipeline fitted to all rows, measured on ${TEST} fresh rows. Target encoding fitted outside the folds reports errors below 1, which is impossible. One-hot encoding learns only the list of categories and gives the same estimate either way.`}
      readouts={{
        'mean over replicates': (
          <>
            <Readout label="encoder outside folds" value={formatNumber(view.mean.outside)} />
            <Readout label="encoder inside folds" value={formatNumber(view.mean.inside)} />
            <Readout label="true error" value={formatNumber(view.mean.truth)} />
          </>
        ),
      }}
    >
      <Plot x={tx} y={ey}>
        <Curve name="estimate = truth" x={view.diag} y={view.diag} muted dashed />
        <Annotation y={1} text="noise variance" dashed emphasis />
        <Points name="encoder outside folds" x={view.truth} y={view.outside} slot={0} stale={computed.stale} />
        <Points name="encoder inside folds" x={view.truth} y={view.inside} slot={1} stale={computed.stale} />
      </Plot>
    </Figure>
  )
}
