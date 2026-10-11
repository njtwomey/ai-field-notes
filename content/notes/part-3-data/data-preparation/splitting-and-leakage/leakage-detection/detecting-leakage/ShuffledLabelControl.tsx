import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, integers, normals, shuffle, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { pipeline } from 'aifn-compute/learning/compose'
import { dataset } from 'aifn-compute/learning/estimators'
import { auroc } from 'aifn-compute/learning/metrics'
import { crossValidate, stratifiedKFold } from 'aifn-compute/learning/validate'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { targetEncoder } from 'aifn-methods/learning/preprocessing'

const PROCEDURES = ['encoder fitted on all rows', 'encoder fitted inside each fold']
const WIDTH = 0.34
const SHUFFLES = 5

/**
 * `rows` rows with one categorical feature of `categories` levels. Each level has an effect drawn from a normal
 * distribution with standard deviation `signal`; a row is positive when its level's effect plus unit noise is above 0.
 */
function categoricalData(seed: number, rows: number, categories: number, signal: number) {
  const s = stream(`shuffled-label-control/${seed}`)
  const level = toFlat(integers(child(s, 'level'), categories, { shape: [rows] }))
  const effect = toFlat(normals(child(s, 'effect'), [categories]))
  const noise = toFlat(normals(child(s, 'noise'), [rows]))
  const x = Array.from(level, (c) => `level ${c}`)
  const y = Float64Array.from(level, (c, i) => (signal * effect[c] + noise[i] > 0 ? 1 : 0))
  // Five independent shuffles of the labels, for the control.
  const shuffled = Array.from({ length: SHUFFLES }, (_, r) => {
    const order = shuffle(
      child(s, 'shuffle', r),
      Array.from({ length: rows }, (_, i) => i),
    )
    return Float64Array.from(order, (j) => y[j])
  })
  return { x, y, shuffled, s }
}

/** Cross-validated AUC of target encoding plus logistic regression, with true and with shuffled labels. */
export function ShuffledLabelControl() {
  const state = useFigureState({
    rows: int(400, { ge: 100, le: 2000, suggestions: [200, 400, 1000], label: 'rows' }),
    categories: int(100, { ge: 5, le: 500, suggestions: [10, 100, 300], label: 'categories' }),
    signal: slider(0, 2, 1, { step: 0.1, label: 'signal in the category' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { rows, categories, signal, seed } = state
  const computed = useComputed(
    () => {
      const { x, y, shuffled, s } = categoricalData(seed, rows, categories, signal)
      const splitter = stratifiedKFold({ k: 5, shuffle: true })
      const score = (labels: Float64Array, inside: boolean) => {
        if (inside) {
          // The pipeline fits the encoder on each training fold. Its type does not yet state that a pipeline ending in a
          // classifier serves scores, or that categorical inputs are features, hence the casts.
          const model = pipeline(targetEncoder({ smooth: 1 }), logisticRegression())
          const data = dataset(x as never, fromData(labels))
          return crossValidate(model as never, data, splitter, [auroc], { stream: child(s, 'cv') }).mean.auroc
        }
        // The leak: every row's encoding is the mean label of its category over all rows, test rows included.
        const encoded = targetEncoder({ smooth: 1 })
          .fit({ x, y: fromData(labels) })
          .transform(x)
        return crossValidate(logisticRegression(), dataset(encoded, fromData(labels)), splitter, [auroc], {
          stream: child(s, 'cv'),
        }).mean.auroc
      }
      return {
        real: [score(y, false), score(y, true)],
        control: [false, true].map((inside) => shuffled.reduce((a, l) => a + score(l, inside), 0) / SHUFFLES),
      }
    },
    [rows, categories, signal, seed],
    { mode: 'release' },
  )
  const view = computed.value
  const px = useAxis({ categories: PROCEDURES })
  const py = useAxis({ label: 'cross-validated AUC', range: [0.3, 1] })
  const positions = useMemo(() => [0, 1], [])
  return (
    <Figure
      title="A shuffled-label control exposes a leak"
      state={state}
      hoverReadout={false}
      caption="One categorical feature, its levels each carrying some signal about a binary label. Each level is replaced by the mean label of its rows (target encoding), and a logistic regression is scored by 5-fold cross-validated AUC. Blue: true labels. Orange: the same pipeline after the labels are shuffled, which destroys any real relation, averaged over five shuffles. Fitted on all rows, the encoder writes each test row's own label into its feature, and the shuffled-label control still scores far above 0.5: the pipeline leaks. Fitted inside each training fold, the encoder sees no test labels, and the control falls to about 0.5. More categories with fewer rows each make the leak larger."
      readouts={
        <>
          <Readout label="all rows, true labels" value={formatNumber(view.real[0])} />
          <Readout label="all rows, shuffled" value={formatNumber(view.control[0])} />
          <Readout label="inside folds, true labels" value={formatNumber(view.real[1])} />
          <Readout label="inside folds, shuffled" value={formatNumber(view.control[1])} />
        </>
      }
    >
      <Plot x={px} y={py}>
        <Curve name="chance" x={[-0.5, 1.5]} y={[0.5, 0.5]} muted dashed />
        <Bars
          name="true labels"
          x={positions.map((p) => p - WIDTH / 2)}
          y={view.real}
          width={WIDTH}
          base={0.3}
          slot={0}
        />
        <Bars
          name="shuffled labels (mean of 5)"
          x={positions.map((p) => p + WIDTH / 2)}
          y={view.control}
          width={WIDTH}
          base={0.3}
          slot={1}
        />
      </Plot>
    </Figure>
  )
}
