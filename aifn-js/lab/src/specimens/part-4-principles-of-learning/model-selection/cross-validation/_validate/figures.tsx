import { useMemo } from 'react'
import { pipeline } from 'aifn/learning/compose'
import { dataset } from 'aifn/learning/estimators'
import { oneOf, space } from 'aifn/foundation/space'
import { accuracy, logLoss, meanSquaredError } from 'aifn/learning/metrics'
import { linearRegression } from 'aifn-applied/learning/linear'
import { logisticRegression } from 'aifn-applied/learning/generalised/glm'
import { standardScaler } from 'aifn-applied/learning/preprocessing'
import { child, normals, stream } from 'aifn/foundation/random'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import {
  crossValidate,
  expandingWindow,
  gridSearch,
  groupKFold,
  kFold,
  nested,
  repeated,
  rollingOrigin,
  shuffleSplit,
  stratifiedKFold,
  type Splitter,
} from 'aifn/learning/validate'
import { Figure } from '@lab/layout'
import { choice, row, slider, useComputed, useFigureState } from '@lab/state'
import { Curve, Plot, Points, Readout, useAxis } from '@lab/viz'
import { CrossValidationPanel, formatValue } from '@lab/views'

type SplitterName =
  | 'k-fold'
  | 'shuffled k-fold'
  | 'repeated k-fold'
  | 'stratified k-fold'
  | 'grouped k-fold'
  | 'shuffle-split'
  | 'expanding window'
  | 'rolling origin'

function makeSplitter(name: SplitterName, k: number, n: number): Splitter {
  switch (name) {
    case 'k-fold':
      return kFold({ k })
    case 'shuffled k-fold':
      return kFold({ k, shuffle: true })
    case 'repeated k-fold':
      return repeated(kFold({ k, shuffle: true }), 2)
    case 'stratified k-fold':
      return stratifiedKFold({ k, shuffle: true })
    case 'grouped k-fold':
      return groupKFold({ k })
    case 'shuffle-split':
      return shuffleSplit({ splits: k, testSize: 0.2, trainSize: 0.6 })
    case 'expanding window':
      return expandingWindow({ splits: k })
    case 'rolling origin':
      return rollingOrigin({ window: Math.floor(n / 2), horizon: Math.max(1, Math.floor(n / (2 * k))) })
  }
}

const SPLITTERS: SplitterName[] = [
  'k-fold',
  'shuffled k-fold',
  'repeated k-fold',
  'stratified k-fold',
  'grouped k-fold',
  'shuffle-split',
  'expanding window',
  'rolling origin',
]

export function SplitterSpecimen() {
  const state = useFigureState({
    data: row('1 · data', { n: slider(20, 200, 60, { label: 'n rows', step: 1 }) }),
    split: row('2 · splitter', {
      name: choice(SPLITTERS, 'stratified k-fold', { label: 'splitter' }),
      k: slider(2, 10, 5, { label: 'k (splits)', step: 1 }),
    }),
  })
  const { n } = state.data
  const { name, k } = state.split
  // Imbalanced classes (about one in four positive) and groups of seven consecutive rows.
  const data = useMemo(() => {
    const s = stream('splitters')
    const x = toFlat(normals(child(s, 'x'), [n, 2]))
    const e = toFlat(normals(child(s, 'e'), [n]))
    const y = Float64Array.from({ length: n }, (_, i) =>
      1.2 * x[2 * i] - 0.6 * x[2 * i + 1] + 0.7 * e[i] > 0.9 ? 1 : 0,
    )
    return dataset(fromData(Float64Array.from(x), [n, 2]), fromData(y), {
      groups: Array.from({ length: n }, (_, i) => Math.floor(i / 7)),
    })
  }, [n])
  const result = useMemo(() => {
    try {
      const cv = crossValidate(
        pipeline(standardScaler(), logisticRegression({ l2: 1 })),
        data,
        makeSplitter(name, k, n),
        [accuracy, logLoss],
        { stream: stream('splitters/cv') },
      )
      return { cv, error: '' }
    } catch (e) {
      return { cv: null, error: (e as Error).message }
    }
  }, [data, name, k, n])
  const positives = toFlat(data.y).reduce((a, b) => a + b, 0)
  return (
    <Figure
      title="Fold assignments and per-fold scores"
      purpose="A splitter decides which rows each fold trains and tests on; the per-fold test scores, and how much they vary, depend on that choice."
      state={state}
      readouts={{ data: <Readout label="positive rows" value={`${positives} of ${n}`} /> }}
      caption="Each row of the matrix is one split: train (slot 0), test (slot 1) or unused (grey). k-fold tests contiguous blocks; stratified k-fold keeps about one positive in four in every test fold; grouped k-fold keeps each block of seven rows together; the time-series splitters never train on rows after the test block. Below, each fold's test score of pipeline(standardScaler(), logisticRegression())."
    >
      {result.cv ? (
        <CrossValidationPanel result={result.cv} />
      ) : (
        <p className="text-sm text-muted-foreground">{result.error}</p>
      )}
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const PENALTIES = [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000] as const
/** The search space: the ridge penalty, one of `PENALTIES`. */
const PENALTY_SPACE = space({ l2: oneOf(PENALTIES) })

export function NestedOptimismSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      n: slider(25, 200, 50, { label: 'n rows', step: 1 }),
      features: slider(3, 60, 30, { label: 'features', step: 1 }),
    }),
    runs: row('2 · replicates', { replicates: slider(1, 20, 8, { label: 'replicate datasets', step: 1 }) }),
  })
  const { n, features } = state.data
  const { replicates } = state.runs
  // Every replicate is a nested 5 × 5 grid search: run on release, not on every slider move.
  const computed = useComputed(
    () => {
      const search = gridSearch((p) => linearRegression({ l2: p.l2 }), PENALTY_SPACE, { metrics: [meanSquaredError] })
      return Array.from({ length: replicates }, (_, r) => {
        // Three informative features among `features`; unit noise.
        const s = child(stream('nested'), 'replicate', r)
        const x = toFlat(normals(child(s, 'x'), [n, features]))
        const e = toFlat(normals(child(s, 'e'), [n]))
        const y = Float64Array.from(
          { length: n },
          (_, i) => x[i * features] - 0.8 * x[i * features + 1] + 0.5 * x[i * features + 2] + e[i],
        )
        return nested(
          kFold({ k: 5 }),
          kFold({ k: 5 }),
          search,
          dataset(fromData(Float64Array.from(x), [n, features]), fromData(y)),
        )
      })
    },
    [n, features, replicates],
    { mode: 'release' },
  )
  const runs = computed.value
  const pts = useMemo(() => {
    const all = runs.flatMap((r) => [r.nestedScore, r.unnestedScore])
    const lo = Math.min(...all)
    const hi = Math.max(...all)
    return { diag: [lo, hi], x: runs.map((r) => r.unnestedScore), y: runs.map((r) => r.nestedScore) }
  }, [runs])
  const ux = useAxis({ label: 'unnested MSE (best inner mean)' })
  const ny = useAxis({ label: 'nested MSE (outer folds)', equal: ux })
  const optimism = runs.reduce((a, r) => a + r.optimism, 0) / runs.length
  const chosen = runs[0]?.perFold.map((f) => formatValue(f.params.l2)).join(', ')
  return (
    <Figure
      purpose="Reporting the best inner score of a tuning run is optimistic, because the folds that chose λ also score it; nested cross-validation scores the tuning on outer folds it never saw, so its points sit above the diagonal."
      title="Nested cross-validation and the optimism of tuning"
      state={state}
      readouts={{
        optimism: (
          <>
            <Readout label="mean optimism (nested − unnested MSE)" value={formatValue(optimism)} />
            <Readout label="λ chosen per outer fold (replicate 1)" value={chosen} />
          </>
        ),
      }}
      caption="Ridge regression tuned over λ by 5-fold grid search. The unnested estimate is the best inner mean MSE on all the data, which selected λ on the same folds it reports; nested cross-validation scores the whole tuning procedure on outer folds it never saw. Points above the diagonal show the unnested estimate's optimism, larger with more features per row."
    >
      <Plot x={ux} y={ny}>
        <Curve name="equal estimates" x={pts.diag} y={pts.diag} muted dashed />
        <Points name="replicates" x={pts.x} y={pts.y} slot={0} stale={computed.stale} />
      </Plot>
    </Figure>
  )
}
