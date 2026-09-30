import { useMemo, useState } from 'react'
import { pipeline } from 'aifn/compose'
import { accuracy, linearRegression, logisticRegression, logLoss, meanSquaredError } from 'aifn/estimators'
import { standardScaler } from 'aifn/preprocess'
import { normals, stream } from 'aifn/random'
import { fromData, toFlat } from 'aifn/tensor'
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
} from 'aifn/validate'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart, type XYSeries } from '@lab/viz'
import { CrossValidationView, formatValue } from '@lab/views'

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

export function SplitterSpecimen() {
  const [name, setName] = useState<SplitterName>('stratified k-fold')
  const [k, setK] = useState(5)
  const [n, setN] = useState(60)
  // Imbalanced classes (about one in four positive) and groups of seven consecutive rows.
  const data = useMemo(() => {
    const s = stream('splitters')
    const x = toFlat(normals(s.child('x'), [n, 2]))
    const e = toFlat(normals(s.child('e'), [n]))
    const y = Float64Array.from({ length: n }, (_, i) =>
      1.2 * x[2 * i] - 0.6 * x[2 * i + 1] + 0.7 * e[i] > 0.9 ? 1 : 0,
    )
    return {
      x: fromData(Float64Array.from(x), [n, 2]),
      y: fromData(y),
      groups: Array.from({ length: n }, (_, i) => Math.floor(i / 7)),
    }
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
  const controls = (
    <>
      <Select
        label="splitter"
        value={name}
        onChange={setName}
        options={[
          'k-fold',
          'shuffled k-fold',
          'repeated k-fold',
          'stratified k-fold',
          'grouped k-fold',
          'shuffle-split',
          'expanding window',
          'rolling origin',
        ]}
      />
      <Slider label="k (splits)" value={k} min={2} max={10} step={1} onChange={setK} />
      <Slider label="n rows" value={n} min={20} max={200} step={1} onChange={setN} />
    </>
  )
  if (!result.cv) {
    return (
      <Figure title="Splitters" controls={controls}>
        <p className="text-sm text-muted-foreground">{result.error}</p>
      </Figure>
    )
  }
  const positives = toFlat(data.y).reduce((a, b) => a + b, 0)
  return (
    <CrossValidationView
      result={result.cv}
      controls={controls}
      readouts={<Readout label="positive rows" value={`${positives} of ${n}`} />}
      caption="Each row of the matrix is one split: train (slot 0), test (slot 1) or unused (grey). k-fold tests contiguous blocks; stratified k-fold keeps about one positive in four in every test fold; grouped k-fold keeps each block of seven rows together; the time-series splitters never train on rows after the test block. Below, each fold's test score of pipeline(standardScaler(), logisticRegression())."
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------

const PENALTIES = [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]

export function NestedOptimismSpecimen() {
  const [n, setN] = useState(50)
  const [features, setFeatures] = useState(30)
  const [replicates, setReplicates] = useState(8)
  const runs = useMemo(() => {
    const search = gridSearch((p) => linearRegression({ l2: p.l2 }), { l2: PENALTIES }, { metrics: [meanSquaredError] })
    return Array.from({ length: replicates }, (_, r) => {
      // Three informative features among `features`; unit noise.
      const s = stream('nested').child('replicate', r)
      const x = toFlat(normals(s.child('x'), [n, features]))
      const e = toFlat(normals(s.child('e'), [n]))
      const y = Float64Array.from(
        { length: n },
        (_, i) => x[i * features] - 0.8 * x[i * features + 1] + 0.5 * x[i * features + 2] + e[i],
      )
      return nested(kFold({ k: 5 }), kFold({ k: 5 }), search, {
        x: fromData(Float64Array.from(x), [n, features]),
        y: fromData(y),
      })
    })
  }, [n, features, replicates])
  const series = useMemo((): XYSeries[] => {
    const all = runs.flatMap((r) => [r.nestedScore, r.unnestedScore])
    const lo = Math.min(...all)
    const hi = Math.max(...all)
    return [
      { name: 'equal estimates', type: 'line', x: [lo, hi], y: [lo, hi], muted: true, dashed: true },
      {
        name: 'replicates',
        type: 'scatter',
        x: runs.map((r) => r.unnestedScore),
        y: runs.map((r) => r.nestedScore),
        slot: 0,
      },
    ]
  }, [runs])
  const optimism = runs.reduce((a, r) => a + r.optimism, 0) / runs.length
  const chosen = runs[0]?.perFold.map((f) => formatValue(f.params.l2)).join(', ')
  return (
    <Figure
      title="Nested cross-validation and the optimism of tuning"
      controls={
        <>
          <Slider label="n rows" value={n} min={25} max={200} step={1} onChange={setN} />
          <Slider label="features" value={features} min={3} max={60} step={1} onChange={setFeatures} />
          <Slider label="replicate datasets" value={replicates} min={1} max={20} step={1} onChange={setReplicates} />
        </>
      }
      readouts={
        <>
          <Readout label="mean optimism (nested − unnested MSE)" value={formatValue(optimism)} />
          <Readout label="λ chosen per outer fold (replicate 1)" value={chosen} />
        </>
      }
      caption="Ridge regression tuned over λ by 5-fold grid search. The unnested estimate is the best inner mean MSE on all the data, which selected λ on the same folds it reports; nested cross-validation scores the whole tuning procedure on outer folds it never saw. Points above the diagonal show the unnested estimate's optimism, larger with more features per row."
    >
      <XYChart series={series} xLabel="unnested MSE (best inner mean)" yLabel="nested MSE (outer folds)" />
    </Figure>
  )
}
