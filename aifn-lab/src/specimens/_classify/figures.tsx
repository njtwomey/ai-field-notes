import { useMemo, useState } from 'react'
import {
  adaBoost,
  codeDistance,
  crammerSinger,
  decisionTree,
  dualDecision,
  exhaustiveCode,
  gaussianNaiveBayes,
  gradientBoosting,
  kNearestNeighbours,
  linearDiscriminant,
  nestedDichotomies,
  oneVersusOne,
  oneVersusOneCode,
  oneVersusRest,
  oneVersusRestCode,
  outputCode,
  perceptron,
  quadraticDiscriminant,
  randomCode,
  randomForest,
  smoSteps,
  supportVectorMachine,
  type SmoState,
} from 'aifn/classify'
import { logisticRegression } from 'aifn/estimators'
import { grid2d } from 'aifn/geometry'
import { rbf } from 'aifn/kernels'
import { stream } from 'aifn/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn/tensor'
import { trace } from 'aifn/trace'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type HeatmapOverlay, type Vec2 } from '@lab/viz'
import { DecisionRegionView, formatValue } from '@lab/views'
import { DATASET_OPTIONS, DATASETS, type DatasetName } from './data'

// ── A gallery of decision regions ─────────────────────────────────────────────────────────────────────────────────

type Decider = { decide(x: Tensor): Tensor }

const CLASSIFIERS = {
  knn: {
    label: 'k-nearest neighbours (k = 7)',
    fit: (x: Tensor, y: Tensor) => kNearestNeighbours({ k: 7 }).fit({ x, y }),
  },
  'naive-bayes': { label: 'Gaussian naive Bayes', fit: (x: Tensor, y: Tensor) => gaussianNaiveBayes().fit({ x, y }) },
  lda: { label: 'linear discriminant', fit: (x: Tensor, y: Tensor) => linearDiscriminant().fit({ x, y }) },
  qda: { label: 'quadratic discriminant', fit: (x: Tensor, y: Tensor) => quadraticDiscriminant().fit({ x, y }) },
  perceptron: {
    label: 'perceptron (one-versus-rest)',
    fit: (x: Tensor, y: Tensor) => oneVersusRest(perceptron({ epochs: 50 })).fit({ x, y }),
  },
  svm: {
    label: 'RBF SVM by SMO (one-versus-one)',
    fit: (x: Tensor, y: Tensor) =>
      oneVersusOne(supportVectorMachine({ C: 1, kernel: rbf({ lengthscale: 0.7 }) })).fit({ x, y }),
  },
  'crammer-singer': {
    label: 'Crammer–Singer linear SVM',
    fit: (x: Tensor, y: Tensor) => crammerSinger({ C: 1, maxEpochs: 300 }).fit({ x, y }),
  },
  tree: {
    label: 'decision tree (depth 5)',
    fit: (x: Tensor, y: Tensor) => decisionTree({ maxDepth: 5 }).fit({ x, y }),
  },
  forest: {
    label: 'random forest (50 trees)',
    fit: (x: Tensor, y: Tensor) =>
      randomForest({ trees: 50, maxFeatures: 1 }).fit({ x, y }, { stream: stream('lab/forest') }),
  },
  adaboost: { label: 'AdaBoost (100 stumps)', fit: (x: Tensor, y: Tensor) => adaBoost({ rounds: 100 }).fit({ x, y }) },
  boosting: {
    label: 'gradient boosting (logistic)',
    fit: (x: Tensor, y: Tensor) =>
      gradientBoosting({ loss: 'logistic', stages: 60, learningRate: 0.2, tree: { maxDepth: 2 } }).fit({ x, y }),
  },
  ecoc: {
    label: 'output code (exhaustive, logistic)',
    fit: (x: Tensor, y: Tensor) =>
      outputCode(logisticRegression({ l2: 0.1 }), exhaustiveCode(Math.max(2, ...toFlat(y)) + 1), {
        decoding: 'loss',
      }).fit({ x, y }),
  },
  dichotomies: {
    label: 'nested dichotomies (logistic)',
    fit: (x: Tensor, y: Tensor) => nestedDichotomies(logisticRegression({ l2: 0.1 })).fit({ x, y }),
  },
} as const
type ClassifierName = keyof typeof CLASSIFIERS

export function DecisionRegionsSpecimen() {
  const [dataset, setDataset] = useState<DatasetName>('moons')
  const [name, setName] = useState<ClassifierName>('knn')
  const [query, setQuery] = useState<Vec2>([0.5, 0.25])
  const data = useMemo(() => {
    const d = DATASETS[dataset].make()
    return { x: d.x, y: d.y!, names: d.meta.labelNames }
  }, [dataset])
  const model = useMemo(() => CLASSIFIERS[name].fit(data.x, data.y) as Decider, [name, data])
  // k-NN: ring the query's neighbours. SVM: ring the support vectors of every pairwise machine.
  const overlay = useMemo((): HeatmapOverlay[] => {
    const rows = toRows(data.x)
    if (name === 'knn') {
      const idx = toFlat(
        (model as ReturnType<typeof CLASSIFIERS.knn.fit>).neighbours(fromData(Float64Array.from(query), [1, 2])).index,
      )
      return [
        {
          name: 'neighbours of the query',
          type: 'scatter',
          x: idx.map((i) => rows[i][0]),
          y: idx.map((i) => rows[i][1]),
          emphasis: true,
        },
      ]
    }
    if (name === 'svm') {
      const m = model as ReturnType<typeof CLASSIFIERS.svm.fit>
      const sv = new Set<number>()
      // Each pairwise machine indexes the rows of its own two classes, in row order.
      m.models.forEach((svm, l) => {
        const code = toRows(m.code)
        const members = toFlat(data.y).flatMap((c, i) => (code[c][l] !== 0 ? [i] : []))
        for (const t of toFlat(svm.supportVectors)) sv.add(members[t])
      })
      const list = [...sv]
      return [
        {
          name: 'support vectors',
          type: 'scatter',
          x: list.map((i) => rows[i][0]),
          y: list.map((i) => rows[i][1]),
          emphasis: true,
        },
      ]
    }
    return []
  }, [name, model, query, data])
  const accuracy = useMemo(() => {
    const d = toFlat(model.decide(data.x))
    const y = toFlat(data.y)
    return d.filter((c, i) => c === y[i]).length / y.length
  }, [model, data])
  return (
    <DecisionRegionView
      title="Decision regions of every classifier"
      description="Each classifier's decide(x) over the plane, fitted on the same points; the query point shows the prediction and, where the model has one, its predictive."
      model={model}
      data={data}
      classNames={data.names}
      query={query}
      onQuery={setQuery}
      overlay={overlay}
      controls={
        <>
          <ControlRow label="1 · data">
            <Select label="dataset" value={dataset} onChange={setDataset} options={DATASET_OPTIONS} />
          </ControlRow>
          <ControlRow label="2 · classifier">
            <Select
              label="classifier"
              value={name}
              onChange={setName}
              options={(Object.keys(CLASSIFIERS) as ClassifierName[]).map((value) => ({
                value,
                label: CLASSIFIERS[value].label,
              }))}
            />
          </ControlRow>
        </>
      }
      readouts={<Readout label="training accuracy" value={formatValue(accuracy)} />}
      caption="Drag the query point. For k-NN its seven neighbours are ringed; for the SVM, the support vectors of every pairwise machine. Linear models (LDA, the perceptron, Crammer–Singer) cannot follow the moons or circles; trees cut the plane into axis-aligned boxes; boosting and forests smooth those boxes by averaging."
    />
  )
}

// ── SMO step by step ───────────────────────────────────────────────────────────────────────────────────────────────

export function SmoSpecimen() {
  const [C, setC] = useState(1)
  const [lengthscale, setLengthscale] = useState(0.6)
  const [step, setStep] = useState(12)
  const data = useMemo(() => {
    const d = DATASETS.moons.make()
    const rows = toRows(d.x)
    // Every other point keeps the working pairs readable.
    const keep = rows.map((_, i) => i).filter((i) => i % 2 === 0)
    return {
      x: fromData(Float64Array.from(keep.flatMap((i) => rows[i])), [keep.length, 2]),
      y: fromData(Float64Array.from(keep.map((i) => (toFlat(d.y!)[i] === 1 ? 1 : -1))), [keep.length]),
    }
  }, [])
  const kernel = useMemo(() => rbf({ lengthscale }), [lengthscale])
  const run = useMemo(
    () =>
      trace(smoSteps({ x: data.x, y: data.y, C, kernel }), {}, 400, {
        record: { dual: (s) => s.dualObjective, gap: (s) => s.gap },
      }),
    [data, C, kernel],
  )
  const k = Math.min(step, run.steps.length - 1)
  const state: SmoState = run.steps[k]
  const rows = useMemo(() => toRows(data.x), [data])
  const labels = useMemo(() => toFlat(data.y).map((v) => (v > 0 ? 1 : 0)), [data])
  const field = useMemo(() => grid2d([-1.6, 2.6], [-1.1, 1.6], 70), [])
  const f = useMemo(() => {
    const values = toFlat(dualDecision({ x: data.x, y: data.y, C, kernel }, state.alpha, state.bias)(field.points))
    const [ny, nx] = field.shape
    return Array.from({ length: ny }, (_, i) => values.slice(i * nx, (i + 1) * nx))
  }, [state, data, C, kernel, field])
  const alpha = toFlat(state.alpha)
  const svIdx = alpha.flatMap((a, i) => (a > 1e-9 ? [i] : []))
  const [pi, pj] = state.pair
  const overlay: HeatmapOverlay[] = [
    {
      name: 'points',
      type: 'scatter',
      x: rows.map((r) => r[0]),
      y: rows.map((r) => r[1]),
      group: labels,
      groupNames: ['y = −1', 'y = +1'],
    },
    {
      name: 'α > 0',
      type: 'scatter',
      x: svIdx.map((i) => rows[i][0]),
      y: svIdx.map((i) => rows[i][1]),
      emphasis: true,
    },
    ...(pi >= 0
      ? [
          {
            name: 'working pair',
            type: 'line' as const,
            x: [rows[pi][0], rows[pj][0]],
            y: [rows[pi][1], rows[pj][1]],
            showPoints: true,
          },
        ]
      : []),
  ]
  const series = toFlat(run.series.gap)
  return (
    <Figure
      title="SMO, one working pair at a time"
      description="Sequential minimal optimisation moves two dual variables per step along yᵢαᵢ + yⱼαⱼ = const, clipped to the box [0, C]; the decision function and the KKT gap follow."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · problem">
            <Slider label="C" value={C} min={0.05} max={10} onChange={setC} />
            <Slider label="lengthscale ℓ" value={lengthscale} min={0.2} max={2} onChange={setLengthscale} />
          </ControlRow>
          <ControlRow label="2 · step">
            <Player value={k} onChange={setStep} count={run.steps.length} label="SMO step" />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="step" value={`${k} of ${run.steps.length - 1}`} />
          <Readout label="working pair" value={pi < 0 ? '—' : `(${pi}, ${pj})`} />
          <Readout label="clipped" value={pi < 0 ? '—' : state.clipped ? 'yes' : 'no'} />
          <Readout label="dual objective" value={formatValue(state.dualObjective)} />
          <Readout label="KKT gap" value={formatValue(state.gap)} />
          <Readout label="α > 0" value={svIdx.length} />
          <Readout label="bias b" value={formatValue(state.bias)} />
        </>
      }
      caption="Step through the run: the line joins the pair just updated; ringed points have αᵢ > 0. The colour is f(x) at the current α, with the f = 0 contour as the boundary. The gap m(α) − M(α) falls to the tolerance 10⁻³, where the run stops. A small C caps every α and keeps many points in the margin."
    >
      <Subplots rows={2} heightRatios={[2, 1]}>
        <Panel>
          <Heatmap
            x={toFlat(field.x)}
            y={toFlat(field.y)}
            z={f}
            scale="diverging"
            range={[-2, 2]}
            contours={{ levels: [0] }}
            overlay={overlay}
            equalAspect
            valueLabel="f(x)"
            xLabel="x₀"
            yLabel="x₁"
          />
        </Panel>
        <Panel>
          <XYChart
            series={[
              { name: 'KKT gap', type: 'line', x: run.index, y: series },
              { name: 'now', type: 'scatter', x: [run.index[k]], y: [series[k]], emphasis: true },
            ]}
            xLabel="step"
            yLabel="gap"
            legend={false}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ── Output codes ───────────────────────────────────────────────────────────────────────────────────────────────────

const CODES = {
  ovr: { label: 'one-versus-rest', make: (K: number) => oneVersusRestCode(K) },
  ovo: { label: 'one-versus-one', make: (K: number) => oneVersusOneCode(K) },
  exhaustive: { label: 'exhaustive', make: (K: number) => exhaustiveCode(K) },
  random: { label: 'random dense (10 columns)', make: (K: number) => randomCode(stream('lab/codes'), K, 10) },
  sparse: {
    label: 'random sparse (10 columns)',
    make: (K: number) => randomCode(stream('lab/codes'), K, 10, { sparse: true }),
  },
} as const
type CodeName = keyof typeof CODES

export function OutputCodeSpecimen() {
  const [code, setCode] = useState<CodeName>('exhaustive')
  const [K, setK] = useState(5)
  const matrix = useMemo(() => CODES[code].make(K), [code, K])
  const rows = toRows(matrix)
  const L = rows[0].length
  const distance = codeDistance(matrix)
  return (
    <Figure
      title="Code matrices and their distances"
      description="Each row is a class's codeword and each column a binary problem (+1 positive, −1 negative, 0 left out); the minimum row distance says how many binary errors decoding can absorb."
      controls={
        <>
          <Select
            label="code"
            value={code}
            onChange={setCode}
            options={(Object.keys(CODES) as CodeName[]).map((v) => ({ value: v, label: CODES[v].label }))}
          />
          <Slider label="classes K" value={K} min={3} max={7} step={1} onChange={setK} />
        </>
      }
      readouts={
        <>
          <Readout label="binary problems L" value={L} />
          <Readout label="minimum distance" value={distance} />
          <Readout label="errors corrected" value={Math.max(0, Math.floor((distance - 1) / 2))} />
        </>
      }
      caption="One-versus-rest rows differ in two columns, so one wrong classifier can already tie two classes; the exhaustive code's rows differ in 2^(K−2) columns. Zeros (one-versus-one, sparse codes) do not count towards the distance."
    >
      <Heatmap
        x={Array.from({ length: L }, (_, l) => l)}
        y={Array.from({ length: K }, (_, k) => k)}
        z={rows}
        scale="diverging"
        range={[-1, 1]}
        xLabel="binary problem"
        yLabel="class"
        valueLabel="code"
      />
    </Figure>
  )
}
