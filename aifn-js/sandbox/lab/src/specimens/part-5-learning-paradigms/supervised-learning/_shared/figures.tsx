import { useMemo, useState } from 'react'
import { dataset } from 'aifn/learning/estimators'
import { adaBoost, gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import {
  codeDistance,
  exhaustiveCode,
  nestedDichotomies,
  oneVersusOne,
  oneVersusOneCode,
  oneVersusRest,
  oneVersusRestCode,
  outputCode,
  randomCode,
} from 'aifn-methods/learning/reductions'
import {
  crammerSinger,
  dualDecision,
  smoSteps,
  supportVectorMachine,
  type SmoState,
} from 'aifn-methods/learning/kernel-methods'
import { decisionTree } from 'aifn-methods/learning/trees-and-ensembles'
import {
  gaussianNaiveBayes,
  linearDiscriminant,
  quadraticDiscriminant,
} from 'aifn-methods/learning/generative-classifiers'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import { perceptron } from 'aifn-methods/learning/linear'
import { randomForest } from 'aifn-methods/learning/trees-and-ensembles/bagging'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { grid2d } from 'aifn/numerics/geometry'
import { rbf } from 'aifn/learning/kernels'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, toggle, useFigureState } from '@lab/state'
import { Contours, Curve, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'
import { datasetChoice, DecisionRegionPanel, formatValue } from '@lab/views'
import { DATASETS } from './data'

// ── A gallery of decision regions ─────────────────────────────────────────────────────────────────────────────────

type Decider = { decide(x: Tensor): Tensor }

const CLASSIFIERS = {
  knn: {
    label: 'k-nearest neighbours (k = 7)',
    fit: (x: Tensor, y: Tensor) => kNearestNeighbours({ k: 7 }).fit(dataset(x, y)),
  },
  'naive-bayes': {
    label: 'Gaussian naive Bayes',
    fit: (x: Tensor, y: Tensor) => gaussianNaiveBayes().fit(dataset(x, y)),
  },
  lda: { label: 'linear discriminant', fit: (x: Tensor, y: Tensor) => linearDiscriminant().fit(dataset(x, y)) },
  qda: { label: 'quadratic discriminant', fit: (x: Tensor, y: Tensor) => quadraticDiscriminant().fit(dataset(x, y)) },
  perceptron: {
    label: 'perceptron (one-versus-rest)',
    fit: (x: Tensor, y: Tensor) => oneVersusRest(perceptron({ epochs: 50 })).fit(dataset(x, y)),
  },
  svm: {
    label: 'RBF SVM by SMO (one-versus-one)',
    fit: (x: Tensor, y: Tensor) =>
      oneVersusOne(supportVectorMachine({ C: 1, kernel: rbf({ lengthscale: 0.7 }) })).fit(dataset(x, y)),
  },
  'crammer-singer': {
    label: 'Crammer–Singer linear SVM',
    fit: (x: Tensor, y: Tensor) => crammerSinger({ C: 1, maxSteps: 300 }).fit(dataset(x, y)),
  },
  tree: {
    label: 'decision tree (depth 5)',
    fit: (x: Tensor, y: Tensor) => decisionTree({ maxDepth: 5 }).fit(dataset(x, y)),
  },
  forest: {
    label: 'random forest (50 trees)',
    fit: (x: Tensor, y: Tensor) =>
      randomForest({ trees: 50, maxFeatures: 1 }).fit(dataset(x, y), { stream: stream('lab/forest') }),
  },
  adaboost: {
    label: 'AdaBoost (100 stumps)',
    fit: (x: Tensor, y: Tensor) => adaBoost({ rounds: 100 }).fit(dataset(x, y)),
  },
  boosting: {
    label: 'gradient boosting (logistic)',
    fit: (x: Tensor, y: Tensor) =>
      gradientBoosting({ loss: 'logistic', stages: 60, learningRate: 0.2, tree: { maxDepth: 2 } }).fit(dataset(x, y)),
  },
  ecoc: {
    label: 'output code (exhaustive, logistic)',
    fit: (x: Tensor, y: Tensor) =>
      outputCode(logisticRegression({ l2: 0.1 }), exhaustiveCode(Math.max(2, ...toFlat(y)) + 1), {
        decoding: 'loss',
      }).fit(dataset(x, y)),
  },
  dichotomies: {
    label: 'nested dichotomies (logistic)',
    fit: (x: Tensor, y: Tensor) => nestedDichotomies(logisticRegression({ l2: 0.1 })).fit(dataset(x, y)),
  },
} as const
type ClassifierName = keyof typeof CLASSIFIERS

const CLASSIFIER_OPTIONS = (Object.keys(CLASSIFIERS) as ClassifierName[]).map((value) => ({
  value,
  label: CLASSIFIERS[value].label,
}))

/** The classify page's sets: registered generators with typed size and noise (`datasetChoice`). */
const CLASSIFY_DATA = datasetChoice({
  blobs: {
    label: 'three blobs',
    n: 150,
    noise: 0.9,
    knobs: {
      centers: [
        [-2, -1],
        [2, -1],
        [0, 2],
      ],
    },
  },
  moons: { label: 'two moons', n: 160, noise: 0.15 },
  circles: { label: 'two circles', n: 160, noise: 0.08, knobs: { factor: 0.45 } },
  xor: { label: 'XOR', n: 160, noise: 0.45, knobs: { kind: 'gaussian' } },
})

export function DecisionRegionsSpecimen() {
  const state = useFigureState({
    data: CLASSIFY_DATA.field({ label: '1 · data', initial: 'moons' }),
    model: row('2 · classifier', { name: choice(CLASSIFIER_OPTIONS, 'knn', { label: 'classifier' }) }),
    show: row('3 · show', { boundary: toggle(true, 'decision boundaries') }),
    qx: slider(-3, 3, 0.5, { onChart: true }),
    qy: slider(-3, 3, 0.25, { onChart: true }),
  })
  const dataKey = CLASSIFY_DATA.key(state.data)
  const name = state.model.name as ClassifierName
  const query = useMemo((): [number, number] => [state.qx, state.qy], [state.qx, state.qy])
  const data = useMemo(() => {
    const d = CLASSIFY_DATA.make(state.data, `lab/classify/${state.data.key}`)
    return { x: d.x, y: d.y!, names: d.meta.labelNames }
    // oxlint-disable-next-line react-hooks/exhaustive-deps -- the key changes exactly when the dataset does
  }, [dataKey])
  const model = useMemo(() => CLASSIFIERS[name].fit(data.x, data.y) as Decider, [name, data])
  // k-NN: ring the query's neighbours. SVM: ring the support vectors of every pairwise machine.
  const overlay = useMemo(() => {
    const rows = toRows(data.x)
    if (name === 'knn') {
      const idx = toFlat(
        (model as ReturnType<typeof CLASSIFIERS.knn.fit>).neighbours(fromData(Float64Array.from(query), [1, 2])).index,
      )
      return (
        <Points name="neighbours of the query" x={idx.map((i) => rows[i][0])} y={idx.map((i) => rows[i][1])} emphasis />
      )
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
      return <Points name="support vectors" x={list.map((i) => rows[i][0])} y={list.map((i) => rows[i][1])} emphasis />
    }
    return null
  }, [name, model, query, data])
  const accuracy = useMemo(() => {
    const d = toFlat(model.decide(data.x))
    const y = toFlat(data.y)
    return d.filter((c, i) => c === y[i]).length / y.length
  }, [model, data])
  return (
    <Figure
      title="Decision regions of every classifier"
      purpose="Each classifier's decide(x) over the plane, fitted on the same points: linear models cut straight, trees cut boxes, neighbours and kernels follow the data's shape."
      state={state}
      defaultSize="L"
      readouts={<Readout label="training accuracy" value={formatValue(accuracy)} />}
      caption="Drag the query point: its prediction and, where the model has one, its predictive are read out. For k-NN its seven neighbours are ringed; for the SVM, the support vectors of every pairwise machine. Linear models (LDA, the perceptron, Crammer–Singer) cannot follow the moons or circles; trees cut the plane into axis-aligned boxes; boosting and forests smooth those boxes by averaging. With decision boundaries on, ink lines mark where the decided class changes."
    >
      <DecisionRegionPanel
        model={model}
        data={data}
        classNames={data.names}
        query={query}
        onQuery={([a, b]) => {
          state.set('qx', a)
          state.set('qy', b)
        }}
        overlay={overlay}
        boundary={state.show.boundary}
      />
    </Figure>
  )
}

// ── SMO step by step ───────────────────────────────────────────────────────────────────────────────────────────────

export function SmoSpecimen() {
  const figure = useFigureState({
    problem: row('1 · problem', {
      C: slider(0.05, 10, 1, { label: 'C' }),
      lengthscale: slider(0.2, 2, 0.6, { label: 'lengthscale ℓ' }),
    }),
  })
  const { C, lengthscale } = figure.problem
  const [step, setStep] = useState(0)
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
  const [pi, pj] = state.pair
  const pts = useMemo(
    () => ({ x: rows.map((r) => r[0]), y: rows.map((r) => r[1]), fx: toFlat(field.x), fy: toFlat(field.y) }),
    [rows, field],
  )
  const sv = useMemo(() => {
    const idx = toFlat(state.alpha).flatMap((a, i) => (a > 1e-9 ? [i] : []))
    return { n: idx.length, x: idx.map((i) => rows[i][0]), y: idx.map((i) => rows[i][1]) }
  }, [state, rows])
  const pair = useMemo(
    () => (pi >= 0 ? { x: [rows[pi][0], rows[pj][0]], y: [rows[pi][1], rows[pj][1]] } : null),
    [pi, pj, rows],
  )
  const gap = useMemo(() => ({ x: Array.from(run.index), y: toFlat(run.series.gap) }), [run])
  const now = useMemo(() => ({ x: [gap.x[k]], y: [gap.y[k]] }), [gap, k])
  const x0 = useAxis({ label: 'x₀', range: [-1.6, 2.6], nice: false })
  const x1 = useAxis({ label: 'x₁', range: [-1.1, 1.6], nice: false, equal: x0 })
  const stepAxis = useAxis({ label: 'step', hold: 'initial', key: run })
  const gapAxis = useAxis({ label: 'KKT gap', hold: 'initial', key: run })
  return (
    <Figure
      title="SMO, one working pair at a time"
      purpose="Sequential minimal optimisation moves two dual variables per step along yᵢαᵢ + yⱼαⱼ = const, clipped to the box [0, C]; the decision function and the KKT gap follow."
      state={figure}
      defaultSize="L"
      controls={
        <ControlRow label="2 · step">
          <Player value={k} onChange={setStep} count={run.steps.length} label="SMO step" />
        </ControlRow>
      }
      readouts={{
        'this step': (
          <>
            <Readout label="step" value={`${k} of ${run.steps.length - 1}`} />
            <Readout label="working pair" value={pi < 0 ? '—' : `(${pi}, ${pj})`} />
            <Readout label="clipped" value={pi < 0 ? '—' : state.clipped ? 'yes' : 'no'} />
          </>
        ),
        solution: (
          <>
            <Readout label="dual objective" value={formatValue(state.dualObjective)} />
            <Readout label="KKT gap" value={formatValue(state.gap)} />
            <Readout label="α > 0" value={sv.n} />
            <Readout label="bias b" value={formatValue(state.bias)} />
          </>
        ),
      }}
      caption="Step through the run: the line joins the pair just updated; ringed points have αᵢ > 0. The colour is f(x) at the current α, with the ink f = 0 contour as the boundary. The gap m(α) − M(α) falls to the tolerance 10⁻³, where the run stops. A small C caps every α and keeps many points in the margin."
    >
      <Plots rows={2} heights={[2, 1]}>
        <Plot x={x0} y={x1}>
          <Raster x={pts.fx} y={pts.fy} z={f} scale="diverging" range={[-2, 2]} valueLabel="f(x)" />
          <Contours x={pts.fx} y={pts.fy} z={f} levels={[0]} />
          <Points name="points" x={pts.x} y={pts.y} group={labels} groupNames={['y = −1', 'y = +1']} />
          <Points name="α > 0" x={sv.x} y={sv.y} emphasis />
          {pair && <Curve name="working pair" x={pair.x} y={pair.y} showPoints emphasis />}
        </Plot>
        <Plot x={stepAxis} y={gapAxis} legend={false}>
          <Curve name="KKT gap" x={gap.x} y={gap.y} />
          <Points name="now" x={now.x} y={now.y} emphasis />
        </Plot>
      </Plots>
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

const CODE_OPTIONS = (Object.keys(CODES) as CodeName[]).map((v) => ({ value: v, label: CODES[v].label }))

export function OutputCodeSpecimen() {
  const state = useFigureState({
    code: choice(CODE_OPTIONS, 'exhaustive', { label: 'code' }),
    K: slider(3, 7, 5, { label: 'classes K', step: 1 }),
  })
  const code = state.code as CodeName
  const { K } = state
  const matrix = useMemo(() => CODES[code].make(K), [code, K])
  const grid = useMemo(() => {
    const rows = toRows(matrix) as number[][]
    return {
      z: rows,
      x: Array.from({ length: rows[0].length }, (_, l) => l),
      y: Array.from({ length: rows.length }, (_, k) => k),
    }
  }, [matrix])
  const L = grid.x.length
  const distance = codeDistance(matrix)
  const xa = useAxis({ label: 'binary problem' })
  const ya = useAxis({ label: 'class' })
  return (
    <Figure
      title="Code matrices and their distances"
      purpose="Each row is a class's codeword and each column a binary problem; the minimum distance between rows says how many binary errors decoding can absorb."
      state={state}
      readouts={
        <>
          <Readout label="binary problems L" value={L} />
          <Readout label="minimum distance" value={distance} />
          <Readout label="errors corrected" value={Math.max(0, Math.floor((distance - 1) / 2))} />
        </>
      }
      caption="Red is +1 (the class is on the positive side of that problem), blue −1, pale 0 (left out). One-versus-rest rows differ in two columns, so one wrong classifier can already tie two classes; the exhaustive code's rows differ in 2^(K−2) columns. Zeros (one-versus-one, sparse codes) do not count towards the distance."
    >
      <Plot x={xa} y={ya}>
        <Raster x={grid.x} y={grid.y} z={grid.z} scale="diverging" range={[-1, 1]} valueLabel="code" />
      </Plot>
    </Figure>
  )
}
