import { useMemo, useState } from 'react'
import { dataset, classProbabilities, hasPredictive, type Distribution } from 'aifn-compute/learning/estimators'
import { fromData, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { stream } from 'aifn-compute/foundation/random'
import { grid2d } from 'aifn-compute/numerics/geometry'
import { rbf } from 'aifn-compute/learning/kernels'
import { blobs, moons, circles, xor } from 'aifn-methods/data/synthetic'
import {
  gaussianNaiveBayes,
  linearDiscriminant,
  quadraticDiscriminant,
} from 'aifn-methods/learning/generative-classifiers'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { perceptron } from 'aifn-methods/learning/linear'
import { supportVectorMachine, crammerSinger } from 'aifn-methods/learning/kernel-methods'
import { decisionTree } from 'aifn-methods/learning/trees-and-ensembles'
import { randomForest } from 'aifn-methods/learning/trees-and-ensembles/bagging'
import { adaBoost, gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import { oneVersusOne, oneVersusRest, outputCode, exhaustiveCode } from 'aifn-methods/learning/reductions'
import {
  Contours,
  ControlGroup,
  Figure,
  Handle,
  NumberSelector,
  Plot,
  Points,
  Raster,
  Readout,
  Select,
  Switch,
  formatNumber,
  useAxis,
  type Vec2,
} from 'aifn-render'

type ClassifierEntry = {
  label: string
  family: 'generative' | 'discriminative'
  fit: (x: Tensor, y: Tensor) => { decide(x: Tensor): Tensor }
}

const CLASSIFIERS: Record<string, ClassifierEntry> = {
  // Generative models
  'naive-bayes': {
    label: 'Gaussian Naive Bayes (Generative)',
    family: 'generative',
    fit: (x, y) => gaussianNaiveBayes().fit(dataset(x, y)),
  },
  lda: {
    label: 'Linear Discriminant Analysis (LDA, Generative)',
    family: 'generative',
    fit: (x, y) => linearDiscriminant().fit(dataset(x, y)),
  },
  qda: {
    label: 'Quadratic Discriminant Analysis (QDA, Generative)',
    family: 'generative',
    fit: (x, y) => quadraticDiscriminant().fit(dataset(x, y)),
  },
  // Discriminative models
  knn: {
    label: 'k-Nearest Neighbours (k = 7, Discriminative)',
    family: 'discriminative',
    fit: (x, y) => kNearestNeighbours({ k: 7 }).fit(dataset(x, y)),
  },
  logistic: {
    label: 'Multinomial Logistic Regression (Discriminative)',
    family: 'discriminative',
    fit: (x, y) => oneVersusRest(logisticRegression({ l2: 0.1 })).fit(dataset(x, y)),
  },
  perceptron: {
    label: 'Perceptron (OvR, Discriminative)',
    family: 'discriminative',
    fit: (x, y) => oneVersusRest(perceptron({ epochs: 50 })).fit(dataset(x, y)),
  },
  svm: {
    label: 'Support Vector Machine (RBF kernel, SMO)',
    family: 'discriminative',
    fit: (x, y) => oneVersusOne(supportVectorMachine({ C: 1, kernel: rbf({ lengthscale: 0.7 }) })).fit(dataset(x, y)),
  },
  'crammer-singer': {
    label: 'Crammer–Singer Multiclass Linear SVM',
    family: 'discriminative',
    fit: (x, y) => crammerSinger({ C: 1, maxSteps: 300 }).fit(dataset(x, y)),
  },
  tree: {
    label: 'Decision Tree (max depth = 5)',
    family: 'discriminative',
    fit: (x, y) => decisionTree({ maxDepth: 5 }).fit(dataset(x, y)),
  },
  forest: {
    label: 'Random Forest (50 trees)',
    family: 'discriminative',
    fit: (x, y) => randomForest({ trees: 50, maxFeatures: 1 }).fit(dataset(x, y), { stream: stream('forest') }),
  },
  adaboost: {
    label: 'AdaBoost (100 stumps)',
    family: 'discriminative',
    fit: (x, y) => adaBoost({ rounds: 100 }).fit(dataset(x, y)),
  },
  boosting: {
    label: 'Gradient Boosting (logistic loss, 60 stages)',
    family: 'discriminative',
    fit: (x, y) =>
      gradientBoosting({ loss: 'logistic', stages: 60, learningRate: 0.2, tree: { maxDepth: 2 } }).fit(dataset(x, y)),
  },
  ecoc: {
    label: 'Error-Correcting Output Codes (ECOC, exhaustive)',
    family: 'discriminative',
    fit: (x, y) => {
      const maxK = Math.max(2, ...toFlat(y)) + 1
      return outputCode(logisticRegression({ l2: 0.1 }), exhaustiveCode(maxK), { decoding: 'loss' }).fit(dataset(x, y))
    },
  },
}

const DATASETS = [
  { value: 'moons', label: 'Two interleaved moons (nonlinear boundary)' },
  { value: 'circles', label: 'Concentric circles (non-convex separation)' },
  { value: 'blobs', label: 'Three Gaussian blobs (multiclass)' },
  { value: 'xor', label: 'XOR pattern (four alternating quadrants)' },
]

export function DecisionRegionsGallery() {
  const [dataChoice, setDataChoice] = useState('moons')
  const [modelChoice, setModelChoice] = useState('naive-bayes')
  const [sampleSize, setSampleSize] = useState(160)
  const [noise, setNoise] = useState(0.18)
  const [showBoundary, setShowBoundary] = useState(true)
  const [query, setQuery] = useState<Vec2>([0.5, 0.25])

  // Generate dataset
  const data = useMemo(() => {
    const s = stream(`data/${dataChoice}/${sampleSize}/${noise}`)
    let d
    if (dataChoice === 'moons') {
      d = moons(s, { n: sampleSize, noise })
    } else if (dataChoice === 'circles') {
      d = circles(s, { n: sampleSize, noise, factor: 0.45 })
    } else if (dataChoice === 'blobs') {
      d = blobs(s, {
        n: sampleSize,
        centers: [
          [-1.8, -1],
          [1.8, -1],
          [0, 1.8],
        ],
        sd: noise * 3.5,
      })
    } else {
      d = xor(s, { n: sampleSize, sd: noise * 1.8 })
    }
    return {
      x: d.x,
      y: d.y!,
      names: d.meta?.labelNames ?? ['Class 0', 'Class 1', 'Class 2'],
    }
  }, [dataChoice, sampleSize, noise])

  // Fit selected classifier
  const currentModelConfig = CLASSIFIERS[modelChoice]
  const fittedModel = useMemo(() => {
    try {
      return currentModelConfig.fit(data.x, data.y)
    } catch {
      return CLASSIFIERS.knn.fit(data.x, data.y)
    }
  }, [currentModelConfig, data])

  // Compute training accuracy
  const accuracy = useMemo(() => {
    const pred = toFlat(fittedModel.decide(data.x))
    const truth = toFlat(data.y)
    const matches = pred.filter((c, i) => c === truth[i]).length
    return matches / truth.length
  }, [fittedModel, data])

  // Evaluate query point
  const queryResult = useMemo(() => {
    const qTensor = fromData(Float64Array.from(query), [1, 2])
    const label = toFlat(fittedModel.decide(qTensor))[0]
    let probs: number[] | null = null
    if (hasPredictive(fittedModel)) {
      const predDist = (fittedModel as unknown as { predictive(x: Tensor): Distribution }).predictive(qTensor)
      probs = toRows(classProbabilities(predDist))[0]
    }
    return { label, probs }
  }, [fittedModel, query])

  // Compute bounding box and grid
  const { xr, yr, grid, pts, labels } = useMemo(() => {
    const flat = toFlat(data.x)
    const x0 = flat.filter((_, i) => i % 2 === 0)
    const x1 = flat.filter((_, i) => i % 2 === 1)
    const minX = Math.min(...x0)
    const maxX = Math.max(...x0)
    const minY = Math.min(...x1)
    const maxY = Math.max(...x1)
    const padX = 0.25 * (maxX - minX || 1)
    const padY = 0.25 * (maxY - minY || 1)
    const xr: [number, number] = [minX - padX, maxX + padX]
    const yr: [number, number] = [minY - padY, maxY + padY]

    const g = grid2d(xr, yr, 70)
    const decided = toFlat(fittedModel.decide(g.points))
    const [ny, nx] = g.shape
    const z = Array.from({ length: ny }, (_, i) => decided.slice(i * nx, (i + 1) * nx))

    return {
      xr,
      yr,
      grid: { x: toFlat(g.x), y: toFlat(g.y), z },
      pts: { x: x0, y: x1 },
      labels: toFlat(data.y),
    }
  }, [data, fittedModel])

  // Dynamic overlay: highlight k-NN neighbours or SVM support vectors. The indices are computed inside try/catch
  // (a model without the expected fields draws no overlay); the marks are built from them outside it.
  const highlight = useMemo((): { name: string; idx: number[] } | null => {
    if (modelChoice === 'knn') {
      try {
        const qTensor = fromData(Float64Array.from(query), [1, 2])
        const knnModel = fittedModel as unknown as { neighbours(q: Tensor): { index: Tensor } }
        if (typeof knnModel.neighbours === 'function') {
          return { name: '7 nearest neighbours', idx: toFlat(knnModel.neighbours(qTensor).index) }
        }
      } catch {
        // ignore
      }
    }
    if (modelChoice === 'svm') {
      try {
        const m = fittedModel as unknown as { models: Array<{ supportVectors: Tensor }>; code: Tensor }
        if (m.models && m.code) {
          const sv = new Set<number>()
          const code = toRows(m.code)
          m.models.forEach((svm, l) => {
            const members = toFlat(data.y).flatMap((c, i) => (code[c][l] !== 0 ? [i] : []))
            for (const t of toFlat(svm.supportVectors)) sv.add(members[t])
          })
          return { name: 'support vectors', idx: [...sv] }
        }
      } catch {
        // ignore
      }
    }
    return null
  }, [modelChoice, fittedModel, query, data])
  const overlay = useMemo(() => {
    if (!highlight) return null
    const rows = toRows(data.x)
    return (
      <Points
        name={highlight.name}
        x={highlight.idx.map((i) => rows[i][0])}
        y={highlight.idx.map((i) => rows[i][1])}
        emphasis
      />
    )
  }, [highlight, data])

  const xAxis = useAxis({ label: 'x₀', range: xr })
  const yAxis = useAxis({ label: 'x₁', range: yr, equal: xAxis })

  const uniqueClasses = Array.from(new Set(labels)).sort((a, b) => a - b)
  const classNames = uniqueClasses.map((c) => data.names[c] ?? `Class ${c}`)

  return (
    <Figure
      title="Decision regions: Generative vs Discriminative Classifiers"
      purpose="Direct visual comparison of how generative models (Naive Bayes, LDA, QDA) and discriminative models (k-NN, Logistic, SVM, Decision Trees, Ensembles) carve out decision regions across complex geometries."
      defaultSize="L"
      controls={
        <>
          <ControlGroup title="1 · Data Generator">
            <Select label="Synthetic Dataset" value={dataChoice} onChange={setDataChoice} options={DATASETS} />
            <NumberSelector
              label="Sample size n"
              value={sampleSize}
              onChange={setSampleSize}
              min={60}
              max={300}
              step={20}
              spacing="lin"
              points={20}
            />
            <NumberSelector
              label="Noise sd"
              value={noise}
              onChange={setNoise}
              min={0.05}
              max={0.5}
              step={0.05}
              spacing="lin"
              points={0.05}
            />
          </ControlGroup>
          <ControlGroup title="2 · Classifier Family">
            <Select
              label="Classification Model"
              value={modelChoice}
              onChange={setModelChoice}
              options={Object.entries(CLASSIFIERS).map(([value, { label }]) => ({ value, label }))}
            />
            <Switch label="Show Decision Boundaries" checked={showBoundary} onChange={setShowBoundary} />
          </ControlGroup>
        </>
      }
      readouts={{
        'Model Family': (
          <>
            <Readout label="family" value={currentModelConfig.family.toUpperCase()} />
            <Readout label="training accuracy" value={formatNumber(accuracy)} />
          </>
        ),
        'Probe Readout': (
          <>
            <Readout label="query decision" value={data.names[queryResult.label] ?? `Class ${queryResult.label}`} />
            {queryResult.probs && (
              <Readout label="posteriors P(y | q)" value={queryResult.probs.map((p) => p.toFixed(2)).join(' : ')} />
            )}
          </>
        ),
      }}
      caption="Drag the query handle around the plane to inspect live class assignment and posteriors. Generative models (Naive Bayes, LDA) project rigid ellipsoidal or linear Gaussian distributions that struggle with interleaved moons or XOR, whereas flexible discriminative models (RBF SVM, k-NN, Random Forest) conform tightly to nonlinear boundaries."
    >
      <Plot x={xAxis} y={yAxis}>
        <Raster
          x={grid.x}
          y={grid.y}
          z={grid.z}
          scale="diverging"
          range={[0, Math.max(1, ...uniqueClasses)]}
          valueLabel="decided class"
        />
        {showBoundary && <Contours x={grid.x} y={grid.y} z={grid.z} levels={[0.5, 1.5]} />}
        <Points name="training points" x={pts.x} y={pts.y} group={labels} groupNames={classNames} />
        {overlay}
        <Handle kind="point" at={query} onDrag={setQuery} label="query" />
      </Plot>
    </Figure>
  )
}
