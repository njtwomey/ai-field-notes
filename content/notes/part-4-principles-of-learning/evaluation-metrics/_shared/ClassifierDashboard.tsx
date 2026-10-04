import { useMemo, useState } from 'react'
import { recipe, type ClassificationTruth } from 'aifn-methods/data'
import { gpClassifier } from 'aifn-methods/learning/gaussian-processes'
import {
  gaussianNaiveBayes,
  linearDiscriminant,
  quadraticDiscriminant,
} from 'aifn-methods/learning/generative-classifiers'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { supportVectorMachine } from 'aifn-methods/learning/kernel-methods'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import { polynomialFeatures, splineFeatures, standardScaler } from 'aifn-methods/learning/preprocessing'
import { decisionTree } from 'aifn-methods/learning/trees-and-ensembles'
import { randomForest } from 'aifn-methods/learning/trees-and-ensembles/bagging'
import { gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import { type Params } from 'aifn/foundation/pytree'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, toRows, unwrap, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { pipeline } from 'aifn/learning/compose'
import { classProbabilities, dataset, type Distribution, type Supervised } from 'aifn/learning/estimators'
import { rbf } from 'aifn/learning/kernels'
import { binaryCrossEntropyWithLogits } from 'aifn/learning/losses'
import {
  binaryRates,
  brierScore,
  countsAtThreshold,
  logLoss,
  precisionRecallCurve,
  rocCurve,
} from 'aifn/learning/metrics'
import { Mlp } from 'aifn/nn/layers'
import { xavierUniform } from 'aifn/nn/init'
import { trainingLoop } from 'aifn/nn/training'
import { contourLines, grid2d } from 'aifn/numerics/geometry'
import { logit, sigmoid } from 'aifn/numerics/special'
import { adamRule } from 'aifn/optim/first-order'
import { histogram } from 'aifn/probability/stats'
import {
  Bars,
  ControlGroup,
  Curve,
  Dashboard,
  DashboardCell,
  DashboardRow,
  Figure,
  Handle,
  NumberSelector,
  Plot,
  Points,
  Raster,
  Readout,
  Select,
  useAxis,
} from 'aifn-render'

// ── Datasets ──

type DatasetKey = 'moons' | 'blobs' | 'circles' | 'xor' | 'spirals' | 'anisotropic' | 'checkerboard'

const DATASETS: { value: DatasetKey; label: string }[] = [
  { value: 'moons', label: 'Two Interlocking Moons' },
  { value: 'blobs', label: 'Two Gaussian Blobs' },
  { value: 'circles', label: 'Concentric Circles' },
  { value: 'xor', label: 'XOR (Four Blobs)' },
  { value: 'spirals', label: 'Two Intertwined Spirals' },
  { value: 'anisotropic', label: 'Anisotropic Gaussians' },
  { value: 'checkerboard', label: 'Checkerboard (3x3)' },
]

// ── Classifiers ──

type ModelKey =
  | 'logistic'
  | 'polynomial'
  | 'spline'
  | 'knn'
  | 'lda'
  | 'qda'
  | 'bayes'
  | 'svm'
  | 'tree'
  | 'forest'
  | 'boosting'
  | 'gp'
  | 'mlp'

const MODELS: { value: ModelKey; label: string }[] = [
  { value: 'logistic', label: 'Logistic Regression' },
  { value: 'polynomial', label: 'Polynomial Logistic (Degree 3)' },
  { value: 'spline', label: 'Spline Logistic (Additive)' },
  { value: 'knn', label: 'k-Nearest Neighbours' },
  { value: 'lda', label: 'Linear Discriminant (LDA)' },
  { value: 'qda', label: 'Quadratic Discriminant (QDA)' },
  { value: 'bayes', label: 'Gaussian Naive Bayes' },
  { value: 'svm', label: 'Support Vector Machine (RBF)' },
  { value: 'tree', label: 'Decision Tree' },
  { value: 'forest', label: 'Random Forest' },
  { value: 'boosting', label: 'Gradient Tree Boosting' },
  { value: 'gp', label: 'Gaussian Process Classifier (RBF)' },
  { value: 'mlp', label: 'Multilayer Perceptron (2-layer Tanh)' },
]

type Scorer = {
  kind: 'log-odds' | 'margin'
  score: (x: Tensor) => Float64Array
  probability?: (x: Tensor) => Float64Array
}

const EPS = 1e-3

function fromProbability(p: (x: Tensor) => Float64Array): Scorer {
  return {
    kind: 'log-odds',
    probability: p,
    score: (x) => p(x).map((v) => logit(Math.min(1 - EPS, Math.max(EPS, v)))),
  }
}

const predictiveScorer = (m: { predictive(x: Tensor): Distribution }) =>
  fromProbability((x) => Float64Array.from(toRows(classProbabilities(m.predictive(x))), (r) => r[1]))

type Fit = Supervised<Tensor, Tensor>

function nearest(x: ArrayLike<number>, y: ArrayLike<number>, p: readonly [number, number]): [number, number] {
  let best = 0
  let distance = Infinity
  for (let i = 0; i < x.length; i++) {
    const d = (x[i] - p[0]) ** 2 + (y[i] - p[1]) ** 2
    if (d < distance) {
      best = i
      distance = d
    }
  }
  return [best, distance]
}

function polylines(lines: Tensor[]): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (const line of lines) {
    for (const [a, b] of toRows(line)) {
      x.push(a)
      y.push(b)
    }
    x.push(NaN)
    y.push(NaN)
  }
  return { x, y }
}

const fmt = (v: number | null | undefined, digits = 3) =>
  v !== null && v !== undefined && Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '—'

const GRID = 60
const BINS = 32

export function ClassifierDashboard() {
  // ── 1. Data Selection ──
  const [datasetKey, setDatasetKey] = useState<DatasetKey>('moons')
  const [pointCount, setPointCount] = useState(240)
  const [prevalence, setPrevalence] = useState(0.3)
  const [noise, setNoise] = useState(0.25)

  // ── 2. Model Selection & Hyperparameters ──
  const [modelKey, setModelKey] = useState<ModelKey>('logistic')
  const [knnNeighbors, setKnnNeighbors] = useState(15)
  const [treeDepth, setTreeDepth] = useState(4)
  const [forestTrees, setForestTrees] = useState(40)
  const [svmC, setSvmC] = useState(1.0)
  const [svmGamma, setSvmGamma] = useState(1.4)
  const [mlpWidth, setMlpWidth] = useState(12)

  // ── 3. Decision Threshold ──
  const [threshold, setThreshold] = useState<number>(0)

  // ── Generate Data ──
  const datasetObj = useMemo(() => {
    let r: any
    if (datasetKey === 'moons') {
      r = { base: 'moons', knobs: { noise, spacing: 'random' } }
    } else if (datasetKey === 'blobs') {
      r = { base: 'blobs', knobs: { centers: 2, sd: 1, layout: 'polygon', separation: 2.2 } }
    } else if (datasetKey === 'circles') {
      r = { base: 'circles', knobs: { noise, factor: 0.5 } }
    } else if (datasetKey === 'xor') {
      r = { base: 'xor', knobs: { kind: 'gaussian', sd: noise + 0.2 } }
    } else if (datasetKey === 'spirals') {
      r = { base: 'spirals', knobs: { noise: noise * 0.4, arms: 2, turns: 1 } }
    } else if (datasetKey === 'anisotropic') {
      r = {
        base: 'blobs',
        knobs: { centers: 2, sd: 1, layout: 'polygon', separation: 2.5 },
        modifiers: [{ op: 'withTransform', params: { rotation: -Math.PI / 4, shear: 1, stretch: 0.6 } }],
      }
    } else {
      r = { base: 'checkerboard', knobs: { tiles: 3 } }
    }
    const d = recipe({ ...r, seed: 1, knobs: { ...r.knobs, n: pointCount, prevalence } })
    const truth = d.meta.truth?.task === 'classification' ? (d.meta.truth as ClassificationTruth) : null
    return { x: d.x, y: d.y!, truth }
  }, [datasetKey, pointCount, prevalence, noise])

  const yFlat = useMemo(() => Array.from(toFlat(datasetObj.y)), [datasetObj])
  const n = yFlat.length

  // ── Train / Fit Model ──
  const scorer = useMemo((): Scorer => {
    const d: Fit = dataset(datasetObj.x, datasetObj.y)
    switch (modelKey) {
      case 'logistic':
        return predictiveScorer(logisticRegression({ l2: 1.0 }).fit(d))
      case 'polynomial':
        return predictiveScorer(
          pipeline(
            standardScaler(),
            polynomialFeatures({ degree: 3, includeBias: false }),
            logisticRegression({ l2: 0.1 }),
          ).fit(d),
        )
      case 'spline':
        return predictiveScorer(
          pipeline(
            splineFeatures({ knots: 6, degree: 3, extrapolation: 'continue', includeBias: false }),
            logisticRegression({ l2: 0.1 }),
          ).fit(d),
        )
      case 'knn':
        return predictiveScorer(kNearestNeighbours({ k: Math.min(knnNeighbors, n) }).fit(d))
      case 'lda':
        return predictiveScorer(linearDiscriminant().fit(d))
      case 'qda':
        return predictiveScorer(quadraticDiscriminant({ regularisation: 0.01 }).fit(d))
      case 'bayes':
        return predictiveScorer(gaussianNaiveBayes().fit(d))
      case 'svm': {
        const kernel = rbf({ lengthscale: 1 / Math.sqrt(2 * svmGamma) })
        const m = supportVectorMachine({ C: svmC, kernel }).fit(d)
        return { kind: 'margin', score: (x) => Float64Array.from(toFlat(m.score(x))) }
      }
      case 'tree':
        return predictiveScorer(decisionTree({ maxDepth: treeDepth }).fit(d))
      case 'forest':
        return predictiveScorer(
          randomForest({ trees: forestTrees, maxDepth: treeDepth, maxFeatures: 1 }).fit(d, {
            stream: stream('dash/forest'),
          }),
        )
      case 'boosting': {
        const m = gradientBoosting({ loss: 'logistic', stages: 60, learningRate: 0.2, tree: { maxDepth: 2 } }).fit(d)
        return predictiveScorer({ predictive: (x) => m.predictive!(x) })
      }
      case 'gp':
        return predictiveScorer(gpClassifier({ kernel: rbf({ lengthscale: 0.7, variance: 4 }) }).fit(d))
      case 'mlp': {
        const scaler = standardScaler().fit({ x: d.x })
        const net = Mlp([2, mlpWidth, mlpWidth, 1], { activation: 'tanh', init: xavierUniform() })
        const yMat = fromData(Float64Array.from(toFlat(d.y)), [d.y.shape[0], 1])
        const run = trace(
          trainingLoop({
            loss: (w: Params[], b: { x: Tensor; y: Tensor }) => binaryCrossEntropyWithLogits(net.apply(w, b.x), b.y),
            data: { x: scaler.transform(d.x), y: yMat },
            optimizer: adamRule({ stepSize: 0.03, weightDecay: 1e-4, decoupled: true }),
          }),
          { params: net.init(stream('dash/mlp')) },
          250,
          { every: 250 },
        )
        const w = run.steps[run.steps.length - 1].params
        return fromProbability((x) =>
          Float64Array.from(toFlat(unwrap(sigmoid(net.apply(w, scaler.transform(x)))) as Tensor)),
        )
      }
    }
  }, [modelKey, datasetObj, knnNeighbors, n, svmGamma, svmC, treeDepth, forestTrees, mlpWidth])

  // ── Scores & Plane Evaluation ──
  const scores = useMemo(() => scorer.score(datasetObj.x), [scorer, datasetObj])

  const plane = useMemo(() => {
    const rows = toRows(datasetObj.x)
    const span = (a: number): [number, number] => {
      const v = rows.map((r) => r[a])
      const [lo, hi] = [Math.min(...v), Math.max(...v)]
      const pad = 0.08 * (hi - lo || 1)
      return [lo - pad, hi + pad]
    }
    const g = grid2d(span(0), span(1), GRID)
    const gy = toFlat(g.y)
    const toGrid = (s: ArrayLike<number>) => gy.map((_, i) => Array.from({ length: GRID }, (_, j) => s[i * GRID + j]))
    return { g, gx: toFlat(g.x), gy, rows, toGrid }
  }, [datasetObj])

  const field = useMemo(() => plane.toGrid(scorer.score(plane.g.points)), [scorer, plane])

  const bayes = useMemo(() => {
    const truth = datasetObj.truth
    if (!truth) return null
    const onGrid = plane.toGrid(toFlat(truth.score(plane.g.points)))
    const truthScores = Float64Array.from(toFlat(truth.score(datasetObj.x)))
    return { onGrid, scores: truthScores, error: truth.bayesError }
  }, [datasetObj, plane])

  // Score dynamic range
  const scoreRange = useMemo((): [number, number] => {
    let m = 0
    for (const v of scores) m = Math.max(m, Math.abs(v))
    for (const row of field) for (const v of row) m = Math.max(m, Math.abs(v))
    const top = Math.ceil(m) || 1
    return [-top, top]
  }, [scores, field])

  // ── ROC, PR Curves and Operating Metrics ──
  const roc = useMemo(() => rocCurve(yFlat, scores), [yFlat, scores])
  const pr = useMemo(() => precisionRecallCurve(yFlat, scores), [yFlat, scores])

  const bayesCurves = useMemo(() => {
    if (!bayes) return null
    const r = rocCurve(yFlat, bayes.scores)
    return {
      auroc: r.area,
      roc: { x: toFlat(r.x), y: toFlat(r.y) },
      pr: {
        x: toFlat(precisionRecallCurve(yFlat, bayes.scores).x),
        y: toFlat(precisionRecallCurve(yFlat, bayes.scores).y),
      },
    }
  }, [bayes, yFlat])

  const counts = countsAtThreshold(yFlat, scores, threshold)
  const rates = binaryRates(counts)

  const probabilistic = useMemo(() => {
    if (!scorer.probability) return null
    const p = scorer.probability(datasetObj.x)
    return { logLoss: logLoss(yFlat, p), brier: brierScore(yFlat, p) }
  }, [scorer, datasetObj, yFlat])

  // Decision boundary lines at threshold
  const boundary = useMemo(
    () => polylines(contourLines(plane.gx, plane.gy, field, threshold)),
    [plane, field, threshold],
  )
  const bayesBoundary = useMemo(
    () => (bayes ? polylines(contourLines(plane.gx, plane.gy, bayes.onGrid, 0)) : null),
    [plane, bayes],
  )

  // Histograms
  const histograms = useMemo(() => {
    const bars = (c: 0 | 1) => {
      const h = histogram(
        Array.from(scores).filter((_, i) => yFlat[i] === c),
        { bins: BINS, range: scoreRange },
      )
      const edges = toFlat(h.edges)
      return {
        x: edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2),
        counts: toFlat(h.counts),
      }
    }
    return [bars(0), bars(1)]
  }, [scores, yFlat, scoreRange])

  // Synchronise threshold drag on ROC or PR
  const fromCurve = (i: number, thresholds: Tensor) => {
    const v = toFlat(thresholds)[i]
    if (Number.isFinite(v)) setThreshold(v)
  }

  const px = useMemo(() => plane.rows.map((r) => r[0]), [plane])
  const py = useMemo(() => plane.rows.map((r) => r[1]), [plane])

  // Axes
  const fx1 = useAxis({ label: 'x₁', nice: false })
  const fx2 = useAxis({ label: 'x₂', equal: fx1, nice: false })
  const sx = useAxis({
    label: scorer.kind === 'log-odds' ? 'Log-Odds Score' : 'SVM Margin Score',
    range: scoreRange,
    nice: false,
  })
  const sy = useAxis({ label: 'Count', nice: true })
  const rocX = useAxis({ label: 'False Positive Rate (FPR)', range: [0, 1], nice: false })
  const rocY = useAxis({ label: 'True Positive Rate (TPR / Recall)', range: [0, 1], nice: false })
  const prX = useAxis({ label: 'Recall (TPR)', range: [0, 1], nice: false })
  const prY = useAxis({ label: 'Precision (PPV)', range: [0, 1], nice: false })

  const scoreName = scorer.kind === 'log-odds' ? 'log-odds' : 'margin'

  return (
    <Figure
      title="Classifier Evaluation Dashboard: Decisions, Scores, and Curves"
      purpose="Explore how a classifier's decision threshold simultaneously shapes the boundary in feature space, divides the score distributions, sets the confusion matrix, and determines the operating point on both the ROC and Precision-Recall curves."
      defaultSize="XL"
      controls={
        <div className="flex flex-col gap-3">
          <ControlGroup title="1 · Data Generation">
            <Select
              label="Synthetic Dataset"
              value={datasetKey}
              onChange={(v) => setDatasetKey(v as DatasetKey)}
              options={DATASETS}
            />
            <NumberSelector
              label="Cases n"
              value={pointCount}
              onChange={setPointCount}
              min={60}
              max={600}
              step={20}
              suggestions={[120, 240, 400]}
            />
            <NumberSelector
              label="Prevalence π (Class 1)"
              value={prevalence}
              onChange={setPrevalence}
              min={0.05}
              max={0.9}
              step={0.05}
              suggestions={[0.1, 0.3, 0.5]}
            />
            <NumberSelector
              label="Noise Standard Dev"
              value={noise}
              onChange={setNoise}
              min={0.05}
              max={0.6}
              step={0.05}
              suggestions={[0.1, 0.25, 0.4]}
            />
          </ControlGroup>

          <ControlGroup title="2 · Classifier & Architecture">
            <Select
              label="Classifier Algorithm"
              value={modelKey}
              onChange={(v) => setModelKey(v as ModelKey)}
              options={MODELS}
            />
            {modelKey === 'knn' && (
              <NumberSelector
                label="Neighbours k"
                value={knnNeighbors}
                onChange={setKnnNeighbors}
                min={1}
                max={51}
                step={2}
                suggestions={[5, 15, 25]}
              />
            )}
            {(modelKey === 'tree' || modelKey === 'forest') && (
              <NumberSelector
                label="Tree Max Depth"
                value={treeDepth}
                onChange={setTreeDepth}
                min={1}
                max={10}
                step={1}
                suggestions={[2, 4, 6]}
              />
            )}
            {modelKey === 'forest' && (
              <NumberSelector
                label="Ensemble Trees"
                value={forestTrees}
                onChange={setForestTrees}
                min={5}
                max={100}
                step={5}
                suggestions={[20, 40, 80]}
              />
            )}
            {modelKey === 'svm' && (
              <>
                <NumberSelector
                  label="SVM Penalty C"
                  value={svmC}
                  onChange={setSvmC}
                  min={0.1}
                  max={20}
                  step={0.5}
                  suggestions={[0.5, 1, 5]}
                />
                <NumberSelector
                  label="RBF Gamma γ"
                  value={svmGamma}
                  onChange={setSvmGamma}
                  min={0.1}
                  max={10}
                  step={0.2}
                  suggestions={[0.5, 1.4, 3]}
                />
              </>
            )}
            {modelKey === 'mlp' && (
              <NumberSelector
                label="Hidden Layer Units"
                value={mlpWidth}
                onChange={setMlpWidth}
                min={4}
                max={32}
                step={2}
                suggestions={[8, 12, 20]}
              />
            )}
          </ControlGroup>

          <ControlGroup title="3 · Decision Threshold">
            <NumberSelector
              label={`Threshold t on ${scoreName}`}
              value={threshold}
              onChange={setThreshold}
              min={scoreRange[0]}
              max={scoreRange[1]}
              step={0.05}
              suggestions={[-1, 0, 1]}
            />
          </ControlGroup>
        </div>
      }
      readouts={{
        [`Threshold Metrics at t = ${fmt(threshold)} (P = ${scorer.kind === 'log-odds' ? fmt(sigmoid(threshold)) : '—'})`]:
          (
            <>
              <Readout label="Accuracy" value={fmt(rates.accuracy)} />
              <Readout label="Balanced Accuracy" value={fmt(rates.balancedAccuracy)} />
              <Readout label="Precision (PPV)" value={fmt(rates.precision)} />
              <Readout label="Recall (TPR)" value={fmt(rates.recall)} />
              <Readout label="Specificity (TNR)" value={fmt(rates.specificity)} />
              <Readout label="F₁ Score" value={fmt(rates.f1)} />
              <Readout label="MCC" value={fmt(rates.matthewsCorrelation)} />
              <Readout label="Cohen's κ" value={fmt(rates.cohensKappa)} />
            </>
          ),
        'Ranking & Discrimination Metrics (all t)': (
          <>
            <Readout label="AUROC" value={fmt(roc.area)} />
            <Readout label="Average Precision (AP)" value={fmt(pr.area)} />
            {bayesCurves && <Readout label="Bayes-Optimal AUROC" value={fmt(bayesCurves.auroc)} />}
          </>
        ),
        'Probabilistic Calibration Metrics': (
          <>
            {probabilistic ? (
              <>
                <Readout label="Log Loss (Cross-Entropy)" value={fmt(probabilistic.logLoss)} />
                <Readout label="Brier Score" value={fmt(probabilistic.brier)} />
              </>
            ) : (
              <Readout label="Probabilities" value="N/A (SVM provides geometric margins)" />
            )}
            {bayes && <Readout label="Bayes Error Rate" value={fmt(bayes.error)} />}
          </>
        ),
      }}
      caption="Interactive evaluation dashboard comparing model behavior across feature space, score distributions, ROC, PR, and the confusion matrix. Drag threshold handle on the score distribution or operating points on the ROC/PR curves to watch the decision boundary and confusion matrix update in unison."
    >
      <Dashboard>
        <DashboardRow ratio={1.15} minHeight={300}>
          {/* Panel 1: Feature Space Decision Boundary */}
          <DashboardCell ratio={1.2}>
            <Plot x={fx1} y={fx2} title="1 · Decision Boundary in Feature Space">
              <Raster
                x={plane.gx}
                y={plane.gy}
                z={field}
                scale="diverging"
                range={scoreRange}
                valueLabel={scoreName}
                fillOpacity={0.4}
              />
              {bayesBoundary && (
                <Curve name="Bayes Boundary" x={bayesBoundary.x} y={bayesBoundary.y} muted dashed width={1.5} />
              )}
              <Curve name="Decision Boundary" x={boundary.x} y={boundary.y} slot={6} width={2.5} />
              <Points
                name="Data Cases"
                x={px}
                y={py}
                group={yFlat}
                groupNames={['Class 0 (Negative)', 'Class 1 (Positive)']}
                size={7}
              />
            </Plot>
          </DashboardCell>

          {/* Panel 2: Score Distributions */}
          <DashboardCell ratio={1.0}>
            <Plot x={sx} y={sy} title="2 · Class Score Distributions & Threshold">
              <Bars name="Class 0 Scores" x={histograms[0].x} y={histograms[0].counts} slot={0} opacity={0.6} />
              <Bars name="Class 1 Scores" x={histograms[1].x} y={histograms[1].counts} slot={7} opacity={0.6} />
              <Curve
                name="Threshold t"
                x={[threshold, threshold]}
                y={[0, Math.max(...histograms[0].counts, ...histograms[1].counts, 1)]}
                slot={6}
                width={2}
                dashed
              />
              <Handle kind="x" at={threshold} onDrag={setThreshold} label="t" />
            </Plot>
          </DashboardCell>
        </DashboardRow>

        <DashboardRow ratio={1.0} minHeight={280}>
          {/* Panel 3: ROC Curve */}
          <DashboardCell ratio={1.0}>
            <Plot x={rocX} y={rocY} title={`3 · ROC Curve (AUROC = ${fmt(roc.area)})`}>
              <Curve name="Chance Baseline" x={[0, 1]} y={[0, 1]} muted dashed width={1} />
              {bayesCurves && (
                <Curve name="Bayes-Optimal" x={bayesCurves.roc.x} y={bayesCurves.roc.y} muted dashed width={1.5} />
              )}
              <Curve name="ROC" x={toFlat(roc.x)} y={toFlat(roc.y)} slot={6} width={2.5} />
              <Points name="Operating Point" x={[rates.falsePositiveRate]} y={[rates.recall]} emphasis size={11} />
              <Handle
                kind="point"
                at={[rates.falsePositiveRate, rates.recall]}
                label="t"
                onDrag={(p) => fromCurve(nearest(toFlat(roc.x), toFlat(roc.y), p)[0], roc.thresholds)}
              />
            </Plot>
          </DashboardCell>

          {/* Panel 4: Precision-Recall Curve */}
          <DashboardCell ratio={1.0}>
            <Plot x={prX} y={prY} title={`4 · Precision-Recall Curve (AP = ${fmt(pr.area)})`}>
              <Curve name="Prevalence Baseline" x={[0, 1]} y={[prevalence, prevalence]} muted dashed width={1} />
              {bayesCurves && (
                <Curve name="Bayes-Optimal" x={bayesCurves.pr.x} y={bayesCurves.pr.y} muted dashed width={1.5} />
              )}
              <Curve name="PR" x={toFlat(pr.x)} y={toFlat(pr.y)} slot={6} width={2.5} />
              <Points
                name="Operating Point"
                x={[rates.recall]}
                y={[Number.isFinite(rates.precision) ? rates.precision : 1]}
                emphasis
                size={11}
              />
              <Handle
                kind="point"
                at={[rates.recall, Number.isFinite(rates.precision) ? rates.precision : 1]}
                label="t"
                onDrag={(p) => fromCurve(nearest(toFlat(pr.x), toFlat(pr.y), p)[0], pr.thresholds)}
              />
            </Plot>
          </DashboardCell>

          {/* Panel 5: Confusion Matrix Contingency Table */}
          <DashboardCell ratio={1.0}>
            <div className="flex h-full flex-col justify-center rounded-lg border border-border/70 bg-card p-3 shadow-2xs">
              <div className="mb-2 text-xs font-semibold tracking-wider text-muted-foreground uppercase">
                5 · Confusion Matrix (Counts & Rates)
              </div>
              <div className="grid grid-cols-3 gap-1.5 text-center text-xs">
                <div className="p-1"></div>
                <div className="rounded bg-muted/60 p-1 font-semibold">Pred Pos</div>
                <div className="rounded bg-muted/60 p-1 font-semibold">Pred Neg</div>

                <div className="flex items-center justify-end pr-1 font-semibold">Act Pos</div>
                <div className="flex flex-col items-center justify-center rounded border border-emerald-500/40 bg-emerald-500/10 p-2">
                  <span className="font-mono text-[10px] text-muted-foreground">TP</span>
                  <span className="font-mono text-base font-bold text-foreground">{counts.tp}</span>
                  <span className="text-[10px] text-muted-foreground">TPR: {fmt(rates.recall)}</span>
                </div>
                <div className="flex flex-col items-center justify-center rounded border border-destructive/30 bg-destructive/10 p-2">
                  <span className="font-mono text-[10px] text-muted-foreground">FN</span>
                  <span className="font-mono text-base font-bold text-foreground">{counts.fn}</span>
                  <span className="text-[10px] text-muted-foreground">FNR: {fmt(rates.falseNegativeRate)}</span>
                </div>

                <div className="flex items-center justify-end pr-1 font-semibold">Act Neg</div>
                <div className="flex flex-col items-center justify-center rounded border border-destructive/30 bg-destructive/10 p-2">
                  <span className="font-mono text-[10px] text-muted-foreground">FP</span>
                  <span className="font-mono text-base font-bold text-foreground">{counts.fp}</span>
                  <span className="text-[10px] text-muted-foreground">FPR: {fmt(rates.falsePositiveRate)}</span>
                </div>
                <div className="flex flex-col items-center justify-center rounded border border-emerald-500/40 bg-emerald-500/10 p-2">
                  <span className="font-mono text-[10px] text-muted-foreground">TN</span>
                  <span className="font-mono text-base font-bold text-foreground">{counts.tn}</span>
                  <span className="text-[10px] text-muted-foreground">TNR: {fmt(rates.specificity)}</span>
                </div>
              </div>
              <div className="mt-3 flex justify-between border-t border-border/50 pt-2 text-[11px] text-muted-foreground">
                <span>
                  Precision (PPV): <strong className="text-foreground">{fmt(rates.precision)}</strong>
                </span>
                <span>
                  Accuracy: <strong className="text-foreground">{fmt(rates.accuracy)}</strong>
                </span>
                <span>
                  F1: <strong className="text-foreground">{fmt(rates.f1)}</strong>
                </span>
              </div>
            </div>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
