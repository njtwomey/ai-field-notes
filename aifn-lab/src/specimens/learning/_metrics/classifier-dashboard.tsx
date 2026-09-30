import { useMemo, useState, type ReactNode } from 'react'
import { decisionTree } from 'aifn-applied/learning/trees-and-ensembles'
import {
  gaussianNaiveBayes,
  linearDiscriminant,
  quadraticDiscriminant,
} from 'aifn-applied/learning/generative-classifiers'
import { gradientBoosting } from 'aifn-applied/learning/trees-and-ensembles/boosting'
import { kNearestNeighbours } from 'aifn-applied/learning/neighbours'
import { randomForest } from 'aifn-applied/learning/trees-and-ensembles/bagging'
import { supportVectorMachine } from 'aifn-applied/learning/kernel-methods'
import { pipeline } from 'aifn/learning/compose'
import { recipe, type DatasetRecipe } from 'aifn-applied/data/synthetic'
import { type ClassificationTruth } from 'aifn-applied/data'
import { classProbabilities, dataset, type Distribution, type Supervised } from 'aifn/learning/estimators'
import { logisticRegression } from 'aifn-applied/learning/generalised/glm'
import { contourLines, grid2d } from 'aifn/numerics/geometry'
import { gpClassifier } from 'aifn-applied/learning/gaussian-processes'
import { rbf } from 'aifn/learning/kernels'
import { binaryCrossEntropyWithLogits } from 'aifn/learning/losses'
import {
  binaryRates,
  brierScore,
  confusionMatrix,
  countsAtThreshold,
  logLoss,
  precisionRecallCurve,
  rocCurve,
} from 'aifn/learning/metrics'
import { trainingLoop } from 'aifn/nn/training'
import { adamRule } from 'aifn/optim/first-order'
import { Mlp } from 'aifn/nn/layers'
import { xavierUniform } from 'aifn/nn/init'
import { type Params } from 'aifn/foundation/pytree'
import { polynomialFeatures, splineFeatures, standardScaler } from 'aifn-applied/learning/preprocessing'
import { stream } from 'aifn/foundation/random'
import { logit, sigmoid } from 'aifn/numerics/special'
import { histogram } from 'aifn/probability/stats'
import { fromData, toFlat, toRows, unwrap, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { chrome, seriesColor } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { defineVariants, Slider, slider, useVariants, VariantControls } from '@lab/controls'
import { ControlRow, Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { Heatmap, Readout, XYChart, type Handle, type HeatmapOverlay, type PlotPointer, type XYSeries } from '@lab/viz'
import { ContingencyTableView, CurveChart, formatValue, histogramBars } from '@lab/views'

// ── Data: aifn/datasets recipes, two classes, class 1 positive ─────────────────────────────────────────────────────

/** A dataset recipe per base; `n` and `prevalence` are shared. Only the data are controlled, never the training. */
const DATA = defineVariants(
  {
    moons: {
      label: 'two moons',
      params: { noise: slider(0, 0.5, 0.25, { label: 'noise sd' }) },
      f: (_: null, p): DatasetRecipe => ({ base: 'moons', noise: p.noise, options: { spacing: 'random' } }),
    },
    blobs: {
      label: 'two Gaussian blobs',
      params: { separation: slider(0, 5, 2, { label: 'separation (blob sds)' }) },
      f: (_: null, p): DatasetRecipe => ({
        base: 'blobs',
        noise: 1,
        separation: p.separation,
        options: { centers: 2 },
      }),
    },
    circles: {
      label: 'two circles',
      params: {
        noise: slider(0, 0.3, 0.12, { label: 'noise sd' }),
        factor: slider(0.1, 0.9, 0.5, { label: 'inner radius' }),
      },
      f: (_: null, p): DatasetRecipe => ({ base: 'circles', noise: p.noise, options: { factor: p.factor } }),
    },
    xor: {
      label: 'XOR (four blobs)',
      params: { noise: slider(0.1, 1, 0.45, { label: 'blob sd' }) },
      f: (_: null, p): DatasetRecipe => ({ base: 'xor', noise: p.noise }),
    },
    spirals: {
      label: 'two spirals',
      params: {
        noise: slider(0, 0.2, 0.05, { label: 'noise sd' }),
        turns: slider(0.5, 2, 1, { label: 'turns' }),
      },
      f: (_: null, p): DatasetRecipe => ({ base: 'spirals', noise: p.noise, options: { arms: 2, turns: p.turns } }),
    },
    anisotropic: {
      label: 'anisotropic Gaussians',
      params: { separation: slider(0, 5, 2.5, { label: 'separation (blob sds)' }) },
      f: (_: null, p): DatasetRecipe => ({
        base: 'blobs',
        noise: 1,
        separation: p.separation,
        options: { centers: 2 },
        transform: {
          matrix: [
            [0.6, -0.6],
            [-0.4, 0.8],
          ],
        },
      }),
    },
    checkerboard: {
      label: 'checkerboard',
      params: { tiles: slider(2, 6, 3, { step: 1, label: 'tiles per side' }) },
      f: (_: null, p): DatasetRecipe => ({ base: 'checkerboard', options: { tiles: p.tiles } }),
    },
  },
  {
    n: slider(60, 600, 240, { step: 10, label: 'cases n' }),
    prevalence: slider(0.05, 0.95, 0.3, { step: 0.05, label: 'prevalence π of class 1' }),
  },
)

// ── Models: fixed, characteristic settings; the score is the log-odds of class 1 (the SVM's: its margin) ──────────────

/** A fitted model's score for class 1 on the log-odds (or margin) scale, and P(y = 1 | x) where it has one. */
type Scorer = {
  kind: 'log-odds' | 'margin'
  score: (x: Tensor) => Float64Array
  probability?: (x: Tensor) => Float64Array
}

/** Probabilities are clipped to [ε, 1 − ε] before the logit, so that certain cases (k-NN, trees) stay on the chart. */
const EPS = 1e-3

/** A scorer from P(y = 1 | x): its log-odds as the score, and the probability itself for log loss and Brier. */
function fromProbability(p: (x: Tensor) => Float64Array): Scorer {
  return {
    kind: 'log-odds',
    probability: p,
    score: (x) => p(x).map((v) => logit(Math.min(1 - EPS, Math.max(EPS, v)))),
  }
}

/** P(y = 1 | x) from the `predictive` capability of aifn/estimators. */
const predictive = (m: { predictive(x: Tensor): Distribution }) =>
  fromProbability((x) => {
    // Class probabilities are an [N, K] tensor: column 1 is P(y = 1 | x).
    return Float64Array.from(toRows(classProbabilities(m.predictive(x))), (r) => r[1])
  })

type Fit = Supervised<Tensor, Tensor>

/** L2 strengths and weight decays span decades: their sliders run over log₁₀ of the value. */
const logSlider = (lo: number, hi: number, initial: number, label: string) =>
  slider(lo, hi, initial, { step: 0.1, label: `log₁₀ ${label}` })

/**
 * Each model with its own hyperparameters, at characteristic defaults. The training procedure (solver, iterations,
 * learning rate, seeds) is fixed: these figures illustrate, they do not validate.
 */
const MODELS = defineVariants({
  logistic: {
    label: 'logistic regression',
    params: { l2: logSlider(-3, 2, 0, 'λ (L2)') },
    f: (d: Fit, p): Scorer => predictive(logisticRegression({ l2: 10 ** p.l2 }).fit(d)),
  },
  polynomial: {
    label: 'polynomial logistic regression',
    params: { degree: slider(1, 6, 3, { step: 1, label: 'degree' }), l2: logSlider(-3, 2, -1, 'λ (L2)') },
    f: (d: Fit, p): Scorer =>
      predictive(
        pipeline(
          standardScaler(),
          polynomialFeatures({ degree: p.degree, includeBias: false }),
          logisticRegression({ l2: 10 ** p.l2 }),
        ).fit(d),
      ),
  },
  spline: {
    label: 'spline logistic regression (additive)',
    params: { knots: slider(3, 12, 6, { step: 1, label: 'knots per feature' }), l2: logSlider(-3, 2, -1, 'λ (L2)') },
    f: (d: Fit, p): Scorer =>
      predictive(
        pipeline(
          splineFeatures({ knots: p.knots, degree: 3, extrapolation: 'continue', includeBias: false }),
          logisticRegression({ l2: 10 ** p.l2 }),
        ).fit(d),
      ),
  },
  knn: {
    label: 'k-nearest neighbours',
    params: { k: slider(1, 51, 15, { step: 1, label: 'neighbours k' }) },
    f: (d: Fit, p): Scorer => predictive(kNearestNeighbours({ k: Math.min(p.k, d.x.shape[0]) }).fit(d)),
  },
  lda: {
    label: 'linear discriminant (LDA)',
    params: {},
    f: (d: Fit): Scorer => predictive(linearDiscriminant().fit(d)),
  },
  qda: {
    label: 'quadratic discriminant (QDA)',
    params: {},
    f: (d: Fit): Scorer => predictive(quadraticDiscriminant({ regularisation: 0.01 }).fit(d)),
  },
  bayes: { label: 'Gaussian naive Bayes', params: {}, f: (d: Fit): Scorer => predictive(gaussianNaiveBayes().fit(d)) },
  svm: {
    label: 'SVM, RBF kernel',
    params: { C: slider(0.1, 20, 1, { label: 'C' }), gamma: slider(0.1, 10, 1.4, { label: 'γ (RBF)' }) },
    f: (d: Fit, p): Scorer => {
      // k(x, x′) = exp(−γ‖x − x′‖²), i.e. lengthscale ℓ = 1/√(2γ).
      const kernel = rbf({ lengthscale: 1 / Math.sqrt(2 * p.gamma) })
      const m = supportVectorMachine({ C: p.C, kernel }).fit(d)
      return { kind: 'margin', score: (x) => Float64Array.from(toFlat(m.score(x))) }
    },
  },
  tree: {
    label: 'decision tree',
    params: { depth: slider(1, 12, 4, { step: 1, label: 'maximum depth' }) },
    f: (d: Fit, p): Scorer => predictive(decisionTree({ maxDepth: p.depth }).fit(d)),
  },
  forest: {
    label: 'random forest',
    params: {
      trees: slider(5, 100, 50, { step: 5, label: 'trees' }),
      depth: slider(1, 12, 6, { step: 1, label: 'maximum depth' }),
    },
    f: (d: Fit, p): Scorer =>
      predictive(
        randomForest({ trees: p.trees, maxDepth: p.depth, maxFeatures: 1 }).fit(d, {
          stream: stream('lab/metrics/forest'),
        }),
      ),
  },
  boosting: {
    label: 'gradient boosting',
    params: {
      trees: slider(5, 200, 60, { step: 5, label: 'trees' }),
      depth: slider(1, 4, 2, { step: 1, label: 'tree depth' }),
    },
    f: (d: Fit, p): Scorer => {
      const m = gradientBoosting({
        loss: 'logistic',
        stages: p.trees,
        learningRate: 0.2,
        tree: { maxDepth: p.depth },
      }).fit(d)
      return predictive({ predictive: (x) => m.predictive!(x) })
    },
  },
  gp: {
    label: 'GP classifier (Laplace, RBF)',
    params: { lengthscale: slider(0.1, 3, 0.6, { label: 'lengthscale ℓ' }) },
    f: (d: Fit, p): Scorer =>
      predictive(gpClassifier({ kernel: rbf({ lengthscale: p.lengthscale, variance: 4 }) }).fit(d)),
  },
  mlp: {
    label: 'MLP (two tanh layers)',
    params: {
      width: slider(2, 32, 12, { step: 1, label: 'units per layer' }),
      decay: logSlider(-5, -1, -4, 'weight decay'),
    },
    f: (d: Fit, p): Scorer => {
      // Standardised inputs, 300 full-batch Adam steps on the logistic loss; only the final parameters are kept.
      const scaler = standardScaler().fit({ x: d.x })
      const net = Mlp([2, p.width, p.width, 1], { activation: 'tanh', init: xavierUniform() })
      const y = fromData(Float64Array.from(toFlat(d.y)), [d.y.shape[0], 1])
      const run = trace(
        trainingLoop({
          loss: (w: Params[], b: { x: Tensor; y: Tensor }) => binaryCrossEntropyWithLogits(net.apply(w, b.x), b.y),
          data: { x: scaler.transform(d.x), y },
          optimizer: adamRule({ stepSize: 0.03, weightDecay: 10 ** p.decay, decoupled: true }),
        }),
        { params: net.init(stream('lab/metrics/mlp')) },
        300,
        { every: 300 },
      )
      const w = run.steps[run.steps.length - 1].params
      return fromProbability((x) =>
        Float64Array.from(toFlat(unwrap(sigmoid(net.apply(w, scaler.transform(x)))) as Tensor)),
      )
    },
  },
})

// ── The dashboard ─────────────────────────────────────────────────────────────────────────────────────────────────

const GRID = 64
const BINS = 36
/** The palette slots of the two classes on every panel: blue for class 0 (negative), red for class 1 (positive). */
const CLASS_SLOTS = [0, 7] as const
/** The slot of the model's curves, apart from the classes' colours. */
const CURVE_SLOT = 6

const fmt = (v: number) =>
  Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : Number.isNaN(v) ? '–' : v > 0 ? '∞' : '−∞'

/** The index of the point (x[i], y[i]) nearest to p, and its squared distance. */
function nearest(x: ArrayLike<number>, y: ArrayLike<number>, p: readonly [number, number]): [number, number] {
  let best = 0
  let distance = Infinity
  for (let i = 0; i < x.length; i++) {
    const d = (x[i] - p[0]) ** 2 + (y[i] - p[1]) ** 2
    if (d < distance) [best, distance] = [i, d]
  }
  return [best, distance]
}

/** A contour's polylines as one overlay line, broken by NaN (ECharts breaks a line at a missing value). */
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

export function ClassifierDashboardSpecimen() {
  const data = useVariants(DATA)
  const model = useVariants(MODELS)
  const { resolved: mode } = useTheme()
  const dp = data.params
  const dataKey = JSON.stringify([data.key, data.state.values[data.key], dp.n, dp.prevalence])

  // The data, from a recipe; the model is fitted to them and evaluated on them.
  const set = useMemo(() => {
    const d = recipe({ ...data.f!(null), seed: 1, n: dp.n, prevalence: dp.prevalence })
    const truth = d.meta.truth?.task === 'classification' ? (d.meta.truth as ClassificationTruth) : null
    return { x: d.x, y: d.y!, names: d.meta.labelNames ?? ['class 0', 'class 1'], truth }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the data's values
  }, [dataKey])
  const names = set.names
  const colors = useMemo(() => CLASS_SLOTS.map((s) => seriesColor(mode, s)), [mode])

  // The fit: only when the data or the model change, never when the threshold moves.
  const modelKey = JSON.stringify([model.key, model.state.values[model.key]])
  const scorer = useMemo(
    () => model.f!(dataset(set.x, set.y)),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the model's values
    [modelKey, set],
  )

  // The feature plane: a grid over the points, scored once per fit (and by the Bayes-optimal score once per data).
  const plane = useMemo(() => {
    const rows = toRows(set.x)
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
  }, [set])
  const field = useMemo(() => plane.toGrid(scorer.score(plane.g.points)), [scorer, plane])
  const bayes = useMemo(() => {
    const truth = set.truth
    if (!truth) return null
    // Truths score [n, d] batches of points.
    const onGrid = plane.toGrid(toFlat(truth.score(plane.g.points)))
    const scores = Float64Array.from(toFlat(truth.score(set.x)))
    return { onGrid, scores, error: truth.bayesError }
  }, [set, plane])

  const y = useMemo(() => toFlat(set.y), [set])
  const scores = useMemo(() => scorer.score(set.x), [scorer, set])
  // A symmetric range about 0 on the score scale, for the threshold, the histogram and the colour wash.
  const range = useMemo((): [number, number] => {
    let m = 0
    for (const v of scores) m = Math.max(m, Math.abs(v))
    for (const row of field) for (const v of row) m = Math.max(m, Math.abs(v))
    const top = Math.ceil(m) || 1
    return [-top, top]
  }, [scores, field])
  const fitKey = `${dataKey}|${modelKey}`
  const [chosen, setChosen] = useState<{ key: string; t: number } | null>(null)
  const t = chosen?.key === fitKey ? chosen.t : 0
  const setT = (v: number) => setChosen({ key: fitKey, t: Math.min(range[1], Math.max(range[0], v)) })

  const roc = useMemo(() => rocCurve(y, scores), [y, scores])
  const pr = useMemo(() => precisionRecallCurve(y, scores), [y, scores])
  const bayesCurves = useMemo(() => {
    if (!bayes) return null
    const r = rocCurve(y, bayes.scores)
    return {
      auroc: r.auc,
      roc: { name: 'Bayes-optimal', curve: r },
      pr: { name: 'Bayes-optimal', curve: precisionRecallCurve(y, bayes.scores) },
    }
  }, [bayes, y])
  const counts = countsAtThreshold(y, scores, t)
  const rates = binaryRates(counts)
  const predicted = useMemo(() => Array.from(scores, (v) => (v >= t ? 1 : 0)), [scores, t])
  // Positive class first, so the table reads TP FN / FP TN.
  const table = useMemo(() => confusionMatrix(y, predicted, { labels: [1, 0] }), [y, predicted])
  const probabilistic = useMemo(() => {
    if (!scorer.probability) return null
    const p = scorer.probability(set.x)
    return { logLoss: logLoss(y, p), brier: brierScore(y, p) }
  }, [scorer, set, y])

  // Hover coupling: a histogram bin rings its points; a point outlines its bin.
  const [lo, hi] = range
  const binWidth = (hi - lo) / BINS
  const binOf = (v: number) => Math.min(BINS - 1, Math.max(0, Math.floor((v - lo) / binWidth)))
  const [hover, setHover] = useState<{ bin: number } | { point: number } | null>(null)
  const onHistogram = (e: PlotPointer) => {
    if (e.type === 'move') setHover({ bin: binOf(e.point[0]) })
    else if (e.type === 'leave') setHover(null)
  }
  const px = useMemo(() => plane.rows.map((r) => r[0]), [plane])
  const py = useMemo(() => plane.rows.map((r) => r[1]), [plane])
  const onPlane = (e: PlotPointer) => {
    if (e.type === 'leave') return setHover(null)
    if (e.type !== 'move') return
    const [i, d2] = nearest(px, py, e.point)
    const r = (plane.gx[GRID - 1] - plane.gx[0]) * 0.03
    setHover(d2 <= r * r ? { point: i } : null)
  }
  const lit = useMemo(() => {
    if (!hover) return []
    if ('point' in hover) return [hover.point]
    const out: number[] = []
    scores.forEach((v, i) => {
      if (Math.min(BINS - 1, Math.max(0, Math.floor((v - lo) / binWidth))) === hover.bin) out.push(i)
    })
    return out
  }, [hover, scores, lo, binWidth])

  // Feature space: points coloured (and shaped) by actual class over the score field; the boundary in ink.
  const overlay = useMemo((): HeatmapOverlay[] => {
    const light = chrome('light')
    const boundary = polylines(contourLines(plane.gx, plane.gy, field, t))
    const out: HeatmapOverlay[] = [
      {
        name: 'points',
        type: 'scatter',
        x: px,
        y: py,
        group: y,
        groupNames: names,
        colors: y.map((c) => colors[c]),
        outline: chrome('light').surface,
      },
      {
        name: '__hovered',
        type: 'scatter',
        x: lit.map((i) => px[i]),
        y: lit.map((i) => py[i]),
        emphasis: true,
      },
      // Black in both themes, over a pale halo so that it reads on the dark theme's wash too.
      { name: '__boundary-halo', type: 'line', ...boundary, color: light.surface, width: 4 },
      { name: 'boundary at t', type: 'line', ...boundary, color: light.ink, width: 2 },
    ]
    if (bayes)
      out.push({
        name: 'Bayes boundary',
        type: 'line',
        ...polylines(contourLines(plane.gx, plane.gy, bayes.onGrid, 0)),
        color: chrome(mode).muted,
        dashed: true,
        width: 1.5,
      })
    return out
  }, [plane, px, py, field, t, bayes, y, names, colors, lit, mode])

  // Score space: each class's scores, binned on the score range; bars touch and overlap translucently.
  const histograms = useMemo((): XYSeries[] => {
    const bars = (c: 0 | 1): XYSeries => {
      const h = histogram(
        Array.from(scores).filter((_, i) => y[i] === c),
        { bins: BINS, range },
      )
      const b = histogramBars(h)
      return {
        name: `${names[c]} (${c})`,
        type: 'bar',
        x: b.x,
        y: b.counts,
        slot: CLASS_SLOTS[c],
        histogram: true,
      }
    }
    return [bars(0), bars(1)]
  }, [scores, y, range, names])
  // The hovered point's bin, in ink, as a patch.
  const litBin = useMemo((): XYSeries[] => {
    const i = hover && 'point' in hover ? hover.point : null
    const k = i === null ? null : Math.min(BINS - 1, Math.max(0, Math.floor((scores[i] - lo) / binWidth)))
    return [
      {
        name: '__hovered-bin',
        type: 'bar',
        x: k === null ? [] : [lo + (k + 0.5) * binWidth],
        y: k === null || i === null ? [] : [histograms[y[i]].y[k]],
        emphasis: true,
        histogram: true,
      },
    ]
  }, [hover, scores, y, histograms, lo, binWidth])

  const scoreName = scorer.kind === 'log-odds' ? 'log-odds of class 1' : 'SVM margin f(x)'
  const precisionAt = Number.isFinite(rates.precision) ? rates.precision : 1
  const fromCurve = (i: number, thresholds: Tensor) => {
    const v = toFlat(thresholds)[i]
    setT(Number.isFinite(v) ? v : range[1])
  }
  const rocHandles: Handle[] = [
    {
      kind: 'point',
      at: [rates.falsePositiveRate, rates.recall],
      label: 't',
      onDrag: (p) => fromCurve(nearest(toFlat(roc.fpr), toFlat(roc.tpr), p)[0], roc.thresholds),
    },
  ]
  const prHandles: Handle[] = [
    {
      kind: 'point',
      at: [rates.recall, precisionAt],
      label: 't',
      onDrag: (p) => fromCurve(nearest(toFlat(pr.recall), toFlat(pr.precision), p)[0], pr.thresholds),
    },
  ]
  const group = (title: string, children: ReactNode) => (
    <span className="inline-flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <span className="font-medium text-foreground">{title}</span>
      {children}
    </span>
  )

  return (
    <Figure
      id="a-threshold-its-confusion-matrix-and-the-roc-and-pr-curves"
      title="A classifier's threshold: decision boundary, contingency table, ROC and PR curves"
      description="A classifier from aifn is fitted to a 2-D two-class dataset and scores every point. One threshold t on that score sets the decision boundary in feature space, the split of the two class histograms, the contingency table, and the operating point on the ROC and precision–recall curves."
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="1 · data">
            <VariantControls variants={data} label="dataset" />
          </ControlRow>
          <ControlRow label="2 · model">
            <VariantControls variants={model} label="classifier" />
          </ControlRow>
          <ControlRow label="3 · threshold">
            <Slider
              label={`threshold t on the ${scoreName}`}
              value={t}
              min={range[0]}
              max={range[1]}
              step={0.05}
              onChange={setT}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          {group(
            scorer.kind === 'log-odds' ? `at t = ${fmt(t)} (P = ${fmt(sigmoid(t))})` : `at t = ${fmt(t)}`,
            <>
              <Readout label="accuracy" value={fmt(rates.accuracy)} />
              <Readout label="balanced accuracy" value={fmt(rates.balancedAccuracy)} />
              <Readout label="precision" value={fmt(rates.precision)} />
              <Readout label="recall" value={fmt(rates.recall)} />
              <Readout label="specificity" value={fmt(rates.specificity)} />
              <Readout label="F₁" value={fmt(rates.f1)} />
              <Readout label="MCC" value={fmt(rates.matthewsCorrelation)} />
              <Readout label="Cohen's κ" value={fmt(rates.cohensKappa)} />
            </>,
          )}
          {group(
            'ranking (every t)',
            <>
              <Readout label="AUROC" value={fmt(roc.auc)} />
              <Readout label="AP" value={fmt(pr.averagePrecision)} />
              {bayesCurves && <Readout label="Bayes-optimal AUROC" value={fmt(bayesCurves.auroc)} />}
            </>,
          )}
          {group(
            'probabilities',
            probabilistic ? (
              <>
                <Readout label="log loss" value={fmt(probabilistic.logLoss)} />
                <Readout label="Brier" value={fmt(probabilistic.brier)} />
              </>
            ) : (
              <span className="text-muted-foreground">none: the SVM gives margins, not probabilities</span>
            ),
          )}
          {bayes && group('population', <Readout label="Bayes error" value={fmt(bayes.error)} />)}
        </>
      }
      caption={`The model is fitted to the points shown and evaluated on the same points; its hyperparameters are in row 2, the training procedure is fixed. Drag the threshold line on the histograms, or drag along the ROC or PR curve: the black boundary, the histogram split, the table and both operating points move together. Colour is the actual class in every panel (blue class 0, red class 1); the field behind the points is the model's ${scoreName}, and the dashed grey contour is the Bayes-optimal boundary where the data's generating process is known (its curves are the grey ones on ROC and PR). Hover a histogram bin to mark its points, or a point to mark its bin; hover a rate in the table's margins to outline the cells it divides. Log loss is ∞ when some point gets probability 0 for its actual class, as k-NN and trees can.`}
    >
      <Dashboard>
        <DashboardRow ratio={1.15} minHeight={320}>
          <DashboardCell ratio={1.2} stackAspect={0.9}>
            <Heatmap
              x={plane.gx}
              y={plane.gy}
              z={field}
              xLabel="x₁"
              yLabel="x₂"
              scale="diverging"
              range={range}
              // The owner prefers the saturated field (full-strength diverging map with its pale midline) to a faded
              // wash, which reads as muddy on the dark surface: do not fade it. Points carry a light outline instead.
              valueLabel={scoreName}
              overlay={overlay}
              onPointer={onPlane}
              equalAspect
              zoom={false}
            />
          </DashboardCell>
          <DashboardCell>
            <XYChart
              series={histograms}
              live={litBin}
              xLabel={`score: ${scoreName}`}
              yLabel="cases"
              xRange={range}
              rescaleOnChange={false}
              axisKey={fitKey}
              handles={[{ kind: 'x', at: t, label: 't', onDrag: setT }]}
              onPointer={onHistogram}
              zoom={false}
            />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={300}>
          <DashboardCell>
            <ContingencyTableView
              table={table}
              classNames={[`${names[1]} (1)`, `${names[0]} (0)`]}
              classColors={[colors[1], colors[0]]}
              positive={0}
            />
          </DashboardCell>
          <DashboardCell aspect="square">
            <CurveChart
              curve={roc}
              handles={rocHandles}
              summaryInLegend
              slot={CURVE_SLOT}
              reference={bayesCurves?.roc}
            />
          </DashboardCell>
          <DashboardCell aspect="square">
            <CurveChart curve={pr} handles={prHandles} summaryInLegend slot={CURVE_SLOT} reference={bayesCurves?.pr} />
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
