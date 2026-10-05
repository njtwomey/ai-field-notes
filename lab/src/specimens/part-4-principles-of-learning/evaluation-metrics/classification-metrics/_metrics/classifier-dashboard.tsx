import { useMemo, useState, type ReactNode } from 'react'
import { decisionTree } from 'aifn-methods/learning/trees-and-ensembles'
import {
  gaussianNaiveBayes,
  linearDiscriminant,
  quadraticDiscriminant,
} from 'aifn-methods/learning/generative-classifiers'
import { gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import { randomForest } from 'aifn-methods/learning/trees-and-ensembles/bagging'
import { supportVectorMachine } from 'aifn-methods/learning/kernel-methods'
import { pipeline } from 'aifn-compute/learning/compose'
import { recipe, type RecipeInput } from 'aifn-methods/data'
import { type ClassificationTruth } from 'aifn-methods/data'
import { classProbabilities, dataset, type Distribution, type Supervised } from 'aifn-compute/learning/estimators'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { contourLines, grid2d } from 'aifn-compute/numerics/geometry'
import { gpClassifier } from 'aifn-methods/learning/gaussian-processes'
import { rbf } from 'aifn-compute/learning/kernels'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import {
  binaryRates,
  brierScore,
  confusionMatrix,
  countsAtThreshold,
  logLoss,
  precisionRecallCurve,
  rocCurve,
} from 'aifn-compute/learning/metrics'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { Mlp } from 'aifn-compute/nn/layers'
import { xavierUniform } from 'aifn-compute/nn/init'
import { type Params } from 'aifn-compute/foundation/pytree'
import { polynomialFeatures, splineFeatures, standardScaler } from 'aifn-methods/learning/preprocessing'
import { stream } from 'aifn-compute/foundation/random'
import { logit, sigmoid } from 'aifn-compute/numerics/special'
import { histogram } from 'aifn-compute/probability/stats'
import { fromData, toFlat, toRows, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { chrome, seriesColor } from 'aifn-render/design'
import { useTheme } from 'aifn-render/design'
import { Slider } from 'aifn-render/controls'
import { ControlRow, Dashboard, DashboardCell, DashboardRow, Figure } from 'aifn-render/layout'
import { slider, useComputed, useFigureState, variants } from 'aifn-render/state'
import {
  Bars,
  Curve,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  Handle,
  type Handle as HandleSpec,
  type PlotPointer,
} from 'aifn-render/viz'
import { ContingencyTableView, CurveChart, formatValue, histogramBars } from '@lab/views'

// ── Data: aifn/datasets recipes, two classes, class 1 positive ─────────────────────────────────────────────────────

/** A dataset recipe per base; `n` and `prevalence` are shared. Only the data are controlled, never the training. */
const DATA = variants(
  {
    moons: { label: 'two moons', params: { noise: slider(0, 0.5, 0.25, { label: 'noise sd' }) } },
    blobs: {
      label: 'two Gaussian blobs',
      params: { separation: slider(0, 5, 2, { label: 'separation (blob sds)' }) },
    },
    circles: {
      label: 'two circles',
      params: {
        noise: slider(0, 0.3, 0.12, { label: 'noise sd' }),
        factor: slider(0.1, 0.9, 0.5, { label: 'inner radius' }),
      },
    },
    xor: { label: 'XOR (four blobs)', params: { noise: slider(0.1, 1, 0.45, { label: 'blob sd' }) } },
    spirals: {
      label: 'two spirals',
      params: {
        noise: slider(0, 0.2, 0.05, { label: 'noise sd' }),
        turns: slider(0.5, 2, 1, { label: 'turns' }),
      },
    },
    anisotropic: {
      label: 'anisotropic Gaussians',
      params: { separation: slider(0, 5, 2.5, { label: 'separation (blob sds)' }) },
    },
    checkerboard: {
      label: 'checkerboard',
      params: { tiles: slider(2, 6, 3, { step: 1, label: 'tiles per side' }) },
    },
  },
  {
    label: '1 · data',
    choiceLabel: 'dataset',
    shared: {
      n: slider(60, 600, 240, { step: 10, label: 'cases n' }),
      prevalence: slider(0.05, 0.95, 0.3, { step: 0.05, label: 'prevalence π of class 1' }),
    },
  },
)

/** One case's values, as the recipe and fit tables read them (each reads only its own case's fields). */
type CaseValues = Readonly<Record<string, number>>

const RECIPES: Record<string, (p: CaseValues) => RecipeInput> = {
  moons: (p) => ({ base: 'moons', knobs: { noise: p.noise, spacing: 'random' } }),
  blobs: (p) => ({ base: 'blobs', knobs: { centers: 2, sd: 1, layout: 'polygon', separation: p.separation } }),
  circles: (p) => ({ base: 'circles', knobs: { noise: p.noise, factor: p.factor } }),
  xor: (p) => ({ base: 'xor', knobs: { kind: 'gaussian', sd: p.noise } }),
  spirals: (p) => ({ base: 'spirals', knobs: { noise: p.noise, arms: 2, turns: p.turns } }),
  anisotropic: (p) => ({
    base: 'blobs',
    knobs: { centers: 2, sd: 1, layout: 'polygon', separation: p.separation },
    // The recipe's linear map: a stretch of 0.6, a shear of 1 and a rotation by −π/4 tilt and elongate the classes.
    modifiers: [{ op: 'withTransform', params: { rotation: -Math.PI / 4, shear: 1, stretch: 0.6 } }],
  }),
  checkerboard: (p) => ({ base: 'checkerboard', knobs: { tiles: p.tiles } }),
}

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
const MODELS = variants(
  {
    logistic: {
      label: 'logistic regression',
      params: { l2: logSlider(-3, 2, 0, 'λ (L2)') },
    },
    polynomial: {
      label: 'polynomial logistic regression',
      params: { degree: slider(1, 6, 3, { step: 1, label: 'degree' }), l2: logSlider(-3, 2, -1, 'λ (L2)') },
    },
    spline: {
      label: 'spline logistic regression (additive)',
      params: { knots: slider(3, 12, 6, { step: 1, label: 'knots per feature' }), l2: logSlider(-3, 2, -1, 'λ (L2)') },
    },
    knn: {
      label: 'k-nearest neighbours',
      params: { k: slider(1, 51, 15, { step: 1, label: 'neighbours k' }) },
    },
    lda: {
      label: 'linear discriminant (LDA)',
      params: {},
    },
    qda: {
      label: 'quadratic discriminant (QDA)',
      params: {},
    },
    bayes: { label: 'Gaussian naive Bayes', params: {} },
    svm: {
      label: 'SVM, RBF kernel',
      params: { C: slider(0.1, 20, 1, { label: 'C' }), gamma: slider(0.1, 10, 1.4, { label: 'γ (RBF)' }) },
    },
    tree: {
      label: 'decision tree',
      params: { depth: slider(1, 12, 4, { step: 1, label: 'maximum depth' }) },
    },
    forest: {
      label: 'random forest',
      params: {
        trees: slider(5, 100, 50, { step: 5, label: 'trees' }),
        depth: slider(1, 12, 6, { step: 1, label: 'maximum depth' }),
      },
    },
    boosting: {
      label: 'gradient boosting',
      params: {
        trees: slider(5, 200, 60, { step: 5, label: 'trees' }),
        depth: slider(1, 4, 2, { step: 1, label: 'tree depth' }),
      },
    },
    gp: {
      label: 'GP classifier (Laplace, RBF)',
      params: { lengthscale: slider(0.1, 3, 0.6, { label: 'lengthscale ℓ' }) },
    },
    mlp: {
      label: 'MLP (two tanh layers)',
      params: {
        width: slider(2, 32, 12, { step: 1, label: 'units per layer' }),
        decay: logSlider(-5, -1, -4, 'weight decay'),
      },
    },
  },
  { label: '2 · model', choiceLabel: 'classifier' },
)

const FITS: Record<string, (d: Fit, p: CaseValues) => Scorer> = {
  logistic: (d: Fit, p: CaseValues): Scorer => predictive(logisticRegression({ l2: 10 ** p.l2 }).fit(d)),
  polynomial: (d: Fit, p: CaseValues): Scorer =>
    predictive(
      pipeline(
        standardScaler(),
        polynomialFeatures({ degree: p.degree, includeBias: false }),
        logisticRegression({ l2: 10 ** p.l2 }),
      ).fit(d),
    ),
  spline: (d: Fit, p: CaseValues): Scorer =>
    predictive(
      pipeline(
        splineFeatures({ knots: p.knots, degree: 3, extrapolation: 'continue', includeBias: false }),
        logisticRegression({ l2: 10 ** p.l2 }),
      ).fit(d),
    ),
  knn: (d: Fit, p: CaseValues): Scorer => predictive(kNearestNeighbours({ k: Math.min(p.k, d.x.shape[0]) }).fit(d)),
  lda: (d: Fit): Scorer => predictive(linearDiscriminant().fit(d)),
  qda: (d: Fit): Scorer => predictive(quadraticDiscriminant({ regularisation: 0.01 }).fit(d)),
  bayes: (d: Fit): Scorer => predictive(gaussianNaiveBayes().fit(d)),
  svm: (d: Fit, p: CaseValues): Scorer => {
    // k(x, x′) = exp(−γ‖x − x′‖²), i.e. lengthscale ℓ = 1/√(2γ).
    const kernel = rbf({ lengthscale: 1 / Math.sqrt(2 * p.gamma) })
    const m = supportVectorMachine({ C: p.C, kernel }).fit(d)
    return { kind: 'margin', score: (x) => Float64Array.from(toFlat(m.score(x))) }
  },
  tree: (d: Fit, p: CaseValues): Scorer => predictive(decisionTree({ maxDepth: p.depth }).fit(d)),
  forest: (d: Fit, p: CaseValues): Scorer =>
    predictive(
      randomForest({ trees: p.trees, maxDepth: p.depth, maxFeatures: 1 }).fit(d, {
        stream: stream('lab/metrics/forest'),
      }),
    ),
  boosting: (d: Fit, p: CaseValues): Scorer => {
    const m = gradientBoosting({
      loss: 'logistic',
      stages: p.trees,
      learningRate: 0.2,
      tree: { maxDepth: p.depth },
    }).fit(d)
    return predictive({ predictive: (x) => m.predictive!(x) })
  },
  gp: (d: Fit, p: CaseValues): Scorer =>
    predictive(gpClassifier({ kernel: rbf({ lengthscale: p.lengthscale, variance: 4 }) }).fit(d)),
  mlp: (d: Fit, p: CaseValues): Scorer => {
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
}

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
  const state = useFigureState({ data: DATA, model: MODELS })
  const data = state.data
  const model = state.model
  const { resolved: mode } = useTheme()
  const dp = data.values as unknown as CaseValues
  const dataKey = JSON.stringify([data.key, data.values])

  // The data, from a recipe; the model is fitted to them and evaluated on them.
  const set = useMemo(() => {
    const r = RECIPES[data.key](dp)
    const d = recipe({ ...r, seed: 1, knobs: { ...r.knobs, n: dp.n, prevalence: dp.prevalence } })
    const truth = d.meta.truth?.task === 'classification' ? (d.meta.truth as ClassificationTruth) : null
    return { x: d.x, y: d.y!, names: d.meta.labelNames ?? ['class 0', 'class 1'], truth }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the data's values
  }, [dataKey])
  const names = set.names
  const colors = useMemo(() => CLASS_SLOTS.map((s) => seriesColor(mode, s)), [mode])

  // The fit: only when the data or the model change, never when the threshold moves. Some fits (a forest, the GP,
  // the MLP's 300 Adam steps) take longer than a frame, so a slider drag refits on release.
  const modelKey = JSON.stringify([model.key, model.values])
  const fitted = useComputed(
    () => FITS[model.key](dataset(set.x, set.y), model.values as unknown as CaseValues),
    [modelKey, set],
    { mode: 'release' },
  )
  const scorer = fitted.value

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
      auroc: r.area,
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

  // Feature space: the decision boundary at t over the score field, and the Bayes-optimal boundary where known.
  const boundary = useMemo(() => polylines(contourLines(plane.gx, plane.gy, field, t)), [plane, field, t])
  const bayesBoundary = useMemo(
    () => (bayes ? polylines(contourLines(plane.gx, plane.gy, bayes.onGrid, 0)) : null),
    [plane, bayes],
  )
  const byClass = useMemo(
    () =>
      ([0, 1] as const).map((c) => {
        const idx = y.flatMap((v, i) => (v === c ? [i] : []))
        return { x: idx.map((i) => px[i]), y: idx.map((i) => py[i]) }
      }),
    [y, px, py],
  )
  const light = chrome('light')

  // Score space: each class's scores, binned on the score range; bars touch and overlap translucently.
  const histograms = useMemo(() => {
    const bars = (c: 0 | 1) => {
      const h = histogram(
        Array.from(scores).filter((_, i) => y[i] === c),
        { bins: BINS, range },
      )
      return histogramBars(h)
    }
    return [bars(0), bars(1)]
  }, [scores, y, range])
  // The hovered point's bin, in ink, as a patch.
  const litBin = useMemo(() => {
    const i = hover && 'point' in hover ? hover.point : null
    if (i === null) return null
    const k = Math.min(BINS - 1, Math.max(0, Math.floor((scores[i] - lo) / binWidth)))
    return {
      x: [lo + (k + 0.5) * binWidth],
      y: [histograms[y[i]].counts[k]],
      edges: [lo + k * binWidth, lo + (k + 1) * binWidth],
    }
  }, [hover, scores, y, histograms, lo, binWidth])

  const scoreName = scorer.kind === 'log-odds' ? 'log-odds of class 1' : 'SVM margin f(x)'
  const fx1 = useAxis({ label: 'x₁', zoom: false })
  const fx2 = useAxis({ label: 'x₂', equal: fx1, zoom: false })
  const sx = useAxis({ label: `score: ${scoreName}`, range, zoom: false })
  const cy = useAxis({ label: 'cases', hold: 'initial', key: fitKey, zoom: false })
  const precisionAt = Number.isFinite(rates.precision) ? rates.precision : 1
  const fromCurve = (i: number, thresholds: Tensor) => {
    const v = toFlat(thresholds)[i]
    setT(Number.isFinite(v) ? v : range[1])
  }
  const rocHandles: HandleSpec[] = [
    {
      kind: 'point',
      at: [rates.falsePositiveRate, rates.recall],
      label: 't',
      onDrag: (p) => fromCurve(nearest(toFlat(roc.x), toFlat(roc.y), p)[0], roc.thresholds),
    },
  ]
  const prHandles: HandleSpec[] = [
    {
      kind: 'point',
      at: [rates.recall, precisionAt],
      label: 't',
      onDrag: (p) => fromCurve(nearest(toFlat(pr.x), toFlat(pr.y), p)[0], pr.thresholds),
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
      purpose="One threshold t on a classifier's score sets, all at once, the decision boundary in feature space, the split of the two class histograms, the contingency table, and the operating point on the ROC and precision–recall curves."
      defaultSize="XL"
      state={state}
      controls={
        <>
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
              <Readout label="AUROC" value={fmt(roc.area)} />
              <Readout label="AP" value={fmt(pr.area)} />
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
            <Plot x={fx1} y={fx2} onPointer={onPlane} toolbar={false} legend={false}>
              {/* The owner prefers the saturated field (full-strength diverging map with its pale midline) to a faded
                  wash, which reads as muddy on the dark surface: do not fade it. Points carry a light ring instead. */}
              <Raster
                x={plane.gx}
                y={plane.gy}
                z={field}
                scale="diverging"
                range={range}
                valueLabel={scoreName}
                stale={fitted.stale}
              />
              {bayesBoundary && (
                <Curve
                  name="Bayes boundary"
                  x={bayesBoundary.x}
                  y={bayesBoundary.y}
                  color={chrome(mode).muted}
                  dashed
                  width={1.5}
                />
              )}
              <Points name={names[0]} x={byClass[0].x} y={byClass[0].y} slot={CLASS_SLOTS[0]} shape={0} />
              <Points name={names[1]} x={byClass[1].x} y={byClass[1].y} slot={CLASS_SLOTS[1]} shape={1} />
              {/* Black in both themes, over a pale halo so that it reads on the dark theme's wash too. */}
              <Curve name="__boundary-halo" x={boundary.x} y={boundary.y} color={light.surface} width={4} silent live />
              <Curve name="boundary at t" x={boundary.x} y={boundary.y} color={light.ink} width={2} silent live />
              <Points name="__hovered" x={lit.map((i) => px[i])} y={lit.map((i) => py[i])} emphasis live />
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plot x={sx} y={cy} onPointer={onHistogram} toolbar={false}>
              {([0, 1] as const).map((c) => (
                <Bars
                  key={c}
                  name={`${names[c]} (${c})`}
                  x={histograms[c].x}
                  y={histograms[c].counts}
                  edges={histograms[c].edges}
                  slot={CLASS_SLOTS[c]}
                  stale={fitted.stale}
                />
              ))}
              {litBin && <Bars name="__hovered-bin" x={litBin.x} y={litBin.y} edges={litBin.edges} emphasis live />}
              <Handle kind="x" at={t} label="t" onDrag={setT} />
            </Plot>
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
