/**
 * Smoke tests for aifn/metrics: one or two hand-checked cases per family, mostly the worked examples of the site's
 * notes. The full scikit-learn / scipy fixture suite is still to come.
 */
import { describe, expect, it } from 'vitest'
import {
  accuracy,
  adjustedMutualInformation,
  adjustedRandIndex,
  auroc,
  averagePrecision,
  averagePrecisionFromMatches,
  balancedAccuracy,
  binaryRates,
  binormalAuroc,
  binormalAveragePrecision,
  bleuScore,
  bootstrapMetric,
  boxOverlap,
  brierDecomposition,
  brierScore,
  calinskiHarabasz,
  chrFScore,
  cohensKappa,
  concordanceCorrelation,
  confusionMargins,
  confusionMatrix,
  cramersV,
  crpsEnsemble,
  crpsGaussian,
  daviesBouldin,
  dcg,
  demographicParityDifference,
  dice,
  dunnIndex,
  equalisedOddsDifference,
  expectedCalibrationError,
  expectedReciprocalRank,
  explainedVariance,
  f1,
  fBeta,
  fleissKappa,
  fowlkesMallows,
  frechetDistance,
  generativePrecisionRecall,
  getMetric,
  giniCoefficient,
  hammingLoss,
  hausdorffDistances,
  hitRate,
  inceptionScore,
  intervalScore,
  intraclassCorrelation,
  jaccardScore,
  kappaFromTable,
  kid,
  krippendorffAlpha,
  listMetrics,
  logLoss,
  macroMeanAbsoluteError,
  matthewsCorrelation,
  maximumCalibrationError,
  meanAbsoluteError,
  meanAbsolutePercentageError,
  meanAbsoluteScaledError,
  meanAveragePrecision,
  meanReciprocalRank,
  meanSquaredError,
  metricRegistry,
  minkowskiDistance,
  cosineSimilarity,
  ndcg,
  normalisedMutualInformation,
  panopticFromMatches,
  partialAuroc,
  pinballLoss,
  precision,
  precisionAtK,
  precisionRecallCurve,
  precisionRecallTrapezoid,
  procrustesDisparity,
  psnr,
  r2Score,
  rPrecision,
  randIndex,
  recall,
  recallAtK,
  reliabilityDiagram,
  rocCurve,
  rootMeanSquaredScaledError,
  rougeLScores,
  rougeNScores,
  siSdr,
  silhouetteScore,
  snr,
  squadExactMatch,
  squadF1,
  ssimMap,
  symmetricMeanAbsolutePercentageError,
  theilsU,
  translationEditRate,
  tweedieDeviance,
  vMeasure,
  wilsonInterval,
  wordErrorRate,
  editAlignment,
  ergas,
  spectralAngle,
  permutationInvariantScore,
  quadraticWeightedKappa,
  logCoshError,
  meanSquaredLogError,
} from 'aifn/metrics'
import { stream } from 'aifn/random'
import { toFlat } from 'aifn/tensor'

/** The screening data of the confusion-matrix note: TP 40, FN 10, FP 90, TN 860. */
function screening() {
  const yTrue: number[] = []
  const yPred: number[] = []
  const push = (t: number, p: number, n: number) => {
    for (let i = 0; i < n; i++) {
      yTrue.push(t)
      yPred.push(p)
    }
  }
  push(1, 1, 40)
  push(1, 0, 10)
  push(0, 1, 90)
  push(0, 0, 860)
  return { yTrue, yPred }
}

/** Labels reproducing a confusion matrix (rows true, columns predicted). */
function fromMatrix(m: number[][]) {
  const yTrue: number[] = []
  const yPred: number[] = []
  m.forEach((row, j) =>
    row.forEach((c, k) => {
      for (let i = 0; i < c; i++) {
        yTrue.push(j)
        yPred.push(k)
      }
    }),
  )
  return { yTrue, yPred }
}

describe('classification', () => {
  const { yTrue, yPred } = screening()
  it('reproduces the screening table', () => {
    const r = binaryRates({ tp: 40, fp: 90, fn: 10, tn: 860 })
    expect(r.accuracy).toBeCloseTo(0.9, 10)
    expect(r.balancedAccuracy).toBeCloseTo(0.853, 3)
    expect(r.matthewsCorrelation).toBeCloseTo(0.457, 3)
    expect(r.cohensKappa).toBeCloseTo(0.401, 3)
    expect(r.jaccard).toBeCloseTo(0.286, 3)
    expect(r.positiveLikelihoodRatio).toBeCloseTo(8.44, 2)
    expect(accuracy(yTrue, yPred)).toBeCloseTo(0.9, 12)
    expect(precision(yTrue, yPred)).toBeCloseTo(40 / 130, 12)
    expect(recall(yTrue, yPred)).toBeCloseTo(0.8, 12)
    expect(f1(yTrue, yPred)).toBeCloseTo(0.444, 3)
    expect(fBeta(yTrue, yPred, { beta: 2 })).toBeCloseTo(0.606, 3)
    expect(fBeta(yTrue, yPred, { beta: 0.5 })).toBeCloseTo(0.351, 3)
    expect(matthewsCorrelation(yTrue, yPred)).toBeCloseTo(0.457, 3)
    expect(cohensKappa(yTrue, yPred)).toBeCloseTo(0.401, 3)
    // Positive class first: rows [TP FN; FP TN], so the margins read TPR/FNR, TNR/FPR, PPV/FDR, NPV/FOR.
    const m = confusionMargins(confusionMatrix(yTrue, yPred, { labels: [1, 0] }))
    expect(toFlat(m.actual)).toEqual([50, 950])
    expect(toFlat(m.predicted)).toEqual([130, 870])
    expect(m.n).toBe(1000)
    expect(toFlat(m.rowRate)[0]).toBeCloseTo(0.8, 12)
    expect(toFlat(m.rowRate)[1]).toBeCloseTo(860 / 950, 12)
    expect(toFlat(m.rowMiss)[1]).toBeCloseTo(90 / 950, 12)
    expect(toFlat(m.columnRate)[0]).toBeCloseTo(40 / 130, 12)
    expect(toFlat(m.columnMiss)[1]).toBeCloseTo(10 / 870, 12)
    expect(toFlat(m.prevalence)[0]).toBeCloseTo(0.05, 12)
    expect(m.accuracy).toBeCloseTo(0.9, 12)
    expect(balancedAccuracy(yTrue, yPred)).toBeCloseTo(0.853, 3)
  })
  it('averages the three-class example', () => {
    const { yTrue: t, yPred: p } = fromMatrix([
      [70, 6, 4],
      [4, 10, 1],
      [2, 1, 2],
    ])
    expect(precision(t, p, { average: 'micro' })).toBeCloseTo(0.82, 12)
    expect(precision(t, p, { average: 'macro' })).toBeCloseTo(0.598, 3)
    expect(recall(t, p, { average: 'macro' })).toBeCloseTo(0.647, 3)
    expect(f1(t, p, { average: 'macro' })).toBeCloseTo(0.619, 3)
    expect(f1(t, p, { average: 'weighted' })).toBeCloseTo(0.828, 3)
    expect(matthewsCorrelation(t, p)).toBeCloseTo(0.507, 3)
    expect(toFlat(confusionMatrix(t, p).matrix)).toEqual([70, 6, 4, 4, 10, 1, 2, 1, 2])
  })
  it('handles multi-label rows', () => {
    const t = [
      [1, 0, 1],
      [0, 1, 0],
      [1, 1, 0],
      [0, 0, 1],
    ]
    const p = [
      [1, 0, 0],
      [0, 1, 0],
      [1, 1, 0],
      [1, 0, 1],
    ]
    expect(accuracy(t, p)).toBeCloseTo(0.5, 12)
    expect(hammingLoss(t, p)).toBeCloseTo(2 / 12, 12)
    expect(jaccardScore(t, p, { average: 'samples' })).toBeCloseTo(0.75, 12)
    expect(jaccardScore(t, p, { average: 'micro' })).toBeCloseTo(5 / 7, 12)
    expect(jaccardScore(t, p, { average: 'macro' })).toBeCloseTo(0.722, 3)
  })
  it('weights kappa and scores ordinal predictions', () => {
    const table = [
      [20, 5, 1],
      [4, 15, 5],
      [1, 4, 15],
    ]
    expect(kappaFromTable(table)).toBeCloseTo(0.57, 3)
    expect(kappaFromTable(table, 'linear')).toBeCloseTo(0.642, 3)
    expect(kappaFromTable(table, 'quadratic')).toBeCloseTo(0.715, 3)
    const a = fromMatrix([
      [3, 1, 0],
      [1, 2, 1],
      [0, 0, 2],
    ])
    expect(quadraticWeightedKappa(a.yTrue, a.yPred)).toBeCloseTo(0.762, 3)
    expect(macroMeanAbsoluteError(a.yTrue, a.yPred)).toBeCloseTo(0.25, 12)
  })
})

describe('curves', () => {
  const y = [1, 1, 1, 1, 0, 0, 0, 0, 0]
  const s = [0.9, 0.8, 0.6, 0.55, 0.7, 0.5, 0.4, 0.3, 0.2]
  it('computes AUROC, AP and the trapezoid of the worked example', () => {
    expect(auroc(y, s)).toBeCloseTo(0.9, 12)
    expect(rocCurve(y, s).auc).toBeCloseTo(0.9, 12)
    expect(averagePrecision(y, s)).toBeCloseTo(0.8875, 12)
    expect(precisionRecallTrapezoid(precisionRecallCurve(y, s))).toBeCloseTo(0.871, 3)
  })
  it('counts ties as one half', () => {
    expect(auroc([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1])).toBeCloseTo(0.875, 12)
    const c = rocCurve([1, 0], [0.5, 0.5])
    expect(toFlat(c.fpr)).toEqual([0, 1])
    expect(toFlat(c.tpr)).toEqual([0, 1])
  })
  it('standardises partial AUROC', () => {
    // A perfect ranking scores 1 whatever the cut-off.
    expect(partialAuroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], { maxFpr: 0.3 })).toBeCloseTo(1, 12)
  })
  it('matches the binormal closed forms', () => {
    expect(binormalAuroc(1.5)).toBeCloseTo(0.856, 3)
    expect(binormalAveragePrecision({ separation: 1.5, prevalence: 0.5 })).toBeCloseTo(0.854, 3)
    expect(binormalAveragePrecision({ separation: 1.5, prevalence: 0.1 })).toBeCloseTo(0.478, 3)
  })
  it('computes multiclass AUROC', () => {
    const t = [0, 1, 2, 0, 1, 2]
    const p = [
      [0.8, 0.1, 0.1],
      [0.2, 0.7, 0.1],
      [0.1, 0.2, 0.7],
      [0.6, 0.3, 0.1],
      [0.3, 0.4, 0.3],
      [0.2, 0.3, 0.5],
    ]
    expect(auroc(t, p)).toBeCloseTo(1, 12)
    expect(auroc(t, p, { multiClass: 'ovo' })).toBeCloseTo(1, 12)
  })
})

describe('probabilistic', () => {
  const p = [0.1, 0.15, 0.25, 0.3, 0.45, 0.55, 0.7, 0.75, 0.85, 0.95]
  const y = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  it('reproduces the log loss, Brier and ECE examples', () => {
    expect(logLoss(y, p)).toBeCloseTo(0.466, 3)
    expect(brierScore(y, p)).toBeCloseTo(0.157, 3)
    expect(expectedCalibrationError(y, p, { bins: 5 })).toBeCloseTo(0.135, 12)
    expect(maximumCalibrationError(y, p, { bins: 5 })).toBeCloseTo(0.225, 12)
    expect(reliabilityDiagram(y, p, { bins: 5 }).ece).toBeCloseTo(0.135, 12)
  })
  it('decomposes the Brier score exactly by distinct forecasts', () => {
    const d = brierDecomposition(y, p)
    expect(d.residual).toBeCloseTo(0, 12)
    expect(d.brier).toBeCloseTo(brierScore(y, p), 12)
  })
  it('scores Gaussian and ensemble forecasts', () => {
    expect(crpsGaussian([0.8], { mean: 0, sd: 1 })).toBeCloseTo(0.476, 3)
    expect(crpsGaussian([0], { mean: 0, sd: 1 })).toBeCloseTo(0.234, 3)
    expect(crpsEnsemble(0.8, [0.3])).toBeCloseTo(0.5, 12)
    expect(crpsEnsemble([1], [[0, 2]])).toBeCloseTo(1 - 0.5, 12)
    expect(intervalScore([2], { lower: -1.2816, upper: 1.2816 }, { alpha: 0.2 })).toBeCloseTo(9.75, 2)
  })
})

describe('regression and forecasting', () => {
  const y = [3, -0.5, 2, 7]
  const p = [2.5, 0, 2, 8]
  it('reproduces the notes', () => {
    expect(meanSquaredError(y, p)).toBeCloseTo(0.375, 12)
    expect(meanAbsoluteError(y, p)).toBeCloseTo(0.5, 12)
    expect(r2Score(y, p)).toBeCloseTo(0.949, 3)
    expect(explainedVariance(y, p)).toBeCloseTo(0.957, 3)
    const a = [1, 2, 4, 8]
    const b = [1.5, 2, 3, 10]
    expect(tweedieDeviance(a, b, { power: 1 })).toBeCloseTo(0.2301, 4)
    expect(tweedieDeviance(a, b, { power: 1.5 })).toBeCloseTo(0.1179, 4)
    expect(tweedieDeviance(a, b, { power: 2 })).toBeCloseTo(0.0705, 4)
    expect(meanSquaredLogError(a, b)).toBeCloseTo(0.035, 3)
    expect(logCoshError(a, b)).toBeCloseTo(0.4697, 4)
    expect(pinballLoss(a, b, { tau: 0.9 })).toBeCloseTo(0.2875, 12)
  })
  it('computes percentage and scaled errors', () => {
    expect(meanAbsolutePercentageError([100, 50, 20, 80], [110, 40, 30, 80])).toBeCloseTo(0.2, 12)
    expect(symmetricMeanAbsolutePercentageError([100, 50, 20, 80], [110, 40, 30, 80])).toBeCloseTo(0.179, 3)
    const train = [10, 12, 11, 13, 15, 14]
    expect(meanAbsoluteScaledError([16, 15], [15, 16], { train })).toBeCloseTo(0.625, 12)
    expect(rootMeanSquaredScaledError([16, 15], [15, 16], { train })).toBeCloseTo(0.6, 2)
  })
})

describe('ranking', () => {
  const grades = [3, 2, 3, 0, 1, 2]
  it('reproduces the nDCG example with both gains', () => {
    expect(ndcg(grades, { gain: 'linear' })).toBeCloseTo(0.961, 3)
    expect(ndcg(grades)).toBeCloseTo(0.949, 3)
    expect(ndcg(grades, { gain: 'linear', k: 3 })).toBeCloseTo(0.978, 3)
    expect(dcg(grades, { gain: 'linear' })).toBeCloseTo(6.861, 3)
  })
  it('ranks by scores with tie averaging', () => {
    // Reversed scores rank the items in the listed order.
    expect(ndcg(grades, [6, 5, 4, 3, 2, 1], { gain: 'linear' })).toBeCloseTo(0.961, 3)
    // All tied: the expected DCG is the mean gain times the sum of discounts.
    const discounts = [1, 2, 3].reduce((s, i) => s + 1 / Math.log2(i + 1), 0)
    expect(dcg([1, 0, 2], [0, 0, 0], { gain: 'linear' })).toBeCloseTo(discounts, 12)
  })
  it('computes the retrieval metrics', () => {
    const list = [1, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    expect(precisionAtK(list, { k: 5 })).toBeCloseTo(0.6, 12)
    expect(recallAtK(list, { k: 10, totalRelevant: 5 })).toBeCloseTo(0.8, 12)
    expect(rPrecision(list, { totalRelevant: 5 })).toBeCloseTo(0.6, 12)
    expect(meanAveragePrecision(list, { totalRelevant: 5 })).toBeCloseTo(0.598, 3)
    const firsts = [
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ]
    expect(meanReciprocalRank(firsts)).toBeCloseTo(0.611, 3)
    expect(
      hitRate(
        [
          [1, 0, 0],
          [0, 0, 1],
          [0, 1, 0],
          [0, 0, 0],
        ],
        { k: 2 },
      ),
    ).toBeCloseTo(0.5, 12)
  })
  it('reproduces ERR and the ideal-including nDCG', () => {
    const a = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    const b = [1, 1, 2, 1, 1, 0, 0, 0, 0, 0]
    expect(expectedReciprocalRank(a, { maxGrade: 3 })).toBeCloseTo(0.875, 3)
    expect(expectedReciprocalRank(b, { maxGrade: 3 })).toBeCloseTo(0.301, 3)
    expect(ndcg(a, { gain: 'linear', ideal: [3, 1, 1, 1, 1, 1] })).toBeCloseTo(0.566, 3)
    expect(giniCoefficient([50, 30, 10, 5, 5])).toBeCloseTo(0.46, 12)
  })
})

describe('clustering', () => {
  const t = [0, 0, 0, 1, 1, 1, 2, 2, 2]
  const p = [0, 0, 1, 1, 1, 1, 2, 2, 0]
  it('reproduces the pair-counting and information-theoretic examples', () => {
    expect(randIndex(t, p)).toBeCloseTo(0.75, 12)
    expect(adjustedRandIndex(t, p)).toBeCloseTo(0.357, 3)
    expect(fowlkesMallows(t, p)).toBeCloseTo(0.527, 3)
    expect(vMeasure(t, p)).toBeCloseTo(0.59, 3)
    expect(normalisedMutualInformation(t, p, { average: 'geometric' })).toBeCloseTo(0.59, 3)
    expect(adjustedMutualInformation(t, p)).toBeCloseTo(0.409, 3)
  })
  it('reproduces the internal indices on five points', () => {
    const x = [[0], [1], [5], [6], [7]]
    const l = [0, 0, 1, 1, 1]
    expect(silhouetteScore(x, l)).toBeCloseTo(0.777, 3)
    expect(calinskiHarabasz(x, l)).toBeCloseTo(43.56, 2)
    expect(daviesBouldin(x, l)).toBeCloseTo(0.212, 3)
    expect(dunnIndex(x, l)).toBeCloseTo(2, 12)
  })
})

describe('agreement', () => {
  it('reproduces the notes', () => {
    expect(concordanceCorrelation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])).toBeCloseTo(0.421, 3)
    expect(
      fleissKappa([
        [4, 0, 0],
        [2, 2, 0],
        [0, 3, 1],
        [1, 1, 2],
        [0, 0, 4],
      ]),
    ).toBeCloseTo(0.398, 3)
    const table = [
      [20, 10, 5],
      [10, 20, 15],
    ]
    expect(cramersV(table)).toBeCloseTo(0.364, 3)
    expect(theilsU(table)).toBeCloseTo(0.062, 3)
    expect(theilsU(table, { of: 'rows' })).toBeCloseTo(0.098, 3)
  })
  it("reproduces Krippendorff's reliability-data example", () => {
    const n = NaN
    const data = [
      [1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
      [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
      [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
      [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n],
    ]
    expect(krippendorffAlpha(data)).toBeCloseTo(0.743, 3)
    expect(krippendorffAlpha(data, { level: 'interval' })).toBeCloseTo(0.849, 3)
  })
  it('reproduces Shrout and Fleiss', () => {
    const r = [
      [9, 2, 5, 8],
      [6, 1, 3, 2],
      [8, 4, 6, 8],
      [7, 1, 2, 6],
      [10, 5, 6, 9],
      [6, 2, 4, 7],
    ]
    expect(intraclassCorrelation(r, { form: 'ICC1' })).toBeCloseTo(0.17, 2)
    expect(intraclassCorrelation(r, { form: 'ICC2' })).toBeCloseTo(0.29, 2)
    expect(intraclassCorrelation(r, { form: 'ICC3' })).toBeCloseTo(0.71, 2)
    expect(intraclassCorrelation(r, { form: 'ICC3k' })).toBeCloseTo(0.91, 2)
  })
})

describe('distances, detection and segmentation', () => {
  it('reproduces the distance examples', () => {
    expect(minkowskiDistance([1, 2, 3], [4, 0, 3], { p: 3 })).toBeCloseTo(3.271, 3)
    expect(cosineSimilarity([1, 2, 3], [4, 0, 3])).toBeCloseTo(0.695, 3)
    const h = hausdorffDistances(
      [
        [0, 0],
        [1, 0],
        [2, 0],
      ],
      [
        [0, 1],
        [1, 1],
        [2, 1],
        [5, 1],
      ],
    )
    expect(h.directedXY).toBeCloseTo(1, 12)
    expect(h.hausdorff).toBeCloseTo(Math.sqrt(10), 12)
    const square = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 2],
    ]
    const moved = square.map(([x, y]) => [
      2 * (x * Math.cos(0.5) - y * Math.sin(0.5)) + 3,
      2 * (x * Math.sin(0.5) + y * Math.cos(0.5)) - 1,
    ])
    expect(procrustesDisparity(square, moved)).toBeCloseTo(0, 10)
  })
  it('reproduces the box and AP examples', () => {
    const o = boxOverlap([2, 2, 6, 6], [0, 0, 4, 4])
    expect(o.iou).toBeCloseTo(4 / 28, 12)
    expect(o.giou).toBeCloseTo(-0.079, 3)
    expect(o.diou).toBeCloseTo(0.032, 3)
    const hits = [true, false, true, true, false]
    expect(averagePrecisionFromMatches(hits, 4).averagePrecision).toBeCloseTo(0.625, 12)
    expect(averagePrecisionFromMatches(hits, 4, '11-point').averagePrecision).toBeCloseTo(0.614, 3)
    expect(averagePrecisionFromMatches(hits, 4, '101-point').averagePrecision).toBeCloseTo(0.629, 3)
    const pq = panopticFromMatches([0.9, 0.7, 0.6], 1, 2)
    expect(pq.pq).toBeCloseTo(0.489, 3)
    expect(dice([1, 1, 0, 0], [1, 0, 1, 0])).toBeCloseTo(0.5, 12)
  })
})

describe('text', () => {
  it('reproduces the BLEU and chrF examples', () => {
    const ref = 'the cat is on the mat'
    const cand = 'the cat sat on the mat'
    expect(bleuScore(ref, cand).score).toBe(0)
    expect(bleuScore(ref, cand, { smoothing: 'add-one' }).score).toBeCloseTo(0.485, 3)
    const corpus = bleuScore([ref, 'there is a dog in the yard'], [cand, 'there is a dog in the garden'])
    expect(corpus.score).toBeCloseTo(0.619, 3)
    expect(bleuScore(ref, 'the cat is on').score).toBeCloseTo(0.607, 3)
    const c = chrFScore(ref, cand)
    expect(c.precision).toBeCloseTo(0.611, 3)
    expect(c.recall).toBeCloseTo(0.655, 3)
    expect(c.score).toBeCloseTo(0.646, 3)
  })
  it('reproduces ROUGE, WER, TER and SQuAD', () => {
    expect(rougeNScores('the cat is on the mat', 'the cat sat on the mat', { n: 2 }).f).toBeCloseTo(0.6, 12)
    expect(rougeLScores('police killed the gunman', 'the gunman killed police').f).toBeCloseTo(0.5, 12)
    const ref = 'the quick brown fox jumps over the lazy dog'
    const hyp = 'the quick brown fox jumped over a lazy dog today'
    expect(wordErrorRate(ref, hyp)).toBeCloseTo(1 / 3, 12)
    expect(editAlignment(ref.split(' '), hyp.split(' ')).insertions).toBe(1)
    expect(translationEditRate('the cat sat on the mat', 'on the mat the cat sat')).toBeCloseTo(1 / 6, 12)
    expect(squadExactMatch([['the Eiffel Tower', 'Eiffel Tower in Paris']], ['Eiffel Tower, Paris'])).toBe(0)
    expect(squadF1([['the Eiffel Tower', 'Eiffel Tower in Paris']], ['Eiffel Tower, Paris'])).toBeCloseTo(0.857, 3)
  })
})

describe('image, audio and generative', () => {
  it('reproduces SNR, SI-SDR and PIT', () => {
    expect(snr([1, 2, 3, 4], [2.5, 3.5, 6.5, 7.5])).toBeCloseTo(0.147, 3)
    expect(siSdr([1, 2, 3, 4], [2.5, 3.5, 6.5, 7.5])).toBeCloseTo(20.8, 1)
    const r = permutationInvariantScore(
      [
        [1, 2, 3, 4],
        [4, -1, 2, 0],
      ],
      [
        [4, -1, 2, 0.1],
        [1, 2, 3, 4.1],
      ],
    )
    expect(r.permutation).toEqual([1, 0])
  })
  it('reproduces PSNR, SSIM and the remote-sensing example', () => {
    const a = Array.from({ length: 64 }, (_, i) => (i * 37) % 255)
    const b = a.map((v) => v + 10)
    expect(psnr(a, b, { dataRange: 255 })).toBeCloseTo(10 * Math.log10(255 ** 2 / 100), 10)
    expect(ssimMap(a, a, { dataRange: 255, width: 8 }).mean).toBeCloseTo(1, 12)
    const ref = [
      [100, 80, 60],
      [90, 85, 70],
    ]
    const est = [
      [98, 84, 58],
      [95, 80, 72],
    ]
    expect((spectralAngle(ref, est) * 180) / Math.PI).toBeCloseTo(2.44, 2)
    expect(ergas(ref, est, { ratio: 0.25 })).toBeCloseTo(1.08, 2)
  })
  it('reproduces FID, KID, IS and precision/recall', () => {
    expect(
      frechetDistance(
        [0, 0],
        [
          [1, 0.5],
          [0.5, 1],
        ],
        [1, 0.5],
        [
          [2, 0],
          [0, 0.5],
        ],
      ).distance,
    ).toBeCloseTo(1.636, 3)
    expect(kid([[0], [1]], [[1], [2]])).toBeCloseTo(9.5, 12)
    const p = [
      [0.9, 0.05, 0.05],
      [0.05, 0.9, 0.05],
      [0.05, 0.05, 0.9],
    ]
    expect(inceptionScore(p)).toBeCloseTo(2.02, 2)
    const pr = generativePrecisionRecall(
      [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ],
      [
        [0.1, 0.1],
        [0.9, 0.1],
        [3, 3],
      ],
      { k: 1 },
    )
    expect(pr.precision).toBeCloseTo(2 / 3, 12)
    expect(pr.recall).toBeCloseTo(3 / 4, 12)
  })
})

describe('fairness and uncertainty', () => {
  it('reproduces the two-group example', () => {
    const yTrue: number[] = []
    const yPred: number[] = []
    const groups: string[] = []
    const add = (g: string, tp: number, fn: number, fp: number, tn: number) => {
      for (const [t, p, n] of [
        [1, 1, tp],
        [1, 0, fn],
        [0, 1, fp],
        [0, 0, tn],
      ])
        for (let i = 0; i < n; i++) {
          yTrue.push(t)
          yPred.push(p)
          groups.push(g)
        }
    }
    add('A', 30, 10, 10, 50)
    add('B', 12, 8, 8, 72)
    expect(demographicParityDifference(yTrue, yPred, { groups })).toBeCloseTo(0.2, 12)
    expect(equalisedOddsDifference(yTrue, yPred, { groups })).toBeCloseTo(0.15, 12)
  })
  it('reproduces the Wilson intervals and bootstraps deterministically', () => {
    const [lo, hi] = wilsonInterval(900, 1000)
    expect(lo).toBeCloseTo(0.88, 3)
    expect(hi).toBeCloseTo(0.917, 3)
    const { yTrue, yPred } = screening()
    const a = bootstrapMetric(stream(1), (t: number[], p: number[]) => f1(t, p), yTrue, yPred, { resamples: 200 })
    const b = bootstrapMetric(stream(1), (t: number[], p: number[]) => f1(t, p), yTrue, yPred, { resamples: 200 })
    expect(a.interval).toEqual(b.interval)
    expect(a.interval[0]).toBeLessThan(0.444)
    expect(a.interval[1]).toBeGreaterThan(0.444)
  })
})

describe('registry', () => {
  it('lists every metric with metadata', () => {
    const all = Object.values(metricRegistry)
    expect(all.length).toBeGreaterThan(100)
    for (const m of all) {
      expect(m.info.note).toMatch(/^[a-z0-9-]+$/)
      expect(m.info.range[0]).toBeLessThanOrEqual(m.info.range[1])
    }
    expect(getMetric('auroc')).toBe(auroc)
    expect(listMetrics({ capability: 'predictive' })).toContain(logLoss)
  })
})

describe('smoke: remaining families', () => {
  it('finds operating points', async () => {
    const m = await import('aifn/metrics')
    const y = [1, 1, 1, 1, 0, 0, 0, 0]
    const s = [0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.05]
    expect(m.equalErrorRate(y, s).rate).toBeCloseTo(0.25, 12)
    expect(m.youdenPoint(y, s).j).toBeCloseTo(0.75, 12)
    expect(m.operatingPoint(y, s, { maxFpr: 0 }).tpr).toBeCloseTo(0.75, 12)
    expect(m.costCurve(y, s).area).toBeGreaterThan(0)
    expect(toFlat(m.gainCurve(y, s).gain).at(-1)).toBe(1)
  })
  it('recovers a Procrustes rotation', async () => {
    const m = await import('aifn/metrics')
    const a = Math.PI / 6
    const x = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 2],
    ]
    const y2 = x.map(([u, v]) => [u * Math.cos(a) - v * Math.sin(a), u * Math.sin(a) + v * Math.cos(a)])
    const r = toFlat(m.orthogonalProcrustes(x, y2).rotation)
    expect(Math.atan2(r[2], r[0])).toBeCloseTo(a, 10)
  })
  it('evaluates detections and segments', async () => {
    const m = await import('aifn/metrics')
    const gt = [
      { image: 0, box: [0, 0, 2, 2] as const },
      { image: 0, box: [5, 5, 7, 7] as const },
    ]
    const det = [
      { image: 0, box: [0, 0, 2, 2] as const, score: 0.9 },
      { image: 0, box: [0, 0, 2, 2] as const, score: 0.8 },
      { image: 0, box: [5, 5, 7, 7] as const, score: 0.7 },
    ]
    expect(m.matchDetections(gt, det).map((d) => d.outcome)).toEqual(['tp', 'fp', 'tp'])
    expect(m.detectionMeanAveragePrecision(gt, det, { iouThresholds: [0.5], interpolation: 'all-points' })).toBeCloseTo(
      0.5 + 0.5 * (2 / 3),
      12,
    )
    const truth = [1, 1, 2, 2, 0, 0]
    const pred = [5, 5, 6, 6, 6, 0]
    const pq = m.panopticQuality(truth, pred)
    // 1↔5 (IoU 1) and 2↔6 (IoU 2/3) match; truth 0 and prediction 0 overlap at IoU 0.5, not above it.
    expect(pq.rq).toBeCloseTo(2 / 3, 12)
    expect(pq.pq).toBeCloseTo((1 + 2 / 3) / 3, 12)
    const mask = [
      [0, 0, 0, 0],
      [0, 1, 1, 0],
      [0, 1, 1, 0],
      [0, 0, 0, 0],
    ]
    expect(m.boundaryF1(mask, mask)).toBe(1)
    expect(m.meanIou([0, 0, 1, 1], [0, 1, 1, 1])).toBeCloseTo((0.5 + 2 / 3) / 2, 12)
  })
  it('handles text embeddings, PIT values and paired tests', async () => {
    const m = await import('aifn/metrics')
    const e = [
      [1, 0],
      [0, 1],
    ]
    expect(m.bertScore(e, e).f).toBeCloseTo(1, 12)
    expect(toFlat(m.pitValues([0], { mean: 0, sd: 1 }))[0]).toBeCloseTo(0.5, 12)
    const y = [1, 0, 1, 0, 1, 0]
    const s = [0.9, 0.2, 0.6, 0.7, 0.8, 0.1]
    expect(m.delongTest(y, s, s).difference).toBe(0)
    const bars = m.consistencyBars(stream(3), [0.1, 0.5, 0.9, 0.95], { bins: 2, resamples: 50 })
    expect(bars.lower.shape).toEqual([2])
    expect(
      m.backretrieval(
        [
          [1, 0],
          [0, 1],
        ],
        [
          [1, 0.2],
          [0.1, 1],
        ],
        { k: 1 },
      ),
    ).toBe(1)
  })
})
