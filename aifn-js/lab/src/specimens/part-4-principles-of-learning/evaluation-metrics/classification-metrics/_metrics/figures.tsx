import {
  adjustedMutualInformation,
  adjustedRandIndex,
  aurocDeLong,
  averagePrecision,
  bootstrapMetric,
  auroc,
  brierDecomposition,
  costCurve,
  dcg,
  debiasedSquaredCalibrationError,
  detCurve,
  expectedCalibrationError,
  expectedReciprocalRank,
  fowlkesMallows,
  gainCurve,
  logLoss,
  maximumCalibrationError,
  meanAveragePrecision,
  meanReciprocalRank,
  ndcg,
  normalisedMutualInformation,
  precisionAtK,
  precisionRecallCurve,
  precisionRecallGainCurve,
  randIndex,
  reliabilityDiagram,
  rmsCalibrationError,
  rocCurve,
  silhouetteScore,
  vMeasure,
} from 'aifn/learning/metrics'
import {
  bleuScore,
  characterErrorRate,
  chrFScore,
  rougeLScores,
  rougeNScores,
  translationEditRate,
  wordErrorRate,
} from 'aifn-applied/evaluation/text'
import { integers, normals, stream, uniform } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Figure } from '@lab/layout'
import { choice, row, setting, slider, useFigureState } from '@lab/state'
import { Input } from '@lab/ui/input'
import { Bars, Curve as CurveLayer, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { CurvePanel, formatValue } from '@lab/views'
import type { Curve } from 'aifn/foundation/contracts'

/** n standard normal draws from a keyed stream, as numbers. */
const gaussians = (key: string, n: number) => toFlat(normals(stream(key), n))

/**
 * A binormal sample: n cases, a fraction `prevalence` positive; negative scores ~ N(0, 1), positive ~ N(d, 1). The
 * same seed gives the same cases whatever the separation, so moving d moves the scores smoothly.
 */
function binormalSample(n: number, prevalence: number, separation: number) {
  const z = gaussians('metrics/binormal/z', n)
  const u = stream('metrics/binormal/u')
  const y = z.map(() => (uniform(u) < prevalence ? 1 : 0))
  return { y, scores: z.map((v, i) => v + separation * y[i]) }
}

const fmt = formatValue

// ---------------------------------------------------------------------------------------------------------------------

const CURVE_KINDS = [
  { value: 'roc', label: 'ROC' },
  { value: 'pr', label: 'precision–recall' },
  { value: 'det', label: 'DET (probit axes)' },
  { value: 'gain', label: 'cumulative gain' },
  { value: 'cost', label: 'cost curve' },
  { value: 'prg', label: 'precision–recall–gain' },
] as const

export function CurveGallerySpecimen() {
  const state = useFigureState({
    data: row('1 · scores', {
      separation: slider(0, 4, 1.2, { label: 'separation d', step: 0.1 }),
      prevalence: slider(0.02, 0.9, 0.3, { label: 'prevalence π', step: 0.01 }),
    }),
    view: row('2 · curve', { kind: choice(CURVE_KINDS, 'roc', { label: 'curve' }) }),
  })
  const { separation, prevalence } = state.data
  const kind = state.view.kind
  const { y, scores } = useMemo(() => binormalSample(400, prevalence, separation), [prevalence, separation])
  // A weaker second model on the same cases, for comparison.
  const weaker = useMemo(() => {
    const z = gaussians('metrics/gallery/noise', scores.length)
    return scores.map((v, i) => v + 1.2 * z[i])
  }, [scores])
  const curves = useMemo(() => {
    const make = (s: number[]): Curve => {
      switch (kind) {
        case 'roc':
          return rocCurve(y, s)
        case 'pr':
          return precisionRecallCurve(y, s)
        case 'det':
          return detCurve(y, s)
        case 'gain':
          return gainCurve(y, s)
        case 'cost':
          return costCurve(y, s)
        case 'prg':
          return precisionRecallGainCurve(y, s)
      }
    }
    return [
      { name: 'model A', curve: make(scores) },
      { name: 'model B (noisier)', curve: make(weaker) },
    ]
  }, [kind, y, scores, weaker])
  const uncertainty = useMemo(() => {
    const d = aurocDeLong(y, scores)
    const b = bootstrapMetric(
      stream('metrics/gallery/bootstrap'),
      (t: number[], s: number[]) => auroc(t, s),
      y,
      scores,
      {
        resamples: 400,
      },
    )
    return { d, b }
  }, [y, scores])
  return (
    <Figure
      purpose="One sweep over the sorted scores gives every curve; the same two classifiers look different on each, and a noisier model is dominated on all of them."
      title="Curves of two scoring classifiers"
      state={state}
      readouts={{
        'model A': (
          <>
            <Readout
              label="AUROC A, DeLong 95%"
              value={`${fmt(uncertainty.d.auroc)} [${fmt(uncertainty.d.interval[0])}, ${fmt(uncertainty.d.interval[1])}]`}
            />
            <Readout
              label="bootstrap 95%"
              value={`[${fmt(uncertainty.b.interval[0])}, ${fmt(uncertainty.b.interval[1])}]`}
            />
            <Readout label="AP A" value={fmt(averagePrecision(y, scores))} />
          </>
        ),
      }}
      caption="Every curve comes from one sweep over the sorted scores. Model B adds noise to model A's scores, so it lies below A on the ROC, PR and gain curves and above it on the DET and cost curves. Lower the prevalence: the ROC curves do not move, the PR curves fall."
    >
      <CurvePanel curve={curves} />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function ReliabilitySpecimen() {
  const state = useFigureState({
    model: row('1 · model', { temperature: slider(0.2, 3, 0.6, { label: 'temperature T', step: 0.05 }) }),
    binning: row('2 · binning', {
      bins: slider(2, 30, 10, { label: 'bins M', step: 1 }),
      strategy: choice(
        [
          { value: 'uniform', label: 'equal width' },
          { value: 'quantile', label: 'equal mass' },
        ],
        'uniform',
        { label: 'binning' },
      ),
    }),
  })
  const { temperature } = state.model
  const { bins, strategy } = state.binning
  const n = 2000
  // True probabilities q from a logistic of a Gaussian score; labels ~ Bernoulli(q); the model reports
  // σ(logit(q)/T): T < 1 is overconfident, T > 1 underconfident.
  const { y, p } = useMemo(() => {
    const z = gaussians('metrics/reliability/z', n)
    const u = stream('metrics/reliability/u')
    const q = z.map((v) => 1 / (1 + Math.exp(-1.5 * v)))
    const y = q.map((v) => (uniform(u) < v ? 1 : 0))
    const p = q.map((v) => 1 / (1 + Math.exp(-Math.log(v / (1 - v)) / temperature)))
    return { y, p }
  }, [temperature])
  const diagram = useMemo(() => reliabilityDiagram(y, p, { bins, strategy }), [y, p, bins, strategy])
  const brier = useMemo(() => brierDecomposition(y, p, { bins }), [y, p, bins])
  return (
    <Figure
      purpose="A calibrated model's predicted probabilities match the observed frequencies, so its binned points lie on the diagonal; a temperature below 1 makes it overconfident and the curve flattens."
      title="Reliability diagram of a mis-scaled model"
      state={state}
      readouts={{
        calibration: (
          <>
            <Readout label="MCE" value={fmt(maximumCalibrationError(y, p, { bins, strategy }))} />
            <Readout label="RMS CE" value={fmt(rmsCalibrationError(y, p, { bins, strategy }))} />
            <Readout label="debiased CE²" value={fmt(debiasedSquaredCalibrationError(y, p, { bins, strategy }))} />
            <Readout label="ECE (check)" value={fmt(expectedCalibrationError(y, p, { bins, strategy }))} />
          </>
        ),
        'Brier decomposition': (
          <>
            <Readout label="log loss" value={fmt(logLoss(y, p))} />
            <Readout label="Brier" value={fmt(brier.brier)} />
            <Readout label="reliability" value={fmt(brier.reliability)} />
            <Readout label="resolution" value={fmt(brier.resolution)} />
            <Readout label="uncertainty" value={fmt(brier.uncertainty)} />
          </>
        ),
      }}
      caption="At T = 1 the model is calibrated and the points follow the diagonal up to sampling noise. T < 1 pushes probabilities towards 0 and 1 (overconfident, a curve flatter than the diagonal); T > 1 does the reverse. Only the reliability term of the Brier score changes with T; resolution depends on the ranking."
    >
      <CurvePanel curve={diagram} />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

/** Ten items with graded relevance 0–3, in their ideal order. */
const GRADES = [3, 3, 2, 2, 2, 1, 1, 0, 0, 0]

export function RankingSpecimen() {
  const state = useFigureState({
    ranking: row('1 · ranking', { noise: slider(0, 4, 2, { label: 'noise sd', step: 0.1 }) }),
    metric: row('2 · metric', {
      exponential: setting(true, 'exponential gain 2^g − 1'),
    }),
    k: slider(1, 10, 5, { label: 'cut-off k', step: 1, onChart: true }),
  })
  const { noise } = state.ranking
  const { exponential } = state.metric
  const k = Math.round(state.k)
  const gain = exponential ? 'exponential' : 'linear'
  // Scores are the grade plus noise; ranking by score gives the list the metrics read.
  const ranked = useMemo(() => {
    const z = gaussians('metrics/ranking/noise', GRADES.length)
    const scores = GRADES.map((g, i) => g + noise * z[i])
    const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a])
    return order.map((i) => GRADES[i])
  }, [noise])
  const bars = useMemo(() => {
    const g = (x: number) => (exponential ? 2 ** x - 1 : x)
    return {
      ranks: ranked.map((_, i) => i + 1),
      ideal: GRADES.map((x, i) => g(x) / Math.log2(i + 2)),
      actual: ranked.map((x, i) => g(x) / Math.log2(i + 2)),
    }
  }, [ranked, exponential])
  const rank = useAxis({ label: 'rank i', range: [0.5, 10.5] })
  const gainAxis = useAxis({ label: 'discounted gain', hold: 'union', key: exponential })
  return (
    <Figure
      title="nDCG of a noisy ranking"
      purpose="nDCG@k is the discounted gain of the top k ranks divided by the ideal ordering's: a swap near the top costs more than one near the bottom, because the discount falls with rank."
      state={state}
      readouts={{
        [`at k = ${k}`]: (
          <>
            <Readout label="ranked grades" value={ranked.join(' ')} />
            <Readout label={`nDCG@${k}`} value={fmt(ndcg(ranked, { k, gain }))} />
            <Readout label={`DCG@${k}`} value={fmt(dcg(ranked, { k, gain }))} />
            <Readout label={`P@${k}`} value={fmt(precisionAtK(ranked, { k }))} />
            <Readout label="AP" value={fmt(meanAveragePrecision(ranked))} />
            <Readout label="RR" value={fmt(meanReciprocalRank(ranked))} />
            <Readout label={`ERR@${k}`} value={fmt(expectedReciprocalRank(ranked, { k, maxGrade: 3 }))} />
          </>
        ),
      }}
      caption="Ten items with grades 0–3 ranked by grade plus Gaussian noise. Bars are each rank's discounted gain; the dashed line is the ideal ordering's. With no noise the ranking is ideal and nDCG is 1. The exponential gain weights the grade-3 items more. Drag the cut-off line to set k."
    >
      <Plot x={rank} y={gainAxis}>
        <Bars name="gain / log₂(i + 1)" x={bars.ranks} y={bars.actual} slot={0} />
        <CurveLayer name="ideal gain / log₂(i + 1)" x={bars.ranks} y={bars.ideal} muted dashed showPoints />
        <Handle kind="x" at={k + 0.5} label="cut-off k" onDrag={(v) => state.set('k', Math.round(v - 0.5))} />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

/** Three Gaussian blobs of 60 points each. */
function blobs() {
  const z = gaussians('metrics/clusters/xy', 360)
  const centres = [
    [0, 0],
    [4, 0.5],
    [2, 3.5],
  ]
  const x: number[][] = []
  const labels: number[] = []
  for (let c = 0; c < 3; c++)
    for (let i = 0; i < 60; i++) {
      const k = c * 60 + i
      x.push([centres[c][0] + 0.8 * z[2 * k], centres[c][1] + 0.8 * z[2 * k + 1]])
      labels.push(c)
    }
  return { x, labels }
}

/** Reassign a fraction of labels at random (the same cases in the same order as the fraction grows). */
function corrupt(labels: number[], fraction: number, clusters: number): number[] {
  const u = stream('metrics/clusters/corrupt')
  const order = labels.map((_, i) => ({ i, key: uniform(u), to: integers(u, clusters) })).sort((a, b) => a.key - b.key)
  const out = [...labels]
  for (let r = 0; r < Math.round(fraction * labels.length); r++) out[order[r].i] = order[r].to
  return out
}

export function ClusteringSpecimen() {
  const state = useFigureState({ fraction: slider(0, 1, 0.3, { label: 'fraction reassigned', step: 0.05 }) })
  const fraction = state.fraction
  const data = useMemo(() => blobs(), [])
  const predicted = useMemo(() => corrupt(data.labels, fraction, 3), [data, fraction])
  const sweep = useMemo(() => {
    const fs = Array.from({ length: 21 }, (_, i) => i / 20)
    const rows = fs.map((f) => {
      const p = corrupt(data.labels, f, 3)
      return {
        ari: adjustedRandIndex(data.labels, p),
        ami: adjustedMutualInformation(data.labels, p),
        nmi: normalisedMutualInformation(data.labels, p),
        rand: randIndex(data.labels, p),
        fmi: fowlkesMallows(data.labels, p),
        silhouette: silhouetteScore(data.x, p),
      }
    })
    const line = (name: string, key: keyof (typeof rows)[number], slot: number) => ({
      name,
      x: fs,
      y: rows.map((r) => r[key]),
      slot,
    })
    return [
      line('ARI', 'ari', 0),
      line('AMI', 'ami', 1),
      line('NMI', 'nmi', 2),
      line('Rand', 'rand', 3),
      line('Fowlkes–Mallows', 'fmi', 4),
      line('silhouette', 'silhouette', 5),
    ]
  }, [data])
  const xy = useMemo(() => ({ x: data.x.map((p) => p[0]), y: data.x.map((p) => p[1]) }), [data])
  const x1 = useAxis({ label: 'x₁' })
  const x2 = useAxis({ label: 'x₂', equal: x1 })
  const fx = useAxis({ label: 'fraction reassigned', range: [0, 1] })
  const sc = useAxis({ label: 'score' })
  return (
    <Figure
      title="Clustering scores for a degrading clustering"
      purpose="Chance-corrected scores (ARI, AMI) fall to 0 as a clustering's labels become random, while the plain Rand index stays high: most pairs are apart in any three-cluster split."
      state={state}
      readouts={{
        'at this fraction': (
          <>
            <Readout label="ARI" value={fmt(adjustedRandIndex(data.labels, predicted))} />
            <Readout label="AMI" value={fmt(adjustedMutualInformation(data.labels, predicted))} />
            <Readout label="V-measure" value={fmt(vMeasure(data.labels, predicted))} />
            <Readout label="Rand" value={fmt(randIndex(data.labels, predicted))} />
            <Readout label="silhouette" value={fmt(silhouetteScore(data.x, predicted))} />
          </>
        ),
      }}
      caption="Three blobs, with a growing fraction of points reassigned to a random cluster (left, coloured by the corrupted labels). Drag the marker line on the right. The chance-corrected scores (ARI, AMI) fall to about 0 once the labels are random; the Rand index stays well above 0 because most pairs are apart in any clustering with three clusters."
    >
      <Plots cols={2}>
        <Plot x={x1} y={x2} legend={false}>
          <Points
            name="points"
            x={xy.x}
            y={xy.y}
            group={predicted}
            groupNames={['cluster 0', 'cluster 1', 'cluster 2']}
          />
        </Plot>
        <Plot x={fx} y={sc}>
          {sweep.map((l) => (
            <CurveLayer key={l.name} name={l.name} x={l.x} y={l.y} slot={l.slot} />
          ))}
          <Handle {...state.handle('fraction', { label: 'now' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function TextSpecimen() {
  const [reference, setReference] = useState('the cat is on the mat')
  const [candidate, setCandidate] = useState('the cat sat on the mat')
  const scores = useMemo(() => {
    const b = bleuScore(reference, candidate)
    const smoothed = bleuScore(reference, candidate, { smoothing: 'add-one' })
    return {
      bleu: b.score,
      smoothed: smoothed.score,
      precisions: b.precisions,
      chrf: chrFScore(reference, candidate).score,
      rouge1: rougeNScores(reference, candidate, { n: 1 }).f,
      rouge2: rougeNScores(reference, candidate, { n: 2 }).f,
      rougeL: rougeLScores(reference, candidate).f,
      wer: wordErrorRate(reference, candidate),
      cer: characterErrorRate(reference, candidate),
      ter: translationEditRate(reference, candidate),
    }
  }, [reference, candidate])
  const orders = useAxis({ label: 'n-gram order n', categories: ['1', '2', '3', '4'] })
  const pn = useAxis({ label: 'pₙ', range: [0, 1] })
  return (
    <Figure
      title="Text metrics of one sentence pair"
      purpose="BLEU multiplies clipped n-gram precisions up to n = 4, so one missing 4-gram zeroes a sentence's BLEU while character and recall-based scores give partial credit."
      controls={
        <>
          <label className="flex flex-col gap-1.5 text-xs">
            <span className="text-muted-foreground">reference</span>
            <Input value={reference} onChange={(e) => setReference(e.target.value)} />
          </label>
          <label className="flex flex-col gap-1.5 text-xs">
            <span className="text-muted-foreground">candidate</span>
            <Input value={candidate} onChange={(e) => setCandidate(e.target.value)} />
          </label>
        </>
      }
      readouts={{
        overlap: (
          <>
            <Readout label="BLEU" value={fmt(scores.bleu)} />
            <Readout label="BLEU (add-one)" value={fmt(scores.smoothed)} />
            <Readout label="chrF₂" value={fmt(scores.chrf)} />
            <Readout label="ROUGE-1" value={fmt(scores.rouge1)} />
            <Readout label="ROUGE-2" value={fmt(scores.rouge2)} />
            <Readout label="ROUGE-L" value={fmt(scores.rougeL)} />
          </>
        ),
        'edit rates': (
          <>
            <Readout label="WER" value={fmt(scores.wer)} />
            <Readout label="CER" value={fmt(scores.cer)} />
            <Readout label="TER" value={fmt(scores.ter)} />
          </>
        ),
      }}
      caption="Tokens are whitespace-separated words; chrF uses characters with spaces removed. One missing 4-gram makes sentence BLEU 0 while chrF and ROUGE give partial credit. Try reordering the candidate: TER charges one shift for a moved block, WER charges every word."
    >
      <Plot x={orders} y={pn}>
        <Bars name="clipped n-gram precision" x={[0, 1, 2, 3]} y={scores.precisions} slot={0} />
      </Plot>
    </Figure>
  )
}
