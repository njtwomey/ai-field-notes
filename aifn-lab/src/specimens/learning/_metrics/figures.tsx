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
import { Select, Slider, Switch } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Input } from '@lab/ui/input'
import { Readout, XYChart, type XYSeries } from '@lab/viz'
import { CurveView, formatValue, type Curve } from '@lab/views'

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

type CurveKind = 'roc' | 'precision-recall' | 'det' | 'gain' | 'cost' | 'precision-recall-gain'

export function CurveGallerySpecimen() {
  const [kind, setKind] = useState<CurveKind>('roc')
  const [separation, setSeparation] = useState(1.2)
  const [prevalence, setPrevalence] = useState(0.3)
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
        case 'precision-recall':
          return precisionRecallCurve(y, s)
        case 'det':
          return detCurve(y, s)
        case 'gain':
          return gainCurve(y, s)
        case 'cost':
          return costCurve(y, s)
        case 'precision-recall-gain':
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
    <CurveView
      title="Curves of two scoring classifiers"
      curve={curves}
      controls={
        <>
          <Select
            label="curve"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'roc', label: 'ROC' },
              { value: 'precision-recall', label: 'precision–recall' },
              { value: 'det', label: 'DET (probit axes)' },
              { value: 'gain', label: 'cumulative gain' },
              { value: 'cost', label: 'cost curve' },
              { value: 'precision-recall-gain', label: 'precision–recall–gain' },
            ]}
          />
          <Slider label="separation d" value={separation} min={0} max={4} step={0.1} onChange={setSeparation} />
          <Slider label="prevalence π" value={prevalence} min={0.02} max={0.9} step={0.01} onChange={setPrevalence} />
        </>
      }
      readouts={
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
      }
      caption="Every curve comes from one sweep over the sorted scores. Model B adds noise to model A's scores, so it lies below A on the ROC, PR and gain curves and above it on the DET and cost curves."
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------

export function ReliabilitySpecimen() {
  const [temperature, setTemperature] = useState(0.6)
  const [bins, setBins] = useState(10)
  const [strategy, setStrategy] = useState<'uniform' | 'quantile'>('uniform')
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
    <CurveView
      title="Reliability diagram of a mis-scaled model"
      curve={diagram}
      controls={
        <>
          <Slider label="temperature T" value={temperature} min={0.2} max={3} step={0.05} onChange={setTemperature} />
          <Slider label="bins M" value={bins} min={2} max={30} step={1} onChange={setBins} />
          <Select
            label="binning"
            value={strategy}
            onChange={setStrategy}
            options={[
              { value: 'uniform', label: 'equal width' },
              { value: 'quantile', label: 'equal mass' },
            ]}
          />
        </>
      }
      readouts={
        <>
          <Readout label="MCE" value={fmt(maximumCalibrationError(y, p, { bins, strategy }))} />
          <Readout label="RMS CE" value={fmt(rmsCalibrationError(y, p, { bins, strategy }))} />
          <Readout label="debiased CE²" value={fmt(debiasedSquaredCalibrationError(y, p, { bins, strategy }))} />
          <Readout label="ECE (check)" value={fmt(expectedCalibrationError(y, p, { bins, strategy }))} />
          <Readout label="log loss" value={fmt(logLoss(y, p))} />
          <Readout label="Brier" value={fmt(brier.brier)} />
          <Readout label="reliability" value={fmt(brier.reliability)} />
          <Readout label="resolution" value={fmt(brier.resolution)} />
          <Readout label="uncertainty" value={fmt(brier.uncertainty)} />
        </>
      }
      caption="At T = 1 the model is calibrated and the points follow the diagonal up to sampling noise. T < 1 pushes probabilities towards 0 and 1 (overconfident, a curve flatter than the diagonal); T > 1 does the reverse. Only the reliability term of the Brier score changes with T; resolution depends on the ranking."
    />
  )
}

// ---------------------------------------------------------------------------------------------------------------------

/** Ten items with graded relevance 0–3, in their ideal order. */
const GRADES = [3, 3, 2, 2, 2, 1, 1, 0, 0, 0]

export function RankingSpecimen() {
  const [noise, setNoise] = useState(1)
  const [k, setK] = useState(5)
  const [exponential, setExponential] = useState(true)
  const gain = exponential ? 'exponential' : 'linear'
  // Scores are the grade plus noise; ranking by score gives the list the metrics read.
  const ranked = useMemo(() => {
    const z = gaussians('metrics/ranking/noise', GRADES.length)
    const scores = GRADES.map((g, i) => g + noise * z[i])
    const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a])
    return order.map((i) => GRADES[i])
  }, [noise])
  const series = useMemo((): XYSeries[] => {
    const g = (x: number) => (exponential ? 2 ** x - 1 : x)
    const ranks = ranked.map((_, i) => i + 1)
    return [
      {
        name: 'ideal gain / log₂(i + 1)',
        type: 'line',
        x: ranks,
        y: GRADES.map((x, i) => g(x) / Math.log2(i + 2)),
        muted: true,
        dashed: true,
      },
      { name: 'gain / log₂(i + 1)', type: 'bar', x: ranks, y: ranked.map((x, i) => g(x) / Math.log2(i + 2)), slot: 0 },
    ]
  }, [ranked, exponential])
  return (
    <Figure
      title="nDCG of a noisy ranking"
      description="Ten items with grades 0–3 are ranked by their grade plus Gaussian noise. Bars are each rank's discounted gain; the dashed line is the ideal ordering's."
      controls={
        <>
          <Slider label="noise sd" value={noise} min={0} max={4} step={0.1} onChange={setNoise} />
          <Slider label="cut-off k" value={k} min={1} max={10} step={1} onChange={setK} />
          <Switch label="exponential gain 2^g − 1" checked={exponential} onChange={setExponential} />
        </>
      }
      readouts={
        <>
          <Readout label="ranked grades" value={ranked.join(' ')} />
          <Readout label={`nDCG@${k}`} value={fmt(ndcg(ranked, { k, gain }))} />
          <Readout label={`DCG@${k}`} value={fmt(dcg(ranked, { k, gain }))} />
          <Readout label={`P@${k}`} value={fmt(precisionAtK(ranked, { k }))} />
          <Readout label="AP" value={fmt(meanAveragePrecision(ranked))} />
          <Readout label="RR" value={fmt(meanReciprocalRank(ranked))} />
          <Readout label={`ERR@${k}`} value={fmt(expectedReciprocalRank(ranked, { k, maxGrade: 3 }))} />
        </>
      }
      caption="With no noise the ranking is ideal and nDCG is 1. Noise swaps items; a swap near the top costs more than one near the bottom, because the discount 1/log₂(i + 1) falls with rank. The exponential gain weights the grade-3 items more."
    >
      <XYChart series={series} xLabel="rank i" yLabel="discounted gain" integerX />
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
  const [fraction, setFraction] = useState(0.3)
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
    const line = (name: string, key: keyof (typeof rows)[number], slot: number): XYSeries => ({
      name,
      type: 'line',
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
  const scatter = useMemo(
    (): XYSeries[] => [
      {
        name: 'points',
        type: 'scatter',
        x: data.x.map((p) => p[0]),
        y: data.x.map((p) => p[1]),
        group: predicted,
        groupNames: ['cluster 0', 'cluster 1', 'cluster 2'],
      },
    ],
    [data, predicted],
  )
  return (
    <Figure
      title="Clustering scores for a degrading clustering"
      description="Three blobs, with a growing fraction of points reassigned to a random cluster."
      controls={
        <Slider label="fraction reassigned" value={fraction} min={0} max={1} step={0.05} onChange={setFraction} />
      }
      readouts={
        <>
          <Readout label="ARI" value={fmt(adjustedRandIndex(data.labels, predicted))} />
          <Readout label="AMI" value={fmt(adjustedMutualInformation(data.labels, predicted))} />
          <Readout label="V-measure" value={fmt(vMeasure(data.labels, predicted))} />
          <Readout label="Rand" value={fmt(randIndex(data.labels, predicted))} />
          <Readout label="silhouette" value={fmt(silhouetteScore(data.x, predicted))} />
        </>
      }
      caption="Drag the marker line on the right. The chance-corrected scores (ARI, AMI) fall to about 0 once the labels are random; the Rand index stays well above 0 because most pairs are apart in any clustering with three clusters."
    >
      <div className="grid h-full grid-cols-1 gap-3 md:grid-cols-2">
        <XYChart series={scatter} xLabel="x₁" yLabel="x₂" equalAspect />
        <XYChart
          series={sweep}
          xLabel="fraction reassigned"
          yLabel="score"
          xRange={[0, 1]}
          handles={[
            {
              kind: 'x',
              at: fraction,
              label: 'now',
              onDrag: (v) => setFraction(Math.round(Math.min(1, Math.max(0, v)) * 20) / 20),
            },
          ]}
        />
      </div>
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
  const series = useMemo(
    (): XYSeries[] => [
      { name: 'clipped n-gram precision', type: 'bar', x: [1, 2, 3, 4], y: scores.precisions, slot: 0 },
    ],
    [scores],
  )
  return (
    <Figure
      title="Text metrics of one sentence pair"
      description="Tokens are whitespace-separated words; chrF uses characters with spaces removed."
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
      readouts={
        <>
          <Readout label="BLEU" value={fmt(scores.bleu)} />
          <Readout label="BLEU (add-one)" value={fmt(scores.smoothed)} />
          <Readout label="chrF₂" value={fmt(scores.chrf)} />
          <Readout label="ROUGE-1" value={fmt(scores.rouge1)} />
          <Readout label="ROUGE-2" value={fmt(scores.rouge2)} />
          <Readout label="ROUGE-L" value={fmt(scores.rougeL)} />
          <Readout label="WER" value={fmt(scores.wer)} />
          <Readout label="CER" value={fmt(scores.cer)} />
          <Readout label="TER" value={fmt(scores.ter)} />
        </>
      }
      caption="One missing 4-gram makes sentence BLEU 0 while chrF and ROUGE give partial credit. Try reordering the candidate: TER charges one shift for a moved block, WER charges every word."
    >
      <XYChart series={series} xLabel="n-gram order n" yLabel="pₙ" yRange={[0, 1]} integerX />
    </Figure>
  )
}
