import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { expit, logit, posterior, rates, sample, scoreForPosterior } from '../_shared/binormal'

const N = 5000
const ITERATIONS = 30
const S = linspace(-4, 6, 201)

/** Posterior re-weighted from prior a to prior b: odds multiplied by (b / (1 − b)) / (a / (1 − a)). */
const adjust = (p: number, a: number, b: number) => expit(logit(p) + logit(b) - logit(a))

/**
 * Label shift between training and deployment. A calibrated classifier for the training prior scores an unlabelled
 * deployment sample; EM re-estimates the deployment prior (Saerens et al.), and black-box shift estimation inverts the
 * classifier's confusion rates (Lipton et al.).
 */
export function PriorShiftEm() {
  const piTrain = useParam(0.5, { min: 0.05, max: 0.5, step: 0.01 })
  const piNew = useParam(0.1, { min: 0.01, max: 0.5, step: 0.01 })
  const d = useParam(1.5, { min: 0.5, max: 4, step: 0.1 })
  const [seed, setSeed] = useState(1)

  const result = useMemo(() => {
    const g = rng(seed)
    const { s, y } = sample(N, piNew.value, d.value, g)
    const p = s.map((v) => posterior(v, piTrain.value, d.value))
    const naive = p.reduce((a, b) => a + b, 0) / N
    // EM: E-step adjusts every posterior to the current prior estimate; M-step sets the prior to their mean.
    const trace = [piTrain.value]
    let est = piTrain.value
    for (let it = 0; it < ITERATIONS; it++) {
      let total = 0
      for (const pi of p) total += adjust(pi, piTrain.value, est)
      est = total / N
      trace.push(est)
    }
    // BBSE with hard predictions at p ≥ 0.5; the rates are those of the training population.
    const t = scoreForPosterior(0.5, piTrain.value, d.value)
    const { tpr, fpr } = rates(t, d.value)
    const predictedPositive = s.filter((v) => v >= t).length / N
    const bbse = Math.min(1, Math.max(0, (predictedPositive - fpr) / (tpr - fpr)))
    const actual = y.reduce((a, b) => a + b, 0) / N
    return { trace, est, naive, bbse, actual }
  }, [seed, piTrain.value, piNew.value, d.value])

  const traceSeries = useMemo<XYSeries[]>(() => {
    const its = result.trace.map((_, i) => i)
    const flat = (v: number) => [v, v]
    return [
      { name: 'EM estimate', type: 'line', x: its, y: result.trace, slot: 0 },
      {
        name: 'mean of unadjusted posteriors',
        type: 'line',
        x: [0, ITERATIONS],
        y: flat(result.naive),
        slot: 1,
        dashed: true,
      },
      { name: 'BBSE estimate', type: 'line', x: [0, ITERATIONS], y: flat(result.bbse), slot: 3, dashed: true },
      {
        name: 'true deployment prior',
        type: 'line',
        x: [0, ITERATIONS],
        y: flat(piNew.value),
        emphasis: true,
        dashed: true,
      },
    ]
  }, [result, piNew.value])

  const posteriorSeries = useMemo<XYSeries[]>(
    () => [
      {
        name: 'training posterior',
        type: 'line',
        x: S,
        y: S.map((v) => posterior(v, piTrain.value, d.value)),
        slot: 1,
      },
      {
        name: 'adjusted with EM prior',
        type: 'line',
        x: S,
        y: S.map((v) => adjust(posterior(v, piTrain.value, d.value), piTrain.value, result.est)),
        slot: 0,
      },
      {
        name: 'true deployment posterior',
        type: 'line',
        x: S,
        y: S.map((v) => posterior(v, piNew.value, d.value)),
        emphasis: true,
        dashed: true,
      },
    ],
    [piTrain.value, piNew.value, d.value, result.est],
  )

  return (
    <Interactive
      title="Estimating the deployment prior"
      caption={`A classifier is calibrated for the training prior. It scores ${N} unlabelled deployment cases drawn at a different prior. The mean of its posteriors is pulled towards the training prior. EM alternates between adjusting every posterior to the current estimate and averaging them, and converges to the maximum-likelihood prior. BBSE inverts the classifier's true- and false-positive rates. Right: the posterior before and after adjustment.`}
      controls={
        <>
          <ParamSlider label="training prior" param={piTrain} />
          <ParamSlider label="deployment prior" param={piNew} />
          <ParamSlider label="separation d" param={d} />
          <ParamButton onClick={() => setSeed((v) => v + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="positives in sample" value={formatNumber(result.actual)} />
          <Readout label="EM" value={formatNumber(result.est)} />
          <Readout label="BBSE" value={formatNumber(result.bbse)} />
          <Readout label="mean unadjusted posterior" value={formatNumber(result.naive)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={traceSeries}
          xLabel="EM iteration"
          yLabel="estimated prior"
          xRange={[0, ITERATIONS]}
          yRange={[0, 0.6]}
        />
        <XYChart series={posteriorSeries} xLabel="score s" yLabel="P(y = 1 | s)" xRange={[-4, 6]} yRange={[0, 1]} />
      </div>
    </Interactive>
  )
}
