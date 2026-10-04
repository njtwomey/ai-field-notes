import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform } from 'aifn/foundation/random'
import { expit, logit, posterior, rates, sample, scoreForPosterior } from '../_shared/binormal'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const N = 5000
const ITERATIONS = 30
const S = toFlat(linspace(-4, 6, 201))

/** Posterior re-weighted from prior a to prior b: odds multiplied by (b / (1 − b)) / (a / (1 − a)). */
const adjust = (p: number, a: number, b: number) => expit(logit(p) + logit(b) - logit(a))

/**
 * Label shift between training and deployment. A calibrated classifier for the training prior scores an unlabelled
 * deployment sample; EM re-estimates the deployment prior (Saerens et al.), and black-box shift estimation inverts the
 * classifier's confusion rates (Lipton et al.).
 */
export function PriorShiftEm() {
  const state = useFigureState({
    piTrain: float(0.5, { min: 0.05, max: 0.5, step: 0.01, label: 'training prior' }),
    piNew: float(0.1, { min: 0.01, max: 0.5, step: 0.01, label: 'deployment prior' }),
    d: float(1.5, { min: 0.5, max: 4, step: 0.1, label: 'separation d' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const result = useMemo(() => {
    const g = stream(state.seed)
    const { s, y } = sample(N, state.piNew, state.d, { uniform: () => uniform(g), normal: () => normal(g) })
    const p = s.map((v) => posterior(v, state.piTrain, state.d))
    const naive = p.reduce((a, b) => a + b, 0) / N
    // EM: E-step adjusts every posterior to the current prior estimate; M-step sets the prior to their mean.
    const trace = [state.piTrain]
    let est = state.piTrain
    for (let it = 0; it < ITERATIONS; it++) {
      let total = 0
      for (const pi of p) total += adjust(pi, state.piTrain, est)
      est = total / N
      trace.push(est)
    }
    // BBSE with hard predictions at p ≥ 0.5; the rates are those of the training population.
    const t = scoreForPosterior(0.5, state.piTrain, state.d)
    const { tpr, fpr } = rates(t, state.d)
    const predictedPositive = s.filter((v) => v >= t).length / N
    const bbse = Math.min(1, Math.max(0, (predictedPositive - fpr) / (tpr - fpr)))
    const actual = y.reduce((a, b) => a + b, 0) / N
    return { trace, est, naive, bbse, actual }
  }, [state.seed, state.piTrain, state.piNew, state.d])

  const traceSeries = useMemo(() => {
    const its = result.trace.map((_, i) => i)
    const flat = (v: number) => [v, v]
    return [
      { name: 'EM estimate', x: its, y: result.trace, slot: 0 },
      {
        name: 'mean of unadjusted posteriors',
        x: [0, ITERATIONS],
        y: flat(result.naive),
        slot: 1,
        dashed: true,
      },
      { name: 'BBSE estimate', x: [0, ITERATIONS], y: flat(result.bbse), slot: 3, dashed: true },
      {
        name: 'true deployment prior',
        x: [0, ITERATIONS],
        y: flat(state.piNew),
        emphasis: true,
        dashed: true,
      },
    ] as const
  }, [result, state.piNew])

  const posteriorSeries = useMemo(
    () =>
      [
        {
          name: 'training posterior',
          x: S,
          y: S.map((v) => posterior(v, state.piTrain, state.d)),
          slot: 1,
        },
        {
          name: 'adjusted with EM prior',
          x: S,
          y: S.map((v) => adjust(posterior(v, state.piTrain, state.d), state.piTrain, result.est)),
          slot: 0,
        },
        {
          name: 'true deployment posterior',
          x: S,
          y: S.map((v) => posterior(v, state.piNew, state.d)),
          emphasis: true,
          dashed: true,
        },
      ] as const,
    [state.piTrain, state.piNew, state.d, result.est],
  )

  const xAxis = useAxis({ label: 'EM iteration', range: [0, ITERATIONS] })
  const yAxis = useAxis({ label: 'estimated prior', range: [0, 0.6] })
  const xAxis2 = useAxis({ label: 'score s', range: [-4, 6] })
  const yAxis2 = useAxis({ label: 'P(y = 1 | s)', range: [0, 1] })
  return (
    <Figure
      title="Estimating the deployment prior"
      state={state}
      caption={`A classifier is calibrated for the training prior. It scores ${N} unlabelled deployment cases drawn at a different prior. The mean of its posteriors is pulled towards the training prior. EM alternates between adjusting every posterior to the current estimate and averaging them, and converges to the maximum-likelihood prior. BBSE inverts the classifier's true- and false-positive rates. Right: the posterior before and after adjustment.`}

      readouts={
        <>
          <Readout label="positives in sample" value={formatNumber(result.actual)} />
          <Readout label="EM" value={formatNumber(result.est)} />
          <Readout label="BBSE" value={formatNumber(result.bbse)} />
          <Readout label="mean unadjusted posterior" value={formatNumber(result.naive)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...traceSeries[0]} />
          <Curve {...traceSeries[1]} />
          <Curve {...traceSeries[2]} />
          <Curve {...traceSeries[3]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...posteriorSeries[0]} />
          <Curve {...posteriorSeries[1]} />
          <Curve {...posteriorSeries[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
