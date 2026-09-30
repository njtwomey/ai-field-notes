import { useMemo, useState } from 'react'
import { dataset, evaluate, withDecision } from 'aifn/learning/estimators'
import { accuracy, logLoss, meanSquaredError } from 'aifn/learning/metrics'
import { linearRegression } from 'aifn-applied/learning/linear'
import { logisticRegression } from 'aifn-applied/learning/generalised/glm'
import { child, normals, stream } from 'aifn/foundation/random'
import { fromData, mean, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { Slider } from '@lab/controls'
import { Readout } from '@lab/viz'
import { FitView, formatValue, TraceView } from '@lab/views'

/** n inputs x ~ N(0, 1.5²) as an [n, 1] matrix and standard-normal noise, from fixed streams. */
function draws(n: number, seed: string): { x: Tensor; noise: number[] } {
  const s = stream(seed)
  const x = toFlat(normals(child(s, 'x'), [n], 0, 1.5))
  return { x: fromData(Float64Array.from(x), [n, 1]), noise: toFlat(normals(child(s, 'noise'), [n])) }
}

export function LinearFitSpecimen() {
  const [n, setN] = useState(40)
  const [noise, setNoise] = useState(0.6)
  const [l2, setL2] = useState(0)
  const data = useMemo(() => {
    const { x, noise: e } = draws(n, 'linear-fit')
    const y = toFlat(x).map((v, i) => 1 + 0.8 * v + noise * e[i])
    return dataset(x, fromData(Float64Array.from(y)))
  }, [n, noise])
  const model = useMemo(() => linearRegression({ l2 }).fit(data), [data, l2])
  const scores = evaluate(model, data, [meanSquaredError])
  // The log score of a Gaussian predictive: the mean negative log density of the targets.
  const logScore = -mean(model.predictive(data.x).logProb(data.y) as Tensor)
  return (
    <FitView
      title="Least squares with a Gaussian predictive"
      model={model}
      data={data}
      controls={
        <>
          <Slider label="n" value={n} min={5} max={200} step={1} onChange={setN} />
          <Slider label="noise sd" value={noise} min={0.05} max={2} onChange={setNoise} />
          <Slider label="ridge l2" value={l2} min={0} max={50} onChange={setL2} />
        </>
      }
      readouts={
        <>
          <Readout label="w" value={formatValue(toFlat(model.weights)[0])} />
          <Readout label="b" value={formatValue(model.intercept)} />
          <Readout label="σ̂" value={formatValue(model.noiseSd)} />
          <Readout label="training MSE" value={formatValue(scores.meanSquaredError)} />
          <Readout label="training log score" value={formatValue(logScore)} />
        </>
      }
      caption="The truth is y = 1 + 0.8x + noise. decide(x) and E[y | x] coincide for a Gaussian; the dashed lines are the 5% and 95% quantiles of predictive(x). The ridge penalty shrinks w towards 0 and widens σ̂."
    />
  )
}

export function DecisionRuleSpecimen() {
  const [l2, setL2] = useState(1)
  const [cost, setCost] = useState(1)
  const data = useMemo(() => {
    const { x, noise } = draws(80, 'decision-rule')
    const y = toFlat(x).map((v, i) => (1.5 * v - 0.5 + noise[i] > 0 ? 1 : 0))
    return dataset(x, fromData(Float64Array.from(y)))
  }, [])
  const fitted = useMemo(() => logisticRegression({ l2 }).fit(data), [data, l2])
  const model = useMemo(
    () =>
      withDecision(fitted, {
        costs: [
          [0, 1],
          [cost, 0],
        ],
      }),
    [fitted, cost],
  )
  const scores = evaluate(model, data, [accuracy, logLoss])
  return (
    <FitView
      title="Bayes decisions under a cost matrix"
      model={model}
      data={data}
      yLabel="y, P(y = 1 | x)"
      controls={
        <>
          <Slider label="l2" value={l2} min={0.01} max={20} onChange={setL2} />
          <Slider label="cost of a false negative" value={cost} min={0.1} max={10} onChange={setCost} />
        </>
      }
      readouts={
        <>
          <Readout label="decide 1 when P(y = 1 | x) >" value={formatValue(1 / (1 + cost))} />
          <Readout label="accuracy" value={formatValue(scores.accuracy)} />
          <Readout label="log loss" value={formatValue(scores.logLoss)} />
          <Readout label="Newton steps" value={model.steps} />
        </>
      }
      caption="withDecision replaces logistic regression's argmax with the decision that minimises expected cost. A false alarm costs 1; as a missed positive costs more, the threshold on P(y = 1 | x) falls to 1/(1 + cost) and decide(x) switches to 1 further left."
    />
  )
}

export function IrlsTraceSpecimen() {
  const [l2, setL2] = useState(0.1)
  const trace = useMemo(() => {
    const { x, noise } = draws(60, 'irls')
    const y = toFlat(x).map((v, i) => (2 * v + 0.8 * noise[i] > 0 ? 1 : 0))
    return logisticRegression({ l2 }).fit(dataset(x, fromData(Float64Array.from(y)))).training
  }, [l2])
  return (
    <TraceView
      title="Newton's method (IRLS) for logistic regression"
      trace={trace}
      show={['loss']}
      controls={<Slider label="l2" value={l2} min={0.001} max={5} onChange={setL2} />}
      caption="The fit's training loop is a traceable Algorithm, kept on the fitted model as `training`. The penalised loss falls fast once full Newton steps are taken; a weaker penalty on nearly separable data needs more steps."
    />
  )
}
