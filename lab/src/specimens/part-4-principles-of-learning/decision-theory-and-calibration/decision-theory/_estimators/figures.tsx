import { useMemo } from 'react'
import { dataset, evaluate, withDecision } from 'aifn-compute/learning/estimators'
import { accuracy, logLoss, meanSquaredError } from 'aifn-compute/learning/metrics'
import { linearRegression } from 'aifn-methods/learning/linear'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { fromData, mean, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Figure } from 'aifn-render/layout'
import { row, slider, useFigureState } from 'aifn-render/state'
import { Readout } from 'aifn-render/viz'
import { FitPanel, formatValue, TracePanel } from '@lab/views'

/** n inputs x ~ N(0, 1.5²) as an [n, 1] matrix and standard-normal noise, from fixed streams. */
function draws(n: number, seed: string): { x: Tensor; noise: number[] } {
  const s = stream(seed)
  const x = toFlat(normals(child(s, 'x'), [n], 0, 1.5))
  return { x: fromData(Float64Array.from(x), [n, 1]), noise: toFlat(normals(child(s, 'noise'), [n])) }
}

export function LinearFitSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      n: slider(5, 200, 40, { label: 'n', step: 1 }),
      noise: slider(0.05, 2, 0.6, { label: 'noise sd' }),
    }),
    fit: row('2 · fit', { l2: slider(0, 50, 0, { label: 'ridge l2' }) }),
  })
  const { n, noise } = state.data
  const { l2 } = state.fit
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
    <Figure
      purpose="Least squares gives a point prediction and, with the residual spread σ̂, a Gaussian predictive: a band that holds about 90% of new targets when the model is right."
      title="Least squares with a Gaussian predictive"
      state={state}
      readouts={{
        fit: (
          <>
            <Readout label="w" value={formatValue(toFlat(model.weights)[0])} />
            <Readout label="b" value={formatValue(model.intercept)} />
            <Readout label="σ̂" value={formatValue(model.noiseSd)} />
            <Readout label="training MSE" value={formatValue(scores.meanSquaredError)} />
            <Readout label="training log score" value={formatValue(logScore)} />
          </>
        ),
      }}
      caption="The truth is y = 1 + 0.8x + noise. decide(x) and E[y | x] coincide for a Gaussian; the dashed lines are the 5% and 95% quantiles of predictive(x). The ridge penalty shrinks w towards 0 and widens σ̂."
    >
      <FitPanel model={model} data={data} />
    </Figure>
  )
}

export function DecisionRuleSpecimen() {
  const state = useFigureState({
    fit: row('1 · fit', { l2: slider(0.01, 20, 1, { label: 'l2' }) }),
    decision: row('2 · decision', { cost: slider(0.1, 10, 4, { label: 'cost of a false negative' }) }),
  })
  const { l2 } = state.fit
  const { cost } = state.decision
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
    <Figure
      purpose="With a cost matrix the Bayes decision is not the most probable class: decide 1 once P(y = 1 | x) exceeds 1/(1 + cost of a miss), so a costly miss moves the switch left."
      title="Bayes decisions under a cost matrix"
      state={state}
      readouts={{
        decision: (
          <>
            <Readout label="decide 1 when P(y = 1 | x) >" value={formatValue(1 / (1 + cost))} />
            <Readout label="accuracy" value={formatValue(scores.accuracy)} />
            <Readout label="log loss" value={formatValue(scores.logLoss)} />
            <Readout label="Newton steps" value={model.steps} />
          </>
        ),
      }}
      caption="withDecision replaces logistic regression's argmax with the decision that minimises expected cost. A false alarm costs 1; as a missed positive costs more, the threshold on P(y = 1 | x) falls to 1/(1 + cost) and decide(x) switches to 1 further left."
    >
      <FitPanel model={model} data={data} yLabel="y, P(y = 1 | x)" />
    </Figure>
  )
}

export function IrlsTraceSpecimen() {
  const state = useFigureState({ fit: row('1 · penalty', { l2: slider(0.001, 5, 0.1, { label: 'l2' }) }) })
  const { l2 } = state.fit
  const trace = useMemo(() => {
    const { x, noise } = draws(60, 'irls')
    const y = toFlat(x).map((v, i) => (2 * v + 0.8 * noise[i] > 0 ? 1 : 0))
    return logisticRegression({ l2 }).fit(dataset(x, fromData(Float64Array.from(y)))).training
  }, [l2])
  return (
    <Figure
      purpose="Logistic regression is fitted by Newton's method (IRLS): a few steps take the penalised loss to its minimum, fewer the stronger the penalty."
      title="Newton's method (IRLS) for logistic regression"
      state={state}
      caption="The fit's training loop is a traceable Algorithm, kept on the fitted model as `training`. The penalised loss falls fast once full Newton steps are taken; a weaker penalty on nearly separable data needs more steps. Play from step 0."
    >
      <TracePanel trace={trace} show={['loss']} />
    </Figure>
  )
}
