import { describe, expect, expectTypeOf, it } from 'vitest'
import {
  accuracy,
  adaptMetric,
  asTensor,
  bernoulliPredictive,
  capabilities,
  categoricalPredictive,
  classProbabilities,
  defineMetric,
  evaluate,
  expectation,
  gaussHermite,
  gaussianPredictive,
  hasDecide,
  hasPredictive,
  hasScore,
  hasTraining,
  logLoss,
  meanAbsoluteError,
  meanSquaredError,
  readout,
  rowCount,
  rSquared,
  takeData,
  takeRows,
  withDecision,
  withExpectation,
  withSampling,
  type BernoulliPredictive,
  type Decides,
  type Fitted,
  type GaussianPredictive,
  type InputOf,
  type Predicts,
  type Scores,
} from 'aifn/learning/estimators'
import { linearRegression } from 'aifn-applied/learning/linear'
import { logisticIrls, logisticRegression, type IrlsProblem } from 'aifn-applied/learning/generalised/glm'
import * as metrics from 'aifn/metrics'
import { stream } from 'aifn/foundation/random'
import { fromData, tensor, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'
import { fixture } from './fixtures'

type Fx = {
  x: number[][]
  y: number[]
  x_test: number[][]
  ols: { coef: number[]; intercept: number; predict: number[]; noise_sd: number }
  ridge: { alpha: number; coef: number[]; intercept: number; predict: number[] }
  deficient: { x: number[][]; coef: number[]; intercept: number }
  binary: { l2: number; y: number[]; coef: number[]; intercept: number; proba: number[]; predict: number[] }
  multinomial: { l2: number; y: number[]; coef: number[][]; intercept: number[]; proba: number[][]; predict: number[] }
}
const fx = fixture<Fx>('estimators')
const X = tensor(fx.x)
const Y = tensor(fx.y)
const XT = tensor(fx.x_test)

const close = (actual: ArrayLike<number>, expected: ArrayLike<number>, tol: number) => {
  expect(actual.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(tol)
}

describe('linearRegression', () => {
  const model = linearRegression().fit({ x: X, y: Y })

  it('matches scikit-learn LinearRegression', () => {
    close(toFlat(model.weights), fx.ols.coef, 1e-10)
    expect(model.intercept).toBeCloseTo(fx.ols.intercept, 10)
    close(toFlat(model.decide(XT)), fx.ols.predict, 1e-10)
    expect(model.noiseSd).toBeCloseTo(fx.ols.noise_sd, 10)
    expect(model.rank).toBe(3)
    expect(model.residualDof).toBe(40 - 4)
  })

  it('matches scikit-learn Ridge', () => {
    const ridge = linearRegression({ l2: fx.ridge.alpha }).fit({ x: X, y: Y })
    close(toFlat(ridge.weights), fx.ridge.coef, 1e-10)
    expect(ridge.intercept).toBeCloseTo(fx.ridge.intercept, 10)
    close(toFlat(ridge.expect(XT)), fx.ridge.predict, 1e-10)
  })

  it('reports rank deficiency and returns the minimum-norm solution', () => {
    const m = linearRegression().fit({ x: tensor(fx.deficient.x), y: Y })
    expect(m.rank).toBe(2)
    close(toFlat(m.weights), fx.deficient.coef, 1e-8)
  })

  it('has a Gaussian predictive with the plug-in noise', () => {
    const d = model.predictive(XT)
    expect(d.name).toBe('Normal')
    close(toFlat(d.mean()), fx.ols.predict, 1e-10)
    close(toFlat(asTensor(d.params.scale)), Array(5).fill(fx.ols.noise_sd), 1e-12)
    // E[y²] = μ² + σ² by quadrature.
    const second = toFlat(model.expect(XT, (v) => v * v))
    close(
      second,
      fx.ols.predict.map((m) => m * m + fx.ols.noise_sd ** 2),
      1e-9,
    )
  })

  it('samples from the predictive', () => {
    const draws = model.sample(stream('lr'), XT, 4000)
    expect(draws.shape).toEqual([4000, 5])
    const rows = toRows(draws)
    const mean0 = rows.reduce((s, r) => s + r[0], 0) / rows.length
    expect(Math.abs(mean0 - fx.ols.predict[0])).toBeLessThan(4 * (fx.ols.noise_sd / Math.sqrt(4000)))
    expect(toFlat(model.sample(stream('lr'), XT, 3))).toEqual(toFlat(model.sample(stream('lr'), XT, 3)))
  })

  it('rejects mismatched shapes', () => {
    expect(() => linearRegression().fit({ x: X, y: tensor([1, 2]) })).toThrow(/targets/)
    expect(() => model.decide(tensor([[1, 2]]))).toThrow(/features/)
    expect(() => linearRegression({ l2: -1 })).toThrow()
  })
})

describe('logisticRegression', () => {
  const yb = tensor(fx.binary.y)
  const binary = logisticRegression({ l2: fx.binary.l2 }).fit({ x: X, y: yb })

  it('matches scikit-learn (binary)', () => {
    expect(binary.converged).toBe(true)
    close(toFlat(binary.weights), fx.binary.coef, 1e-6)
    expect(toFlat(binary.intercept)[0]).toBeCloseTo(fx.binary.intercept, 6)
    const d = binary.predictive(XT) as BernoulliPredictive
    close(toFlat(asTensor(d.params.probs)), fx.binary.proba, 1e-7)
    close(toFlat(binary.decide(XT)), fx.binary.predict, 0.5)
    close(toFlat(binary.expect(XT)), fx.binary.proba, 1e-7)
  })

  it('matches scikit-learn (multinomial), with intercepts summing to zero', () => {
    const m = logisticRegression({ l2: fx.multinomial.l2 }).fit({ x: X, y: tensor(fx.multinomial.y) })
    expect(m.multinomial).toBe(true)
    expect(m.classes).toBe(3)
    close(toFlat(m.weights), fx.multinomial.coef.flat(), 1e-6)
    close(toFlat(m.intercept), fx.multinomial.intercept, 1e-6)
    const probs = toFlat(asTensor(m.predictive(XT).params.probs))
    close(probs, fx.multinomial.proba.flat(), 1e-7)
    close(toFlat(m.decide(XT)), fx.multinomial.predict, 0.5)
    expect(m.score(XT).shape).toEqual([5, 3])
  })

  it('keeps a monotone training trace that converges quadratically', () => {
    expect(hasTraining(binary)).toBe(true)
    const t = binary.training
    expect(t.meta.stopped).toBe('done')
    const loss = toFlat(t.series.loss)
    for (let k = 1; k < loss.length; k++) expect(loss[k]).toBeLessThanOrEqual(loss[k - 1] + 1e-12)
    expect(t.steps.length).toBeLessThan(15)
    expect(t.steps.at(-1)!.stepSize).toBe(1)
  })

  it('satisfies the trace protocol', () => {
    const n = 40
    const problem: IrlsProblem = {
      design: fromData(
        Float64Array.from({ length: n * 4 }, (_, k) => (k % 4 === 3 ? 1 : toFlat(X)[Math.floor(k / 4) * 3 + (k % 4)])),
        [n, 4],
      ),
      labels: yb,
      columns: 1,
      l2: 0.5,
      intercept: true,
      tol: 0,
    }
    const alg = logisticIrls(problem)
    const a = trace(alg, {}, 6)
    const b = trace(alg, {}, 6)
    expect(toFlat(a.steps[6].weights)).toEqual(toFlat(b.steps[6].weights))
    for (const i of [0, 2, 5]) expect(toFlat(seek(alg, {}, i).weights)).toEqual(toFlat(run(alg, {}, i).weights))
    const long = trace(alg, {}, 8)
    const extended = extend(trace(alg, {}, 3), alg, {}, 5)
    expect(extended.index).toEqual(long.index)
    expect(toFlat(extended.steps.at(-1)!.weights)).toEqual(toFlat(long.steps.at(-1)!.weights))
  })

  it('reports failure to converge on separable data without a penalty', () => {
    const x = tensor([[-2], [-1], [1], [2]])
    const y = tensor([0, 0, 1, 1])
    const m = logisticRegression({ l2: 0, maxIterations: 8 }).fit({ x, y })
    expect(m.converged).toBe(false)
    expect(m.training.meta.stopped).toBe('limit')
  })

  it('rejects labels that are not 0 … K−1', () => {
    expect(() => logisticRegression().fit({ x: X, y: fromData(new Float64Array(40).fill(0.5)) })).toThrow(/labels/)
  })
})

describe('predictive stopgaps', () => {
  it('Gaussian: logProb, cdf and quantile', () => {
    const d = gaussianPredictive(tensor([0, 1]), tensor([1, 2]))
    close(
      toFlat(d.logProb(tensor([0, 3]))),
      [-0.5 * Math.log(2 * Math.PI), -0.5 - Math.log(2) - 0.5 * Math.log(2 * Math.PI)],
      1e-14,
    )
    close(toFlat(d.cdf(tensor([0, 1]))), [0.5, 0.5], 1e-15)
    close(toFlat(d.quantile(tensor(0.5))), [0, 1], 1e-15)
    close(
      toFlat(d.logProb(tensor(0))).map(Math.exp),
      [1 / Math.sqrt(2 * Math.PI), Math.exp(-1 / 8) / (2 * Math.sqrt(2 * Math.PI))],
      1e-14,
    )
  })

  it('Bernoulli and categorical: probabilities, modes and class matrices', () => {
    const b = bernoulliPredictive(tensor([0.2, 0.7]))
    close(toFlat(b.logProb(tensor([1, 0]))), [Math.log(0.2), Math.log(0.3)], 1e-14)
    expect(toFlat(b.mode())).toEqual([0, 1])
    close(classProbabilities(b).probs, [0.8, 0.2, 0.3, 0.7], 1e-15)
    const c = categoricalPredictive(tensor([[0.1, 0.6, 0.3]]))
    close(toFlat(c.logProb(tensor(2))), [Math.log(0.3)], 1e-14)
    expect(toFlat(c.mode())).toEqual([1])
    close(toFlat(c.mean()), [1.2], 1e-14)
    expect(c.sample(stream('c'), { shape: [5] }).shape).toEqual([5, 1])
  })

  it('Gauss–Hermite integrates polynomials exactly', () => {
    const { nodes, weights } = gaussHermite()
    let m0 = 0
    let m4 = 0
    for (let i = 0; i < nodes.length; i++) [m0, m4] = [m0 + weights[i], m4 + weights[i] * nodes[i] ** 4]
    expect(m0).toBeCloseTo(1, 13)
    expect(m4).toBeCloseTo(3, 11)
  })

  it('expectation: sums for classes, quadrature otherwise', () => {
    const c = categoricalPredictive(tensor([[0.1, 0.6, 0.3]]))
    close(toFlat(expectation(c, (k) => k * k)), [0.6 + 1.2], 1e-14)
    const g = gaussianPredictive(tensor([1]), tensor([0.5]))
    close(toFlat(expectation(g, Math.exp)), [Math.exp(1 + 0.125)], 1e-10)
  })
})

describe('mixins', () => {
  // A score-only model: logits [N, 3] from a fixed matrix.
  const W = [
    [1, 0, -1],
    [0, 1, 1],
  ]
  const scorer: Scores<Tensor> & { W: number[][] } = {
    W,
    score: (x) => {
      const rows = toRows(x)
      return tensor(rows.map((r) => [0, 1, 2].map((k) => r[0] * W[0][k] + r[1] * W[1][k])))
    },
  }
  const x = tensor([
    [2, 0],
    [0, 2],
    [-1, 1],
  ])

  it('withDecision argmax from scores', () => {
    const m = withDecision(scorer, 'argmax')
    expect(toFlat(m.decide(x))).toEqual([0, 1, 2])
    expect(m.W).toBe(W)
    expect(hasDecide(scorer)).toBe(false)
    expect(capabilities(m)).toEqual(['decide', 'score'])
  })

  const probModel: Predicts<Tensor, BernoulliPredictive> = {
    predictive: (x) => bernoulliPredictive(fromData(Float64Array.from(toFlat(x)), [x.shape[0]])),
  }
  const p = tensor([0.1, 0.3, 0.6, 0.9])

  it('withDecision threshold, mode and cost matrix', () => {
    expect(toFlat(withDecision(probModel, { threshold: 0.5 }).decide(p))).toEqual([0, 0, 1, 1])
    expect(toFlat(withDecision(probModel, { threshold: 0.2 }).decide(p))).toEqual([0, 1, 1, 1])
    expect(toFlat(withDecision(probModel, 'mode').decide(p))).toEqual([0, 0, 1, 1])
    // Missing a positive costs 4 and a false alarm 1: decide 1 when 4p > 1 − p, i.e. p > 0.2.
    const costs = [
      [0, 1],
      [4, 0],
    ]
    expect(toFlat(withDecision(probModel, { costs }).decide(p))).toEqual([0, 1, 1, 1])
    expect(() =>
      withDecision(probModel, {
        costs: [
          [0, 1, 1],
          [1, 0, 1],
          [1, 1, 0],
        ],
      }).decide(p),
    ).toThrow(/classes/)
  })

  it('withExpectation and withSampling', () => {
    const m = withSampling(withExpectation(probModel))
    close(toFlat(m.expect(p)), toFlat(p), 0)
    close(
      toFlat(m.expect(p, (y) => 3 * y + 1)),
      toFlat(p).map((q) => 3 * q + 1),
      1e-15,
    )
    expect(m.sample(stream('b'), p, 10).shape).toEqual([10, 4])
  })

  it('readout completes a forward pass; capabilities follow the completers', () => {
    const latent: Fitted<Tensor, Tensor> & { weight: number } = {
      weight: 2,
      forward: (x) =>
        fromData(
          Float64Array.from(toFlat(x), (v) => 2 * v),
          [x.shape[0]],
        ),
    }
    const bern = readout(latent, (h) =>
      bernoulliPredictive(fromData(Float64Array.from(toFlat(h), (e) => 1 / (1 + Math.exp(-e))))),
    )
    close(toFlat(asTensor(bern.predictive(tensor([0])).params.probs)), [0.5], 1e-15)
    expect(bern.weight).toBe(2)
    expect(hasPredictive(bern)).toBe(true)
    expect(hasDecide(bern)).toBe(false)
    const both = readout(latent, {
      decide: (h) => fromData(Int32Array.from(toFlat(h), (e) => (e > 0 ? 1 : 0))),
      score: (h) => h,
    })
    expect(toFlat(both.decide(tensor([-1, 1])))).toEqual([0, 1])
    expect(hasScore(both)).toBe(true)
    expect(hasPredictive(both)).toBe(false)

    expectTypeOf(bern.predictive).returns.toEqualTypeOf<BernoulliPredictive>()
    expectTypeOf(both.decide).parameter(0).toEqualTypeOf<Tensor>()
    // @ts-expect-error: a decide-and-score readout has no predictive.
    void both.predictive
  })
})

describe('capability types', () => {
  it('asking a decide-only model for its predictive is a compile error', () => {
    const decider: Decides<Tensor, Tensor> = { decide: (x) => x }
    // @ts-expect-error: Decides has no predictive.
    void decider.predictive
    // @ts-expect-error: logLoss needs a predictive, which a decide-only model lacks.
    expect(() => evaluate(decider, { x: X, y: Y }, [logLoss])).toThrow(/predictive/)
    // A decide-only model serves accuracy.
    expect(evaluate(decider, { x: tensor([1, 2]), y: tensor([1, 3]) }, [accuracy])).toEqual({ accuracy: 0.5 })
  })

  it('mixins add exactly the declared capabilities', () => {
    const scorer: Scores<Tensor> = { score: (x) => x }
    const decided = withDecision(scorer, 'argmax')
    expectTypeOf(decided).toHaveProperty('decide')
    expectTypeOf(decided).not.toHaveProperty('predictive')
    // @ts-expect-error: withExpectation needs a predictive.
    expect(() => withExpectation(scorer)).toThrow()
    const lr = linearRegression().fit({ x: X, y: Y })
    expectTypeOf(lr.predictive).returns.toEqualTypeOf<GaussianPredictive>()
    expectTypeOf<InputOf<typeof lr>>().toEqualTypeOf<Tensor>()
  })

  it('guards narrow unknown models', () => {
    const models: unknown[] = [linearRegression().fit({ x: X, y: Y }), { decide: (x: Tensor) => x }]
    const withPredictive = models.filter((m) => hasPredictive(m))
    expect(withPredictive.length).toBe(1)
    const m = models[0]
    if (hasPredictive(m)) expect(m.predictive(XT).batchShape).toEqual([5])
  })
})

describe('evaluate', () => {
  it('computes every metric from one call per capability', () => {
    const lr = linearRegression().fit({ x: X, y: Y })
    const r = evaluate(lr, { x: X, y: Y }, [meanSquaredError, meanAbsoluteError, rSquared, logLoss])
    const mse = lr.rss / 40
    expect(r.mse).toBeCloseTo(mse, 12)
    expect(r.r2).toBeGreaterThan(0.95)
    // Gaussian log loss at the plug-in σ: ½ log 2πσ² + mse / (2σ²).
    const s2 = lr.noiseSd ** 2
    expect(r['log-loss']).toBeCloseTo(0.5 * Math.log(2 * Math.PI * s2) + mse / (2 * s2), 12)
  })

  it('accepts custom metrics', () => {
    const lr = linearRegression().fit({ x: X, y: Y })
    const maxError = defineMetric({
      name: 'max-error',
      needs: 'expect',
      direction: 'lower',
      compute: (y, m) => Math.max(...toFlat(y).map((v, i) => Math.abs(v - toFlat(m)[i]))),
    })
    expect(evaluate(lr, { x: X, y: Y }, [maxError])['max-error']).toBeGreaterThan(0)
  })
})

describe('adaptMetric', () => {
  it('feeds aifn/metrics functions the output they declare', () => {
    const yb = tensor(fx.binary.y)
    const model = logisticRegression({ l2: fx.binary.l2 }).fit({ x: X, y: yb })
    const r = evaluate(model, { x: X, y: yb }, [
      adaptMetric(metrics.logLoss, 'predictive'),
      adaptMetric(metrics.auroc, 'score'),
      adaptMetric(metrics.accuracy, 'decide'),
      logLoss,
    ])
    expect(r.logLoss).toBeCloseTo(r['log-loss'], 12)
    expect(r.auroc).toBeCloseTo(0.7275, 12) // scikit-learn's roc_auc_score
    expect(r.accuracy).toBeCloseTo(0.625, 12)
    expect(() => adaptMetric(metrics.auroc, 'predictive')).toThrow(/score/)
  })
})

describe('rows', () => {
  it('rowCount and takeRows on tensors, lists and tables', () => {
    const t = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ])
    expect(rowCount(t)).toBe(3)
    expect(toRows(takeRows(t, [2, 0]))).toEqual([
      [5, 6],
      [1, 2],
    ])
    expect(takeRows(['a', 'b', 'c'], [1, 1])).toEqual(['b', 'b'])
    const table = { age: tensor([30, 40, 50]), city: ['Cork', 'Paris', 'Oslo'] }
    expect(rowCount(table)).toBe(3)
    const sub = takeRows(table, [2])
    expect(toFlat(sub.age as Tensor)).toEqual([50])
    expect(sub.city).toEqual(['Oslo'])
    expect(() => rowCount({ a: [1], b: [1, 2] })).toThrow(/rows/)
    const data = takeData({ x: t, y: tensor([0, 1, 2]) }, [1])
    expect(toFlat(data.y)).toEqual([1])
    expect(() => takeRows(t, [3])).toThrow(/range/)
  })
})
