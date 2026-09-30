// Smoke tests (one per export family) and the pipeline's capability types.
import { describe, expect, expectTypeOf, it } from 'vitest'
import {
  affineTarget,
  columns,
  logTarget,
  pipeline,
  powerTarget,
  pushForward,
  transformedPredictive,
  transformTarget,
} from 'aifn/learning/compose'
import {
  asTensor,
  gaussianPredictive,
  withDecision,
  type BernoulliPredictive,
  type CategoricalPredictive,
  type Estimator,
  type Scores,
  type Supervised,
  type Dataset,
  type Transforms,
} from 'aifn/learning/estimators'
import { linearRegression } from 'aifn-applied/learning/linear'
import { logisticRegression } from 'aifn-applied/learning/generalised/glm'
import { mean } from 'aifn/probability/stats'
import { normals, stream } from 'aifn/foundation/random'
import { add, exp, matmul, mul, sub, tensor, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

/** A tiny transformer (subtract the training column means), so that these tests need no application. */
const centre = (): Estimator<Dataset<Tensor, unknown>, Transforms<Tensor>> => ({
  name: 'centre',
  fit({ x }) {
    const m = mean(x, { axis: 0 })
    return { transform: (z: Tensor) => sub(z, m) as Tensor }
  },
})

const s = stream('compose')
const x = normals(s.child('x'), [60, 2])
const logits = toFlat(matmul(x, tensor([2, -1])) as Tensor)
const u = toFlat(normals(s.child('u'), [60]))
const yClass = tensor(logits.map((l, i) => (l + 0.5 * u[i] > 0 ? 1 : 0)))

describe('pipeline', () => {
  it('fits each step on the previous output and has the capabilities of its last step', () => {
    const model = pipeline(centre(), logisticRegression({ l2: 1 })).fit({ x, y: yClass })
    const [c, logistic] = model.steps
    close(
      toFlat(asTensor(model.predictive(x).params.probs)),
      toFlat(asTensor(logistic.predictive(c.transform(x)).params.probs)),
      0,
    )
    expect(model.names).toEqual(['centre', 'logistic-regression'])
    expectTypeOf(model.predictive).returns.toEqualTypeOf<BernoulliPredictive | CategoricalPredictive>()
    const scorer: Estimator<Supervised<Tensor, Tensor>, Scores<Tensor>> = {
      name: 'scorer',
      fit: () => ({ score: (z) => z }),
    }
    const scored = pipeline(centre(), scorer).fit({ x, y: yClass })
    // @ts-expect-error: the last step only scores, so the pipeline has no predictive.
    void scored.predictive
    expect(toFlat(withDecision(scored, 'argmax').decide(tensor([[1, 2]])))).toEqual([1])
  })
})

describe('columns', () => {
  it('transforms named columns side by side', () => {
    const table = { a: tensor([1, 2, 3]), b: tensor([10, 20, 60]), id: tensor([7, 8, 9]) }
    const model = columns({ a: centre(), b: centre(), id: 'drop' }).fit({ x: table })
    expect(toRows(model.transform(table))).toEqual([
      [-1, -20],
      [0, -10],
      [1, 30],
    ])
    expect(model.slices).toEqual({ a: [0, 1], b: [1, 2] })
  })
})

describe('transformTarget', () => {
  // log y = 0.5 + 0.8 x + N(0, 0.2²).
  const xr = normals(s.child('xr'), [200, 1])
  const logY = add(add(0.5, mul(0.8, xr)), mul(0.2, normals(s.child('e'), [200, 1])))
  const y = tensor(toFlat(exp(logY) as Tensor))

  it('pushes a Gaussian on log y forward to a log-normal and inverts point predictions', () => {
    const model = transformTarget(linearRegression(), logTarget()).fit({ x: xr, y })
    const xt = tensor([[0], [1]])
    const d = model.predictive(xt)
    expect(d.name).toBe('LogNormal')
    const inner = model.regressor.predictive(xt)
    const mu = toFlat(inner.mean())
    const sd = toFlat(asTensor(inner.params.scale))
    close(toFlat(model.decide(xt)), mu.map(Math.exp), 1e-12)
    close(
      toFlat(d.mean()),
      mu.map((m, i) => Math.exp(m + sd[i] ** 2 / 2)),
      1e-12,
    )
    // Quadrature agrees with the closed form.
    close(toFlat(model.expect(xt, (v) => v)), toFlat(d.mean()), 1e-8)
    // log p(y) = log N(log y; μ, σ²) − log y.
    const lp = toFlat(d.logProb(tensor([2, 3])))
    close(
      lp,
      [0, 1].map((i) => toFlat(inner.logProb(tensor([Math.log(2), Math.log(3)])))[i] - Math.log([2, 3][i])),
      1e-12,
    )
    close(toFlat(d.cdf(d.quantile(tensor(0.3)))), [0.3, 0.3], 1e-12)
    expect(model.sample(stream('ln'), xt, 5).shape).toEqual([5, 2])
    expect(() =>
      transformTarget(linearRegression(), logTarget()).fit({ x: tensor([[0], [1]]), y: tensor([1, -1]) }),
    ).toThrow(/domain/)
  })

  it('handles general monotone maps and fitted power transforms', () => {
    const base = gaussianPredictive(tensor([0]), tensor([1]))
    const flipped = transformedPredictive(base, affineTarget(1, -2)) // y = 1 − 2z
    close(toFlat(flipped.mean()), [1], 1e-12)
    close(toFlat(flipped.variance()), [4], 1e-10)
    close(toFlat(flipped.cdf(tensor(1))), [0.5], 1e-15)
    expect(pushForward(base, logTarget()).name).toBe('LogNormal')
    const model = transformTarget(linearRegression(), powerTarget({ method: 'box-cox' })).fit({ x: xr, y })
    const lambda = (model.map as unknown as { lambda: number }).lambda
    expect(Math.abs(lambda)).toBeLessThan(0.3)
    expect(model.predictive(tensor([[0]])).name).toBe('Transformed')
  })
})
