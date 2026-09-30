// Smoke tests (one per export family); scikit-learn 1.9 split indices inlined where comparable.
import { describe, expect, it } from 'vitest'
import {
  accuracy,
  logLoss,
  meanSquaredError,
  type Decides,
  type Estimator,
  type Supervised,
} from 'aifn/learning/estimators'
import { linearRegression } from 'aifn-applied/learning/linear'
import { logisticRegression } from 'aifn-applied/learning/generalised/glm'
import { normals, stream } from 'aifn/foundation/random'
import { add, matmul, mul, tensor, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import {
  assignment,
  crossValidate,
  expandingWindow,
  gridSearch,
  groupKFold,
  kFold,
  leaveOneOut,
  logUniformRange,
  nested,
  randomSearch,
  repeated,
  rollingOrigin,
  shuffleSplit,
  stratifiedKFold,
  type Split,
} from 'aifn/learning/validate'

const tests = (splits: Split[]) => splits.map((s) => toFlat(s.test))
const trains = (splits: Split[]) => splits.map((s) => toFlat(s.train))

/** Every row is tested exactly once and never trained on in its own fold. */
function isPartition(splits: Split[], n: number) {
  const seen = new Array(n).fill(0)
  for (const s of splits) {
    for (const i of toFlat(s.test)) seen[i]++
    const test = new Set(toFlat(s.test))
    for (const i of toFlat(s.train)) expect(test.has(i)).toBe(false)
    expect(toFlat(s.train).length + toFlat(s.test).length).toBe(n)
  }
  expect(seen).toEqual(new Array(n).fill(1))
}

describe('splitters', () => {
  it('k-fold matches scikit-learn and partitions; shuffled is a deterministic partition', () => {
    expect(tests(kFold({ k: 3 }).split({ n: 10 }))).toEqual([
      [0, 1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ])
    const shuffled = kFold({ k: 4, shuffle: true })
    isPartition(shuffled.split({ n: 23 }, stream('k')), 23)
    expect(tests(shuffled.split({ n: 23 }, stream('k')))).toEqual(tests(shuffled.split({ n: 23 }, stream('k'))))
    expect(() => shuffled.split({ n: 23 })).toThrow(/stream/)
    isPartition(leaveOneOut().split({ n: 5 }), 5)
  })

  it('stratified k-fold matches scikit-learn and keeps class proportions', () => {
    const y = tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0, 1])
    expect(tests(stratifiedKFold({ k: 3 }).split({ n: 12, y }))).toEqual([
      [0, 1, 3, 7],
      [2, 4, 5, 8],
      [6, 9, 10, 11],
    ])
    const labels = Array.from({ length: 60 }, (_, i) => (i % 5 === 0 ? 'rare' : 'common'))
    const splits = stratifiedKFold({ k: 4, shuffle: true }).split({ n: 60, y: labels }, stream('strat'))
    isPartition(splits, 60)
    for (const s of splits) expect(toFlat(s.test).filter((i) => labels[i] === 'rare').length).toBe(3)
  })

  it('grouped k-fold matches scikit-learn and never splits a group', () => {
    const groups = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4]
    const splits = groupKFold({ k: 3 }).split({ n: 15, groups })
    expect(tests(splits)).toEqual([
      [9, 10, 11, 12, 13],
      [0, 1, 2, 3, 14],
      [4, 5, 6, 7, 8],
    ])
    for (const s of splits) {
      const testGroups = new Set(toFlat(s.test).map((i) => groups[i]))
      for (const i of toFlat(s.train)) expect(testGroups.has(groups[i])).toBe(false)
    }
  })

  it('time-series splits match scikit-learn and never train on the future', () => {
    const splits = expandingWindow({ splits: 3, gap: 1 }).split({ n: 10 })
    expect(trains(splits)).toEqual([
      [0, 1, 2],
      [0, 1, 2, 3, 4],
      [0, 1, 2, 3, 4, 5, 6],
    ])
    expect(tests(splits)).toEqual([
      [4, 5],
      [6, 7],
      [8, 9],
    ])
    const rolling = rollingOrigin({ window: 4, horizon: 2, gap: 1 }).split({ n: 12 })
    for (const s of rolling) {
      expect(toFlat(s.train).length).toBe(4)
      expect(Math.max(...toFlat(s.train))).toBeLessThan(Math.min(...toFlat(s.test)) - 1)
    }
    expect(rolling.length).toBe(3)
  })

  it('shuffle-split and repeated k-fold', () => {
    const ss = shuffleSplit({ splits: 4, testSize: 0.25, trainSize: 0.5 }).split({ n: 20 }, stream('ss'))
    for (const s of ss) expect([toFlat(s.test).length, toFlat(s.train).length]).toEqual([5, 10])
    const rep = repeated(kFold({ k: 5, shuffle: true }), 3).split({ n: 20 }, stream('rep'))
    expect(rep.length).toBe(15)
    isPartition(rep.slice(5, 10), 20)
    expect(tests(rep.slice(0, 5))).not.toEqual(tests(rep.slice(5, 10)))
  })

  it('assignment matrix', () => {
    const a = assignment(shuffleSplit({ splits: 2, testSize: 1, trainSize: 2 }).split({ n: 4 }, stream('a')), 4)
    expect(a.shape).toEqual([2, 4])
    for (const row of toRows(a)) expect(row.slice().sort()).toEqual([-1, 0, 0, 1])
  })
})

describe('crossValidate', () => {
  const s = stream('cv')
  const x = normals(s.child('x'), [80, 3])
  const y = tensor(toFlat(add(matmul(x, tensor([1, -2, 0])), mul(0.5, normals(s.child('e'), [80]))) as Tensor))
  const labels = tensor(toFlat(y).map((v) => (v > 0 ? 1 : 0)))

  it('keeps folds, models, predictions, metrics and traces', () => {
    const cv = crossValidate(logisticRegression(), { x, y: labels }, kFold({ k: 4 }), [accuracy, logLoss])
    expect(cv.folds.length).toBe(4)
    expect(cv.assignment.shape).toEqual([4, 80])
    expect(cv.scores.accuracy.shape).toEqual([4])
    expect(cv.mean.accuracy).toBeGreaterThan(0.8)
    expect(cv.folds[0].training?.meta.stopped).toBe('done')
    expect(cv.outOfFold.decide?.shape).toEqual([80])
    expect(cv.directions['log-loss']).toBe('lower')
  })

  it('rejects metrics the model cannot serve (types)', () => {
    const decider: Estimator<Supervised<Tensor, Tensor>, Decides<Tensor, Tensor>> = {
      name: 'always-one',
      fit: () => ({ decide: (z) => tensor(new Array(z.shape[0]).fill(1)) }),
    }
    // @ts-expect-error: log loss needs a predictive; the fitted model only decides.
    expect(() => crossValidate(decider, { x, y: labels }, kFold({ k: 2 }), [logLoss])).toThrow(/predictive/)
    expect(crossValidate(decider, { x, y: labels }, kFold({ k: 2 }), [accuracy]).folds.length).toBe(2)
  })
})

describe('search and nested cross-validation', () => {
  const s = stream('search')
  const x = normals(s.child('x'), [40, 8])
  const y = tensor(toFlat(add(matmul(x, tensor([1, 0.5, 0, 0, 0, 0, 0, 0])), normals(s.child('e'), [40])) as Tensor))

  it('grid search keeps the full table and refits the best', () => {
    const search = gridSearch(
      (p) => linearRegression({ l2: p.l2 }),
      { l2: [0, 1, 10, 100] },
      { metrics: [meanSquaredError] },
    )
    const r = search.run({ x, y }, kFold({ k: 5 }))
    expect(r.rows.length).toBe(4)
    expect(r.scores.shape).toEqual([4, 5])
    expect(r.best.rank).toBe(1)
    expect(r.rows.map((row) => row.params.l2)).toEqual([0, 1, 10, 100])
    expect(r.model?.l2).toBe(r.best.params.l2)
  })

  it('random search draws candidates from the stream', () => {
    const search = randomSearch(
      (p) => linearRegression({ l2: p.l2 }),
      { l2: logUniformRange(0.01, 100) },
      { metrics: [meanSquaredError], iterations: 5 },
    )
    const a = search.run({ x, y }, kFold({ k: 4 }), stream('rs'))
    const b = search.run({ x, y }, kFold({ k: 4 }), stream('rs'))
    expect(a.rows.map((r) => r.params.l2)).toEqual(b.rows.map((r) => r.params.l2))
    for (const r of a.rows) expect(r.params.l2 >= 0.01 && r.params.l2 < 100).toBe(true)
  })

  it('nested cross-validation reports both levels and the optimism', () => {
    const search = gridSearch(
      (p) => linearRegression({ l2: p.l2 }),
      { l2: [0, 3, 30] },
      { metrics: [meanSquaredError] },
    )
    const result = nested(kFold({ k: 4 }), kFold({ k: 3 }), search, { x, y })
    expect(result.perFold.length).toBe(4)
    expect(result.outer.folds[0].model.search.rows.length).toBe(3)
    expect(result.unnested.rows.length).toBe(3)
    expect(Number.isFinite(result.optimism)).toBe(true)
    expect(result.optimism).toBeCloseTo(result.nestedScore - result.unnestedScore, 12)
  })
})
