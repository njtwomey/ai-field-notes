import { describe, expect, it } from 'vitest'
import {
  backfitting,
  cyclic,
  expectileGam,
  explainableBoostingMachine,
  factorTerm,
  gam,
  linearTerm,
  s,
  te,
  thinPlate,
  ebmBoosting,
} from 'aifn/gam'
import { poisson } from 'aifn/glm'
import { normals, poisson as poissonDraws, stream, uniform } from 'aifn/random'
import { fromData, linspace, tensor, toFlat, toRows, type Tensor } from 'aifn/tensor'
import { extend, run, seek, trace } from 'aifn/trace'
import { fixture } from './fixtures'

type Fixture = {
  x: number[][]
  y: number[]
  k: number
  lambdas: number[]
  fitted: number[]
  edf: number
  gcv_best: number
}
const F = fixture<Fixture>('gam')
const x = tensor(F.x)
const y = tensor(F.y)

function close(actual: number[], expected: number[], tol: number) {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThan(tol * (1 + Math.abs(expected[i]))))
}

describe('gam', () => {
  it('a Gaussian additive fit at fixed λ matches the direct constrained solve', () => {
    const m = gam({
      terms: [s(0, { k: F.k, lambda: F.lambdas[0] }), s(1, { k: F.k, lambda: F.lambdas[1] })],
      method: 'fixed',
    }).fit({ x, y })
    close(toFlat(m.fitted), F.fitted, 1e-7)
    expect(m.edf).toBeCloseTo(F.edf, 6)
    expect(m.termEdf.reduce((a, b) => a + b, 1)).toBeCloseTo(m.edf, 8)
  })

  it('GCV selects the λ that minimises the GCV curve', () => {
    const m = gam({ terms: [s(0, { k: F.k })], method: 'gcv' }).fit({ x, y })
    expect(Math.abs(Math.log(m.lambdas[0]) - F.gcv_best)).toBeLessThan(0.06)
  })

  it('REML selects a λ, and partial effects come with bands and draws', () => {
    const m = gam({ terms: [s(0), s(1)] }).fit({ x, y })
    expect(m.lambdas.every((l) => l > 0 && Number.isFinite(l))).toBe(true)
    expect(m.score.method).toBe('reml')
    const grid = linspace(0, 1, 11)
    const p = m.partial(0, grid)
    const truth = toFlat(grid).map((v) => Math.sin(2 * Math.PI * v))
    toFlat(p.fit).forEach((f, i) => expect(Math.abs(f - truth[i])).toBeLessThan(4 * toFlat(p.se)[i] + 0.1))
    const draws = m.partialDraws(stream('draws'), 0, grid, 5)
    expect(draws.shape).toEqual([5, 11])
    expect(m.sample(stream('y'), x, 2).shape).toEqual([2, F.y.length])
  })

  it('backfitting converges to the penalised fit at the same λ', () => {
    const terms = [s(0, { k: F.k, lambda: F.lambdas[0] }), s(1, { k: F.k, lambda: F.lambdas[1] })]
    const final = run(backfitting({ terms, x, y }), {}, 500)
    expect(final.converged).toBe(true)
    close(toFlat(final.eta), F.fitted, 1e-5)
    const alg = backfitting({ terms, x, y })
    for (const i of [0, 3]) expect(toFlat(seek(alg, {}, i).eta)).toEqual(toFlat(run(alg, {}, i).eta))
    expect(toFlat(extend(trace(alg, {}, 2), alg, {}, 2).steps.at(-1)!.eta)).toEqual(toFlat(run(alg, {}, 4).eta))
  })

  it('a Poisson GAM with a factor, a linear, a cyclic, a thin-plate and a tensor term fits', () => {
    const st = stream('poisson-gam')
    const n = 200
    const u = toRows(uniform(st.child('x'), 0, 1, { shape: [n, 4] }) as Tensor)
    const g = u.map((r) => (r[3] < 0.5 ? 0 : 1))
    const eta = u.map((r, i) => 0.5 + Math.sin(2 * Math.PI * r[0]) * 0.5 + 0.3 * r[1] + 0.4 * g[i])
    const counts = toFlat(poissonDraws(st.child('y'), tensor(eta.map(Math.exp))) as Tensor)
    const X = tensor(u.map((r, i) => [r[0], r[1], r[2], g[i]]))
    const m = gam({
      terms: [cyclic(0, { range: [0, 1], k: 8 }), factorTerm(3), te(1, 2, { k: [4, 4] })],
      family: poisson(),
      method: 'gcv',
    }).fit({ x: X, y: tensor(counts) })
    expect(m.converged).toBe(true)
    expect(m.labels).toEqual(['s(x0)', 'factor(x3)', 'te(x1, x2)'])
    expect(m.jitter).toBe(0)
    expect(m.dispersion).toBe(1)
    const c = toFlat(m.partial(0, tensor([0, 1])).fit)
    expect(c[0]).toBeCloseTo(c[1], 8)
  })

  it('shape constraints leave no violations', () => {
    const st = stream('shape')
    const n = 80
    const xs = toFlat(uniform(st.child('x'), 0, 1, { shape: [n] }) as Tensor)
    const e = toFlat(normals(st.child('e'), n, 0, 0.3))
    const X = fromData(Float64Array.from(xs), [n, 1])
    const Y = tensor(xs.map((v, i) => 2.5 * v ** 3 + e[i]))
    const m = gam({ terms: [s(0, { k: 15, lambda: 0.01, constraint: 'increasing' })], method: 'fixed' }).fit({
      x: X,
      y: Y,
    })
    expect(m.shape.violations).toBe(0)
    const f = toFlat(m.partial(0, linspace(0, 1, 101)).fit)
    for (let i = 1; i < f.length; i++) expect(f[i]).toBeGreaterThanOrEqual(f[i - 1] - 1e-5)
  })

  it('thin-plate and linear terms', () => {
    const m = gam({ terms: [thinPlate(0, { k: 8 }), linearTerm(1)] }).fit({ x, y })
    expect(m.labels).toEqual(['s(x0)', 'x1'])
    expect(m.termEdf[1]).toBeCloseTo(1, 8)
  })

  it('numeric by terms vary a coefficient', () => {
    const m = gam({ terms: [s(0, { by: 1, k: 8 })], method: 'fixed' }).fit({ x, y })
    expect(m.labels[0]).toBe('s(x0, by = x1)')
  })
})

describe('expectile GAM', () => {
  it('the τ = 0.9 curve has about 90% of points below it, and LAWS converges', () => {
    const m = expectileGam({ terms: [s(0, { k: 10, lambda: 1 })], method: 'fixed', tau: 0.9 }).fit({ x, y })
    const last = m.laws.steps.at(-1)!
    expect(last.converged).toBe(true)
    expect(last.below).toBeGreaterThan(0.7)
    expect(last.below).toBeLessThan(0.97)
  })
})

describe('explainable boosting machine', () => {
  it('reduces the loss and recovers additive shapes', () => {
    const m = explainableBoostingMachine({ rounds: 400, learningRate: 0.05 }).fit({ x, y })
    const loss = toFlat(m.training.series.loss)
    expect(loss.at(-1)!).toBeLessThan(loss[0] * 0.3)
    expect(m.shapes.shape).toEqual([2, 32])
    const s0 = toRows(m.shapes)[0]
    // sin(2πx) is high near x = 0.25 and low near 0.75.
    expect(s0[8]).toBeGreaterThan(s0[24])
  })
  it('classification and the trace protocol', () => {
    const labels = tensor(F.y.map((v) => (v > 0.5 ? 1 : 0)))
    const m = explainableBoostingMachine({ task: 'classification', rounds: 100, learningRate: 0.1 }).fit({
      x,
      y: labels,
    })
    const p = toFlat(m.expect(x))
    expect(p.every((v) => v > 0 && v < 1)).toBe(true)
    const alg = ebmBoosting({ x, y }, { learningRate: 0.1 })
    expect(toFlat(seek(alg, {}, 3).shapes)).toEqual(toFlat(run(alg, {}, 3).shapes))
  })
})
