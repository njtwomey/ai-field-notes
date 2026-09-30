import { describe, expect, it } from 'vitest'
import {
  barycenter1d,
  costMatrix,
  exactTransport,
  gromovWasserstein,
  monotonePlan,
  sinkhorn,
  sinkhornSteps,
  slicedWasserstein,
  uniformWeights,
  wasserstein1d,
} from 'aifn/ot'
import { stream } from 'aifn/random'
import { toFlat } from 'aifn/tensor'
import { extend, run, seek, trace } from 'aifn/trace'
import { fixture } from './fixtures'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const F = fixture<any>('ot')

const close = (a: number[], b: number[], tol: number) => {
  expect(a.length).toBe(b.length)
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(tol))
}

describe('costs and exact transport', () => {
  it('costMatrix is the squared distance by default', () => {
    close(toFlat(costMatrix(F.assignment.x, F.assignment.y)), (F.assignment.cost as number[][]).flat(), 1e-12)
    expect(toFlat(costMatrix([0, 1], [3], { p: 1 }))).toEqual([3, 2])
  })

  it('uniform equal masses use the Hungarian algorithm and match scipy', () => {
    const r = exactTransport(uniformWeights(6), uniformWeights(6), F.assignment.cost)
    expect(r.method).toBe('hungarian')
    expect(toFlat(r.assignment!)).toEqual(F.assignment.cols)
    expect(r.cost).toBeCloseTo(F.assignment.value, 10)
  })

  it('unequal masses solve the linear program, with marginals and duals', () => {
    const e = F.exact
    const r = exactTransport(e.a, e.b, e.cost)
    expect(r.method).toBe('simplex')
    expect(r.cost).toBeCloseTo(e.value, 9)
    const P = toFlat(r.plan)
    for (let i = 0; i < 5; i++) expect(P.slice(i * 4, i * 4 + 4).reduce((s, v) => s + v, 0)).toBeCloseTo(e.a[i], 10)
    const f = toFlat(r.f!)
    const g = toFlat(r.g!)
    const dual = f.reduce((s, v, i) => s + v * e.a[i], 0) + g.reduce((s, v, j) => s + v * e.b[j], 0)
    expect(dual).toBeCloseTo(e.value, 8)
  })
})

describe('Sinkhorn', () => {
  const S = F.sinkhorn
  const opts = { a: S.a, b: S.b, cost: S.cost, epsilon: S.eps, tolerance: 0 }

  it('matches a direct log-domain implementation after 50 iterations', () => {
    const s = run(sinkhornSteps, opts, 50)
    close(toFlat(s.f), S.f, 1e-10)
    close(toFlat(s.g), S.g, 1e-10)
    close(toFlat(s.plan), (S.plan as number[][]).flat(), 1e-12)
    expect(s.transportCost).toBeCloseTo(S.transport, 10)
  })

  it('converges to the exact cost as ε shrinks, with the dual increasing', () => {
    const e = F.exact
    const small = sinkhorn(e.a, e.b, e.cost, { epsilon: 0.005, maxIterations: 5000 })
    expect(small.converged).toBe(true)
    expect(small.transportCost).toBeCloseTo(e.value, 2)
    const t = trace(sinkhornSteps, { a: e.a, b: e.b, cost: e.cost, epsilon: 0.1 }, 30, {
      record: { dual: (s) => s.dual },
    })
    const d = toFlat(t.series.dual)
    for (let k = 2; k < d.length; k++) expect(d[k]).toBeGreaterThanOrEqual(d[k - 1] - 1e-12)
  })

  it('follows the trace protocol', () => {
    expect(toFlat(seek(sinkhornSteps, opts, 7).f)).toEqual(toFlat(run(sinkhornSteps, opts, 7).f))
    const rec = { record: { err: (s: { marginalError: number }) => s.marginalError } }
    const short = trace(sinkhornSteps, opts, 5, rec)
    const long = trace(sinkhornSteps, opts, 12, rec)
    expect(toFlat(extend(short, sinkhornSteps, opts, 7).series.err)).toEqual(toFlat(long.series.err))
  })
})

describe('one-dimensional transport', () => {
  it('W1 matches scipy, plain and weighted', () => {
    const w = F.w1
    expect(wasserstein1d(w.u, w.v)).toBeCloseTo(w.plain, 12)
    expect(wasserstein1d(w.u, w.v, { uWeights: w.uw, vWeights: w.vw })).toBeCloseTo(w.weighted, 12)
  })

  it('W2 between equal-size samples pairs sorted values', () => {
    const x = [3, 1, 2]
    const y = [10, 30, 20]
    expect(wasserstein1d(x, y, { p: 2 })).toBeCloseTo(Math.sqrt((81 + 324 + 729) / 3), 12)
  })

  it('the monotone plan moves every unit of mass and matches W1', () => {
    const p = monotonePlan([0.5, 0.5, 0], [0, 0.25, 0.75])
    expect(toFlat(p.mass).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const cost = toFlat(p.mass).reduce((s, m, k) => s + m * Math.abs(toFlat(p.i)[k] - toFlat(p.j)[k]), 0)
    expect(cost).toBeCloseTo(
      wasserstein1d([0, 1, 2], [0, 1, 2], { uWeights: [0.5, 0.5, 1e-300], vWeights: [1e-300, 0.25, 0.75] }),
      9,
    )
  })

  it('the barycentre of two shifted samples sits halfway', () => {
    const a = Array.from({ length: 200 }, (_, i) => i / 199)
    const b = a.map((v) => v + 4)
    const bar = barycenter1d([a, b], { levels: 50 })
    const q = toFlat(bar.quantiles)
    const qa = toFlat(barycenter1d([a], { levels: 50 }).quantiles)
    q.forEach((v, i) => expect(v).toBeCloseTo(qa[i] + 2, 10))
  })

  it('sliced Wasserstein is zero for identical clouds and grows with a shift', () => {
    const x = F.assignment.x as number[][]
    expect(slicedWasserstein(stream(1), x, x).distance).toBeCloseTo(0, 12)
    const shifted = x.map(([a, b]) => [a + 3, b])
    const d = slicedWasserstein(stream(1), x, shifted, { projections: 200 }).distance
    // For a pure shift t, SW₂² = E[(θ·t)²] = |t|²/2 in two dimensions.
    expect(d).toBeCloseTo(3 / Math.SQRT2, 0)
  })
})

describe('Gromov–Wasserstein', () => {
  it('recovers the matching between a point set and its rotated copy', () => {
    const x = F.assignment.x as number[][]
    const rot = x.map(([a, b]) => [b, -a])
    const cx = costMatrix(x, x, { p: 1 })
    const cy = costMatrix(rot, rot, { p: 1 })
    const r = gromovWasserstein({
      cx,
      cy,
      a: uniformWeights(6),
      b: uniformWeights(6),
      epsilon: 0.01,
      maxIterations: 100,
    })
    const P = toFlat(r.plan)
    for (let i = 0; i < 6; i++) {
      const row = P.slice(i * 6, i * 6 + 6)
      expect(row.indexOf(Math.max(...row))).toBe(i)
    }
  })
})
