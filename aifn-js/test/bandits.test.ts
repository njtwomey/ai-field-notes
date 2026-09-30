import { describe, expect, it } from 'vitest'
import {
  banditRun,
  bernoulliBandit,
  epsilonGreedy,
  exp3,
  exploreThenCommit,
  gaussianBandit,
  klBernoulli,
  klUcb,
  klUcbIndex,
  laiRobbinsBound,
  linearBandit,
  linearThompson,
  linUcb,
  regretCurves,
  thompsonBernoulli,
  thompsonGaussian,
  ucb1,
  uniformPolicy,
} from 'aifn/bandits'
import { stream } from 'aifn/random'
import { toFlat } from 'aifn/tensor'
import { extend, run, seek, trace } from 'aifn/trace'

const env = bernoulliBandit([0.3, 0.5, 0.7])

describe('bandit helpers', () => {
  it('KL-UCB index solves n kl(p, q) = level', () => {
    expect(klBernoulli(0.5, 0.5)).toBe(0)
    const q = klUcbIndex(0.4, 20, Math.log(100))
    expect(20 * klBernoulli(0.4, q)).toBeCloseTo(Math.log(100), 6)
    expect(klUcbIndex(1, 5, 1)).toBe(1)
  })
  it('Lai–Robbins constant', () => {
    const r = laiRobbinsBound([0.5, 0.6], [10])
    expect(r.constant).toBeCloseTo(0.1 / klBernoulli(0.5, 0.6), 12)
  })
})

describe('regret', () => {
  it('UCB1, KL-UCB and Thompson sampling beat uniform play and grow sublinearly', () => {
    const r = regretCurves(env, [uniformPolicy(), ucb1(), klUcb(), thompsonBernoulli()], {
      horizon: 1000,
      runs: 20,
      stream: stream(1),
      points: 10,
    })
    const final = toFlat(r.mean).filter((_, i) => i % 10 === 9)
    expect(final[0]).toBeCloseTo(0.4 * 1000 * (1 / 3) + 0.2 * 1000 * (1 / 3), -1)
    for (const k of [1, 2, 3]) expect(final[k]).toBeLessThan(final[0] / 4)
    // Thompson sampling and KL-UCB are near-optimal for Bernoulli arms: well below UCB1.
    expect(final[3]).toBeLessThan(final[1])
    const pulls = toFlat(r.pulls)
    expect(pulls.slice(9, 12).reduce((a, b) => a + b, 0)).toBeCloseTo(1000, 6)
    expect(pulls[11]).toBeGreaterThan(800)
  })

  it('every context-free policy runs on Bernoulli and Gaussian arms', () => {
    const policies = [
      exploreThenCommit({ m: 10 }),
      epsilonGreedy(),
      epsilonGreedy({ decay: 5 }),
      exp3(),
      thompsonGaussian(),
    ]
    for (const e of [env, gaussianBandit([0, 0.5, 1])]) {
      const r = regretCurves(e, policies, { horizon: 300, runs: 3, stream: stream(2), points: 5 })
      expect(toFlat(r.mean).every(Number.isFinite)).toBe(true)
    }
  })

  it('LinUCB and linear Thompson sampling learn a linear bandit', () => {
    const lin = linearBandit({ theta: [1, 0.5], mode: 'random' })
    const r = regretCurves(lin, [linUcb({ alpha: 0.5 }), linearThompson(), linUcb({ alpha: 0 })], {
      horizon: 300,
      runs: 5,
      stream: stream(3),
      points: 3,
    })
    const m = toFlat(r.mean)
    // The per-round regret late in the run is far below the early rate.
    expect(m[2] - m[1]).toBeLessThan(m[0])
  })
})

describe('banditRun follows the trace protocol', () => {
  const alg = banditRun(env, thompsonBernoulli())
  const opts = {}
  it('same stream, same trace; seek equals run; extend equals a longer trace', () => {
    const rec = { record: { regret: (s: { cumulativeRegret: number }) => s.cumulativeRegret } }
    const a = trace(alg, opts, 50, { ...rec, stream: stream(4) })
    const b = trace(alg, opts, 50, { ...rec, stream: stream(4) })
    expect(toFlat(a.series.regret)).toEqual(toFlat(b.series.regret))
    expect(seek(alg, opts, 20, { stream: stream(4) }).arm).toBe(run(alg, opts, 20, { stream: stream(4) }).arm)
    const short = trace(alg, opts, 20, { ...rec, stream: stream(4) })
    expect(toFlat(extend(short, alg, opts, 30).series.regret)).toEqual(toFlat(a.series.regret))
  })
  it('policies on one stream see the same rewards (common random numbers)', () => {
    const u = run(banditRun(env, uniformPolicy()), opts, 30, { stream: stream(5) })
    const v = run(banditRun(env, uniformPolicy()), opts, 30, { stream: stream(5) })
    expect(u.cumulativeReward).toBe(v.cumulativeReward)
  })
})
