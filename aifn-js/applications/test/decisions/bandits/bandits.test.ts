import { describe, expect, it } from 'vitest'
import {
  banditRun,
  epsilonGreedy,
  exp3,
  exploreThenCommit,
  klBernoulli,
  klUcb,
  klUcbIndex,
  laiRobbinsBound,
  linearThompson,
  linUcb,
  regretCurves,
  thompsonBernoulli,
  thompsonGaussian,
  ucb1,
  uniformPolicy,
} from 'aifn-applied/decisions/bandits'
import { bernoulliBandit, gaussianBandit, linearBandit } from 'aifn-applied/data/environments'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { run } from 'aifn/foundation/trace'
import { expectProtocol } from '../../protocol'

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
  it('every policy: same key, same trace; seek = run; extend = longer trace; clones step the same', () => {
    const lin = linearBandit({ theta: [1, 0.5], mode: 'random' })
    const record = { regret: (s: { cumulativeRegret: number }) => s.cumulativeRegret }
    for (const p of [
      uniformPolicy(),
      ucb1(),
      klUcb(),
      thompsonBernoulli(),
      exploreThenCommit({ m: 2 }),
      epsilonGreedy(),
      exp3(),
      thompsonGaussian(),
    ])
      expectProtocol(banditRun(env, p), undefined, { n: 30, record })
    for (const p of [linUcb({ alpha: 0.5 }), linearThompson()])
      expectProtocol(banditRun(lin, p), undefined, { n: 20, record })
  })
  it('policies on one stream see the same rewards (common random numbers)', () => {
    const u = run(banditRun(env, uniformPolicy()), undefined, 30, { stream: stream(5) })
    const v = run(banditRun(env, uniformPolicy()), undefined, 30, { stream: stream(5) })
    expect(u.cumulativeReward).toBe(v.cumulativeReward)
  })
})
