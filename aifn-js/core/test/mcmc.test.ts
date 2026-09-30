import {
  autocorrelation,
  bivariateGaussianConditionals,
  effectiveSampleSize,
  essFromLogWeights,
  gibbs,
  hmc,
  independenceMetropolis,
  integratedAutocorrelationTime,
  leapfrog,
  mala,
  monteCarloStandardError,
  nuts,
  particleFilter,
  randomWalkMetropolis,
  resample,
  resamplingSchemes,
  sampleChains,
  sgld,
  sliceSampler,
  splitRhat,
  summarise,
  temperedSmc,
  unadjustedLangevin,
  type ChainStart,
  type Target,
} from 'aifn/inference/stochastic'
import { banana, funnel, gaussianMixtureTarget, gaussianTarget } from 'aifn-applied/data/targets'
import { normal, normals, stream } from 'aifn/foundation/random'
import { mul, square, sum, tensor, toFlat, type Tensor, type Value, type Vector } from 'aifn/foundation/tensor'
import { extend, run, seek, trace, type Algorithm } from 'aifn/foundation/trace'
import { describe, expect, it } from 'vitest'
import { fixture } from './fixtures'

const std2 = gaussianTarget(
  [0, 0],
  [
    [1, 0.5],
    [0.5, 1],
  ],
)

/** Mean and variance of column k of an m×n×d draws tensor, pooled. */
function moments(draws: Tensor, k: number) {
  const [m, n, d] = draws.shape
  const x = toFlat(draws)
  let s = 0
  let s2 = 0
  for (let i = 0; i < m * n; i++) {
    s += x[i * d + k]
    s2 += x[i * d + k] ** 2
  }
  const mean = s / (m * n)
  return { mean, variance: s2 / (m * n) - mean * mean }
}

describe('targets', () => {
  it.each([
    ['banana', banana({ a: 1.2, b: 0.7 })],
    ['gaussian', std2],
    [
      'mixture',
      gaussianMixtureTarget(
        [
          [-1, 0],
          [2, 1],
        ],
        0.8,
        [1, 2],
      ),
    ],
    ['funnel', funnel({ dim: 3 })],
  ] as [string, Target][])('%s: analytic gradient matches finite differences', (_, t) => {
    const x = Array.from({ length: t.dim }, (_, i) => 0.3 + 0.2 * i)
    const g = toFlat(tensor(Array.from(t.grad!(tensor(x)) as ArrayLike<number>)))
    const f = (v: number[]) => t.logDensity(tensor(v)) as number
    for (let i = 0; i < t.dim; i++) {
      const up = [...x]
      const dn = [...x]
      up[i] += 1e-5
      dn[i] -= 1e-5
      expect(g[i]).toBeCloseTo((f(up) - f(dn)) / 2e-5, 5)
    }
  })

  it('gaussianTarget is normalised', () => {
    // At the mean, log N = −½ log|2πΣ| with |Σ| = 0.75.
    expect(std2.logDensity(tensor([0, 0]))).toBeCloseTo(-Math.log(2 * Math.PI) - 0.5 * Math.log(0.75), 12)
  })
})

describe('samplers recover known moments', () => {
  const start: ChainStart = { x0: [0.5, -0.5] }
  it.each([
    ['random-walk', randomWalkMetropolis(std2, { scale: 1.2 }), 3000],
    ['hmc', hmc(std2, { stepSize: 0.3, steps: 8 }), 800],
    ['nuts', nuts(std2, { stepSize: 0.4 }), 800],
    ['mala', mala(std2, { stepSize: 0.4 }), 3000],
    ['slice', sliceSampler(std2, { width: 2 }), 1500],
    ['gibbs', gibbs(bivariateGaussianConditionals(0.5)), 1500],
  ] as [string, Algorithm<ChainStart, { x: Vector }>, number][])('%s', (_, alg, steps) => {
    const { draws } = sampleChains(alg, start, { chains: 4, steps, stream: stream(11), warmup: 100 })
    for (const k of [0, 1]) {
      const { mean, variance } = moments(draws, k)
      expect(Math.abs(mean)).toBeLessThan(0.12)
      expect(Math.abs(variance - 1)).toBeLessThan(0.15)
    }
    expect(Math.max(...toFlat(splitRhat(draws) as Tensor))).toBeLessThan(1.05)
  })

  it('banana moments under HMC', () => {
    const t = banana({ a: 1, b: 0.5 })
    const { draws } = sampleChains(
      hmc(t, { stepSize: 0.2, steps: 15 }),
      { x0: [0, 0] },
      {
        chains: 4,
        steps: 1500,
        stream: stream(3),
        warmup: 100,
      },
    )
    expect(Math.abs(moments(draws, 0).mean)).toBeLessThan(0.1)
    expect(moments(draws, 1).variance).toBeCloseTo(1.5, 0)
  })

  it('independence sampler with a wide Gaussian proposal', () => {
    const target = gaussianTarget([1], [[0.25]])
    const alg = independenceMetropolis(target, {
      sample: (s) => [normal(s, 0, 2)],
      logDensity: (x) => -0.5 * (toFlat(x)[0] / 2) ** 2,
    })
    const { draws } = sampleChains(alg, { x0: [0] }, { chains: 2, steps: 4000, stream: stream(2) })
    expect(moments(draws, 0).mean).toBeCloseTo(1, 1)
    expect(moments(draws, 0).variance).toBeCloseTo(0.25, 1)
  })

  it('ULA is biased by O(h) and MALA is not', () => {
    const t = gaussianTarget([0], [[1]])
    const h = 0.5
    const run2 = (alg: Algorithm<ChainStart, { x: Vector }>) =>
      moments(sampleChains(alg, { x0: [0] }, { chains: 4, steps: 5000, stream: stream(8), warmup: 100 }).draws, 0)
    // ULA's stationary variance is σ²/(1 − h/(2σ²)) = 4/3.
    expect(run2(unadjustedLangevin(t, { stepSize: h })).variance).toBeCloseTo(4 / 3, 1)
    expect(run2(mala(t, { stepSize: h })).variance).toBeCloseTo(1, 1)
  })

  it('SGLD samples a Gaussian-mean posterior', () => {
    // y_i ~ N(θ, 1), prior N(0, 10²): posterior N(Σy/(N + 0.01), 1/(N + 0.01)).
    const ys = toFlat(normals(stream('data'), 100, 1, 1))
    const N = ys.length
    const post = ys.reduce((a, b) => a + b, 0) / (N + 0.01)
    const alg = sgld(
      {
        dim: 1,
        size: N,
        gradLogPrior: (th) => [-toFlat(th)[0] / 100],
        gradLogLikelihood: (th, i) => [ys[i] - toFlat(th)[0]],
      },
      { batchSize: 20, stepSize: 2e-3 },
    )
    const { draws } = sampleChains(alg, { x0: [0] }, { chains: 2, steps: 4000, stream: stream(4), warmup: 500 })
    expect(moments(draws, 0).mean).toBeCloseTo(post, 1)
    expect(moments(draws, 0).variance).toBeGreaterThan(0.005)
    expect(moments(draws, 0).variance).toBeLessThan(0.03)
  })

  it('autodiff supplies the gradient when the target has none', () => {
    const t: Target = { dim: 2, logDensity: (x) => mul(-0.5, sum(square(x as Value))) as number }
    const withGrad = hmc(t, { stepSize: 0.2, steps: 5 })
    const s = run(withGrad, { x0: [1, 1] }, 3, { stream: stream(1) })
    const ref = run(
      hmc({ ...t, grad: (x) => toFlat(x).map((v) => -v) }, { stepSize: 0.2, steps: 5 }),
      { x0: [1, 1] },
      3,
      {
        stream: stream(1),
      },
    )
    expect(toFlat(s.x)).toEqual(toFlat(ref.x).map((v) => expect.closeTo(v, 12)) as unknown as number[])
  })
})

describe('internals', () => {
  it('leapfrog conserves energy approximately and is reversible', () => {
    const lf = leapfrog(std2, [1, 0], [0.3, -0.7], { stepSize: 0.1, steps: 50 })
    const H = toFlat(lf.energies)
    expect(Math.max(...H) - Math.min(...H)).toBeLessThan(0.01)
    const P = toFlat(lf.positions)
    const M = toFlat(lf.momenta)
    const end = P.slice(-2)
    const back = leapfrog(
      std2,
      end,
      M.slice(-2).map((v) => -v),
      { stepSize: 0.1, steps: 50 },
    )
    const B = toFlat(back.positions)
    expect(B.slice(-2)[0]).toBeCloseTo(1, 10)
    expect(B.slice(-2)[1]).toBeCloseTo(0, 10)
  })

  it('HMC with a huge step flags divergences on the funnel', () => {
    const tr = trace(hmc(funnel(), { stepSize: 1.5, steps: 20 }), { x0: [-2, 0.1] }, 50, { stream: stream(1) })
    expect(tr.steps.at(-1)!.divergentCount).toBeGreaterThan(0)
  })

  it('NUTS builds trees and stops on a U-turn', () => {
    const s = run(nuts(std2, { stepSize: 0.2 }), { x0: [0, 0] }, 20, { stream: stream(2) })
    expect(s.treeDepth).toBeGreaterThan(0)
    expect(s.trajectory.shape[0]).toBe(s.leapfrogSteps + 1)
    expect(s.hitMaxDepth).toBe(false)
  })

  it('Gibbs moves are axis-parallel', () => {
    const s = run(gibbs(bivariateGaussianConditionals(0.9)), { x0: [2, -2] }, 3, { stream: stream(1) })
    const m = toFlat(s.moves)
    expect(m[2 + 1]).toBe(m[1]) // first update changes x₀ only
    expect(m[4]).toBe(m[2]) // second changes x₁ only
  })

  it('random-walk acceptance falls as the scale grows', () => {
    const rate = (scale: number) =>
      run(randomWalkMetropolis(banana(), { scale }), { x0: [0, 0] }, 2000, { stream: stream(1) }).acceptanceRate
    expect(rate(0.1)).toBeGreaterThan(rate(1))
    expect(rate(1)).toBeGreaterThan(rate(5))
  })
})

describe('protocol', () => {
  const algs: Record<string, Algorithm<ChainStart, { x: Vector }>> = {
    rwm: randomWalkMetropolis(banana()),
    hmc: hmc(banana(), { stepSize: 0.1, steps: 5 }),
    nuts: nuts(banana()),
    mala: mala(banana()),
    ula: unadjustedLangevin(banana()),
    slice: sliceSampler(banana()),
    gibbs: gibbs(bivariateGaussianConditionals(0.8), { scan: 'random' }),
  }
  it.each(Object.entries(algs))(
    '%s: same stream → same trace; seek equals run; extend equals a longer trace',
    (_, alg) => {
      const opts = { x0: [0.2, 0.1] }
      const record = { x: (s: { x: Vector }) => s.x }
      const a = trace(alg, opts, 30, { record, stream: stream(5) })
      const b = trace(alg, opts, 30, { record, stream: stream(5) })
      expect(toFlat(a.series.x)).toEqual(toFlat(b.series.x))
      const r = run(alg, opts, 17, { stream: stream(5) })
      expect(toFlat(seek(alg, opts, 17, { stream: stream(5) }).x)).toEqual(toFlat(r.x))
      const short = trace(alg, opts, 12, { record, stream: stream(5), checkpointEvery: 5 })
      expect(toFlat(seek(alg, opts, 17, { stream: stream(5), checkpoints: short }).x)).toEqual(toFlat(r.x))
      const longer = extend(short, alg, opts, 18)
      expect(toFlat(longer.series.x)).toEqual(toFlat(a.series.x))
    },
  )
})

describe('sequential Monte Carlo', () => {
  it.each(resamplingSchemes)('%s resampling is unbiased and sorted', (scheme) => {
    const w = [0.1, 0.4, 0.05, 0.45]
    const counts = [0, 0, 0, 0]
    const R = 2000
    for (let r = 0; r < R; r++) for (const i of toFlat(resample(stream(r), w, scheme, 10))) counts[i]++
    counts.forEach((c, i) => expect(c / (10 * R)).toBeCloseTo(w[i], 2))
    const one = toFlat(resample(stream(1), w, scheme, 10))
    expect([...one].sort((a, b) => a - b)).toEqual(one)
    expect(resample(stream(1), w, scheme).dtype).toBe('int32')
  })

  it('systematic resampling keeps ⌊Nw⌋ copies', () => {
    const c = toFlat(resample(stream(3), [0.5, 0.25, 0.25], 'systematic', 8))
    expect(c.filter((i) => i === 0).length).toBe(4)
  })

  it('ESS from log-weights', () => {
    expect(essFromLogWeights([0, 0, 0, 0])).toBeCloseTo(4, 12)
    expect(essFromLogWeights([0, -1000, -1000])).toBeCloseTo(1, 12)
  })

  it('particle filter on a linear-Gaussian model matches the Kalman filter', () => {
    // xₜ = 0.9xₜ₋₁ + N(0, 0.5²), x₀ ~ N(0, 1), yₜ = xₜ + N(0, 0.4²).
    const ys: number[] = []
    let x = normal(stream('x0'), 0, 1)
    for (let t = 0; t < 30; t++) {
      if (t > 0) x = 0.9 * x + normal(stream('tr').child(t), 0, 0.5)
      ys.push(x + normal(stream('ob').child(t), 0, 0.4))
    }
    const model = {
      dim: 1,
      sampleInitial: (s: ReturnType<typeof stream>) => normal(s, 0, 1),
      sampleTransition: (v: Vector, _t: number, s: ReturnType<typeof stream>) => normal(s, 0.9 * toFlat(v)[0], 0.5),
      logObservation: (y: number, v: Vector) =>
        -0.5 * ((y - toFlat(v)[0]) / 0.4) ** 2 - Math.log(0.4 * Math.sqrt(2 * Math.PI)),
    }
    const tr = trace(particleFilter(model, { particles: 3000 }), { observations: ys }, 100, {
      stream: stream(1),
      record: { mean: (s) => s.mean },
    })
    expect(tr.meta.stopped).toBe('done')
    // Kalman filter reference.
    let m = 0
    let P = 1
    let logZ = 0
    const kf: number[] = []
    ys.forEach((y, t) => {
      if (t > 0) {
        m = 0.9 * m
        P = 0.81 * P + 0.25
      }
      const S = P + 0.16
      logZ += -0.5 * Math.log(2 * Math.PI * S) - (0.5 * (y - m) ** 2) / S
      const K = P / S
      m += K * (y - m)
      P *= 1 - K
      kf.push(m)
    })
    const means = toFlat(tr.series.mean).slice(1)
    means.forEach((v, t) => expect(Math.abs(v - kf[t])).toBeLessThan(0.08))
    expect(tr.steps.at(-1)!.logEvidence).toBeCloseTo(logZ, 0)
  })

  it('tempered SMC estimates the evidence of a conjugate Gaussian model', () => {
    // θ ~ N(0, 1), y ~ N(θ, 0.5²) with y = 1.2: Z = N(1.2 | 0, 1.25).
    const y = 1.2
    const s = run(
      temperedSmc(
        {
          dim: 1,
          samplePrior: (st) => normal(st, 0, 1),
          logPrior: (v) => -0.5 * toFlat(v)[0] ** 2 - 0.5 * Math.log(2 * Math.PI),
          logLikelihood: (v) => -0.5 * ((y - toFlat(v)[0]) / 0.5) ** 2 - Math.log(0.5 * Math.sqrt(2 * Math.PI)),
        },
        { particles: 2000 },
      ),
      {},
      50,
      { stream: stream(2) },
    )
    expect(s.beta).toBe(1)
    expect(s.logEvidence).toBeCloseTo(-0.5 * Math.log(2 * Math.PI * 1.25) - (0.5 * y * y) / 1.25, 1)
    const p = toFlat(s.particles)
    const mean = p.reduce((a, b) => a + b, 0) / p.length
    expect(mean).toBeCloseTo(y / 1.25, 1)
  })
})

describe('diagnostics against ArviZ', () => {
  type Case = {
    draws: number[][]
    ess_bulk: number
    ess_mean: number
    ess_tail: number
    iact: number
    rhat_rank: number
    rhat_split: number
    rhat_basic: number
    mcse_mean: number
    acf: number[]
  }
  const cases = fixture<Record<string, Case>>('mcmc')
  it.each(Object.entries(cases))('%s', (_, c) => {
    const x = c.draws
    expect(effectiveSampleSize(x)).toBeCloseTo(c.ess_bulk, 8)
    expect(effectiveSampleSize(x, { method: 'mean' })).toBeCloseTo(c.ess_mean, 8)
    expect(effectiveSampleSize(x, { method: 'tail' })).toBeCloseTo(c.ess_tail, 8)
    expect(integratedAutocorrelationTime(x)).toBeCloseTo(c.iact, 8)
    expect(splitRhat(x)).toBeCloseTo(c.rhat_rank, 10)
    expect(splitRhat(x, { method: 'split' })).toBeCloseTo(c.rhat_split, 10)
    if (x.length > 1) expect(splitRhat(x, { method: 'basic' })).toBeCloseTo(c.rhat_basic, 10)
    expect(monteCarloStandardError(x)).toBeCloseTo(c.mcse_mean, 10)
    const acf = toFlat(autocorrelation(x[0], { maxLag: c.acf.length - 1 }))
    acf.forEach((v, k) => expect(v).toBeCloseTo(c.acf[k], 10))
  })

  it('per-parameter diagnostics for m×n×d draws, and edge cases', () => {
    const { draws } = sampleChains(
      randomWalkMetropolis(std2),
      { x0: [0, 0] },
      { chains: 2, steps: 200, stream: stream(1) },
    )
    expect(effectiveSampleSize(draws)).toHaveProperty('shape', [2])
    expect(effectiveSampleSize([1, 1, 1, 1, 1, 1, 1, 1])).toBe(8)
    expect(effectiveSampleSize([1, 2, 3])).toBeNaN()
    expect(summarise([toFlat(normals(stream(1), 400)), toFlat(normals(stream(2), 400))]).rhat).toBeLessThan(1.02)
  })

  it('shifted chains give a large R̂ and a small ESS', () => {
    const a = toFlat(normals(stream(1), 500))
    const b = toFlat(normals(stream(2), 500)).map((v) => v + 3)
    expect(splitRhat([a, b])).toBeGreaterThan(1.5)
    expect(effectiveSampleSize([a, b])).toBeLessThan(50)
  })
})
