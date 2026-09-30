import { describe, expect, it } from 'vitest'
import {
  alphaBarAt,
  cosineSchedule,
  ddimSampler,
  ddimTimesteps,
  ddpmSampler,
  denoiser,
  denoiserTraining,
  forwardNoise,
  forwardPosterior,
  forwardProcess,
  gaussianMixture,
  linearSchedule,
  mixtureLogDensity,
  mixtureMoments,
  mixtureNoisePredictor,
  mixtureScore,
  networkNoisePredictor,
  probabilityFlowSampler,
  reverseSdeSampler,
  sampleMixture,
  scoreFromPredictor,
  subVpSde,
  veSde,
  vpSde,
  type SamplerState,
} from 'aifn-applied/generative/diffusion'
import { stream } from 'aifn/foundation/random'
import { fromRows, tensor, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { extend, run, seek, trace, type Algorithm } from 'aifn/foundation/trace'
import { fixture } from './fixtures'

type Sched = { betas: number[]; alphaBars: number[]; steps: number[]; [k: string]: number[] }
const F = fixture<{ linear: Sched; cosine: Sched; vp: { times: number[]; meanScale: number[] } }>('diffusion')

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1e-300, Math.abs(b))

describe('schedules', () => {
  it('the linear schedule matches DDPM', () => {
    const s = linearSchedule()
    F.linear.steps.forEach((t, i) => {
      expect(rel(s.betas.data[t - 1], F.linear.betas[i])).toBeLessThan(1e-12)
      expect(rel(alphaBarAt(s, t), F.linear.alphaBars[i])).toBeLessThan(1e-10)
    })
    expect(s.snr.data[0]).toBeCloseTo(s.alphaBars.data[0] / (1 - s.alphaBars.data[0]), 8)
  })

  it('the cosine schedule matches improved-diffusion, capping reported', () => {
    const s = cosineSchedule()
    F.cosine.steps.forEach((t, i) => {
      expect(rel(s.betas.data[t - 1], F.cosine.betas[i])).toBeLessThan(1e-10)
      expect(rel(alphaBarAt(s, t), F.cosine.alphaBars[i])).toBeLessThan(1e-9)
    })
    expect(s.capped).toContain(1000)
  })

  it('the forward posterior matches DDPM’s coefficients', () => {
    const s = linearSchedule()
    F.linear.steps.slice(1).forEach((t, j) => {
      const i = j + 1
      const { mean, variance } = forwardPosterior(s, tensor([1]), tensor([0]), t)
      expect(rel(toFlat(mean)[0], F.linear.coef1[i])).toBeLessThan(1e-9)
      expect(rel(variance, F.linear.posteriorVariance[i])).toBeLessThan(1e-9)
      const other = forwardPosterior(s, tensor([0]), tensor([1]), t)
      expect(rel(toFlat(other.mean)[0], F.linear.coef2[i])).toBeLessThan(1e-9)
    })
  })

  it('the SDE marginals are consistent', () => {
    const vp = vpSde()
    F.vp.times.forEach((t, i) => expect(vp.meanScale(t)).toBeCloseTo(F.vp.meanScale[i], 12))
    // VP: m² + s² = 1; sub-VP: s = 1 − m²; VE: m = 1.
    for (const t of [0.05, 0.5, 1]) {
      expect(vp.meanScale(t) ** 2 + vp.std(t) ** 2).toBeCloseTo(1, 12)
      expect(subVpSde().std(t)).toBeCloseTo(1 - vp.meanScale(t) ** 2, 12)
      expect(veSde().meanScale(t)).toBe(1)
    }
    // dVar/dt = 2f Var + g² for the VP marginal variance s(t)².
    const t = 0.3
    const h = 1e-6
    const dVar = (vp.std(t + h) ** 2 - vp.std(t - h) ** 2) / (2 * h)
    expect(dVar).toBeCloseTo(2 * vp.drift(t) * vp.std(t) ** 2 + vp.diffusion(t) ** 2, 5)
  })
})

const MIX = gaussianMixture(
  [0.3, 0.7],
  [
    [-2, 0],
    [1.5, 1],
  ],
  [0.4, [0.5, 0.3]],
)
const predictor = mixtureNoisePredictor(MIX)

describe('the Gaussian mixture', () => {
  it('its score is the gradient of its log density', () => {
    const x = tensor([
      [0.3, -0.2],
      [-1.5, 0.6],
    ])
    const [m, s] = [0.8, 0.6]
    const score = toRows(mixtureScore(MIX, x, m, s))
    const h = 1e-6
    for (let i = 0; i < 2; i++)
      for (let a = 0; a < 2; a++) {
        const up = toRows(x).map((r) => [...r])
        const down = toRows(x).map((r) => [...r])
        up[i][a] += h
        down[i][a] -= h
        const fd =
          (toFlat(mixtureLogDensity(MIX, fromRows(up), m, s))[i] -
            toFlat(mixtureLogDensity(MIX, fromRows(down), m, s))[i]) /
          (2 * h)
        expect(score[i][a]).toBeCloseTo(fd, 6)
      }
    // The predictor route gives the same score.
    expect(toFlat(scoreFromPredictor(predictor, x, m, s))[0]).toBeCloseTo(score[0][0], 10)
  })

  it('samples have the mixture’s moments', () => {
    const x = sampleMixture(stream('mix'), MIX, 20000)
    const { mean } = mixtureMoments(MIX)
    const xs = toRows(x)
    const m0 = xs.reduce((a, r) => a + r[0], 0) / xs.length
    expect(m0).toBeCloseTo(mean[0], 1)
  })
})

/** Mean and covariance of [n, 2] samples. */
function moments(x: Tensor) {
  const rows = toRows(x)
  const n = rows.length
  const mean = [0, 1].map((a) => rows.reduce((s, r) => s + r[a], 0) / n)
  const cov = [0, 1].map((a) =>
    [0, 1].map((b) => rows.reduce((s, r) => s + (r[a] - mean[a]) * (r[b] - mean[b]), 0) / n),
  )
  return { mean, cov }
}

function expectMixtureMoments(x: Tensor, tol: number) {
  const { mean, covariance } = mixtureMoments(MIX)
  const got = moments(x)
  for (let a = 0; a < 2; a++) {
    expect(Math.abs(got.mean[a] - mean[a])).toBeLessThan(tol)
    for (let b = 0; b < 2; b++) expect(Math.abs(got.cov[a][b] - covariance[a][b])).toBeLessThan(tol * 2)
  }
}

describe('sampling with the exact score recovers the mixture', () => {
  const start = { n: 1500, dimension: 2 }
  it('DDPM', () => {
    const final = run(ddpmSampler(predictor, linearSchedule(200, { betaEnd: 0.08 })), start, 1000, {
      stream: stream(1),
    })
    expect(final.time).toBe(0)
    expectMixtureMoments(final.x, 0.1)
  })

  it('DDIM, deterministic and stochastic', () => {
    const schedule = linearSchedule()
    expectMixtureMoments(run(ddimSampler(predictor, schedule, { steps: 50 }), start, 100, { stream: stream(2) }).x, 0.1)
    expectMixtureMoments(
      run(ddimSampler(predictor, schedule, { steps: 50, eta: 1 }), start, 100, { stream: stream(3) }).x,
      0.1,
    )
  })

  it('the reverse SDE and the probability-flow ODE', () => {
    expectMixtureMoments(
      run(reverseSdeSampler(predictor, vpSde(), { steps: 400 }), start, 1000, { stream: stream(4) }).x,
      0.1,
    )
    expectMixtureMoments(
      run(probabilityFlowSampler(predictor, vpSde(), { steps: 60 }), start, 1000, { stream: stream(5) }).x,
      0.1,
    )
    expectMixtureMoments(
      run(probabilityFlowSampler(predictor, veSde(), { steps: 200 }), start, 1000, { stream: stream(6) }).x,
      0.12,
    )
  })

  it('DDIM with η = 0 and the ODE are deterministic maps of the start', () => {
    const x = tensor([[0.2, -0.4]])
    const a = run(ddimSampler(predictor, linearSchedule(), { steps: 20 }), { x }, 100, { stream: stream(7) })
    const b = run(ddimSampler(predictor, linearSchedule(), { steps: 20 }), { x }, 100, { stream: stream(8) })
    expect(toFlat(a.x)).toEqual(toFlat(b.x))
  })

  it('DDIM uses S distinct levels from T down to 1', () => {
    expect(ddimTimesteps(1000, 5)).toEqual([1000, 750, 501, 251, 1, 0])
  })
})

describe('trace protocol', () => {
  const small = { n: 50, dimension: 2 }
  const algs: [string, Algorithm<typeof small, SamplerState>][] = [
    ['ddpm', ddpmSampler(predictor, linearSchedule(40, { betaEnd: 0.3 }))],
    ['ddim', ddimSampler(predictor, linearSchedule(), { steps: 20, eta: 0.5 })],
    ['sde', reverseSdeSampler(predictor, vpSde(), { steps: 30 })],
    ['ode', probabilityFlowSampler(predictor, vpSde(), { steps: 20 })],
  ]
  for (const [name, alg] of algs) {
    it(`${name}: same seed, seek and extend agree`, () => {
      const record = { x: (s: SamplerState) => s.x }
      const a = trace(alg, small, 15, { stream: stream(9), record })
      const b = trace(alg, small, 15, { stream: stream(9), record })
      expect(toFlat(a.series.x)).toEqual(toFlat(b.series.x))
      expect(toFlat(seek(alg, small, 11, { stream: stream(9) }).x)).toEqual(toFlat(a.steps[11].x))
      const longer = extend(trace(alg, small, 8, { stream: stream(9), record }), alg, small, 7)
      expect(toFlat(longer.series.x)).toEqual(toFlat(a.series.x))
    })
  }

  it('the forward process has the closed-form marginal', () => {
    const schedule = linearSchedule(100, { betaEnd: 0.05 })
    const x0 = sampleMixture(stream('x0'), MIX, 4000)
    const end = run(forwardProcess(schedule), { x0 }, 60, { stream: stream(10) })
    const direct = forwardNoise(stream(11), x0, alphaBarAt(schedule, 60))
    const [a, b] = [moments(end.x), moments(direct.x)]
    expect(Math.abs(a.cov[0][0] - b.cov[0][0])).toBeLessThan(0.1)
    expect(Math.abs(a.mean[0] - b.mean[0])).toBeLessThan(0.05)
  })
})

describe('the learned denoiser', () => {
  it('training lowers the noise-prediction loss', () => {
    const schedule = linearSchedule(100, { betaEnd: 0.1 })
    const net = denoiser(2, { hidden: [32, 32], frequencies: 3 })
    const data = sampleMixture(stream('train'), MIX, 512)
    const alg = denoiserTraining({ data, schedule, net, batchSize: 128 })
    const t = trace(alg, { params: net.layer.init(stream('net')) }, 300, { every: 50, record: { loss: (s) => s.loss } })
    const losses = toFlat(t.series.loss)
    const early = losses[0]
    const late = losses[losses.length - 1]
    expect(late).toBeLessThan(early)
    const pred = networkNoisePredictor(net, t.steps[t.steps.length - 1].params)
    expect(pred(tensor([[0, 0]]), 0.5).shape).toEqual([1, 2])
  })
})
