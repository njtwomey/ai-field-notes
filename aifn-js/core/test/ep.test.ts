import { describe, expect, it } from 'vitest'
import { Beta, Dirichlet, Gamma, Normal } from 'aifn/probability/distributions'
import {
  adfSteps,
  dampGaussian,
  divideGaussians,
  epLogEvidence,
  expectationPropagationSteps,
  gaussianMoments,
  intervalTilted,
  messageOf,
  messageToDistribution,
  multiplyGaussians,
  multiplyMessages,
  naturalGaussian,
  probitTilted,
  stepTilted,
  tiltedByQuadrature,
  type EpOptions,
} from 'aifn/inference/expectation-propagation'
import { bayesPointMachinePredict, bayesPointMachineSteps } from 'aifn-applied/inference/classifier-models'
import {
  clutterEp,
  clutterLogLikelihood,
  clutterPosterior,
  clutterTilted,
  sampleClutter,
} from 'aifn-applied/inference/mixture-models'
import {
  drawMargin,
  trueSkillEpSteps,
  trueSkillUpdate,
  type TrueSkillEpState,
} from 'aifn-applied/inference/rating-models'
import { integrate } from 'aifn/numerics/quadrature'
import { stream } from 'aifn/foundation/random'
import { normalCdf, normalPdf } from 'aifn/numerics/special'
import { tensor, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { extend, run, seek, trace, type Algorithm } from 'aifn/foundation/trace'

/**
 * ∫ f over [a, b] as a sum over unit panels: a single adaptive rule over a wide range can miss a narrow peak (its
 * first Kronrod estimate sees nothing there and reports convergence).
 */
function panels(f: (t: number) => number, a: number, b: number): number {
  let total = 0
  for (let lo = a; lo < b; lo += 1) total += integrate(f, lo, Math.min(lo + 1, b), { rtol: 1e-13 }).value
  return total
}

/** Moments of N(θ; m, v) f(θ) on [lower, upper] (default ±12 sd) by quadrature: log Z, mean, variance. */
function quadTilted(m: number, v: number, f: (t: number) => number, bounds: [number, number] = [-Infinity, Infinity]) {
  const s = Math.sqrt(v)
  const w = (t: number) => (Math.exp((-0.5 * (t - m) ** 2) / v) / Math.sqrt(2 * Math.PI * v)) * f(t)
  const lo = Math.max(m - 12 * s, bounds[0])
  const hi = Math.min(m + 12 * s, bounds[1])
  const z = panels(w, lo, hi)
  const mean = panels((t) => t * w(t), lo, hi) / z
  const second = panels((t) => (t - mean) ** 2 * w(t), lo, hi) / z
  return { logZ: Math.log(z), mean, variance: second }
}

describe('message algebra', () => {
  it('multiply, divide and damp in natural parameters', () => {
    const a = naturalGaussian(1, 2)
    const b = naturalGaussian(-1, 0.5)
    const ab = multiplyGaussians(a, b)
    const m = gaussianMoments(ab)
    expect(m.variance).toBeCloseTo(1 / (1 / 2 + 1 / 0.5), 14)
    expect(m.mean).toBeCloseTo(m.variance * (1 / 2 - 1 / 0.5), 14)
    const back = divideGaussians(ab, b)
    expect(back.precision).toBeCloseTo(a.precision, 14)
    expect(dampGaussian(a, b, 0.25).precision).toBeCloseTo(0.75 * a.precision + 0.25 * b.precision, 14)
  })
  it('batches broadcast elementwise', () => {
    const g = naturalGaussian(tensor([0, 1]), tensor([1, 4]))
    expect(toFlat(gaussianMoments(multiplyGaussians(g, g)).variance as Tensor)).toEqual([0.5, 2])
  })
  it('exponential-family messages round-trip through distribution objects', () => {
    for (const d of [Normal(1, 2), Gamma(3, 2), Beta(2, 5), Dirichlet(tensor([1, 2, 3]))]) {
      const back = messageToDistribution(messageOf(d))
      expect(back.name).toBe(d.name)
      expect(toFlat(tensor(back.mean() as number) as Tensor)).toEqual(
        toFlat(tensor(d.mean() as number) as Tensor).map((x) => expect.closeTo(x, 12)),
      )
    }
    // Beta(2, 5) · Beta(3, 1) ∝ Beta(4, 5): natural parameters add.
    const prod = messageToDistribution(multiplyMessages(messageOf(Beta(2, 5)), messageOf(Beta(3, 1))))
    expect(prod.params.a).toBeCloseTo(4, 12)
    expect(prod.params.b).toBeCloseTo(5, 12)
  })
})

describe('tilted moments against quadrature', () => {
  const cavities = [
    [0.3, 1.2],
    [-2, 0.5],
    [4, 3],
  ]
  it.each(cavities)('step, probit, interval, clutter at N(%d, %d)', (m, v) => {
    const cases = [
      [stepTilted(m, v, 0.5), () => 1, [0.5, Infinity]],
      [
        probitTilted(m, v, -1, { offset: 0.2, noiseVariance: 0.7 }),
        (t: number) => normalCdf(-(t - 0.2) / Math.sqrt(0.7)) as number,
      ],
      [intervalTilted(m, v, -1, 1.5), () => 1, [-1, 1.5]],
      [
        clutterTilted(1.5, m, v, { weight: 0.3 }),
        (t: number) => Math.exp(clutterLogLikelihood(t, [1.5], { weight: 0.3 })),
      ],
    ] as const
    for (const [closed, f, bounds] of cases) {
      const q = quadTilted(m, v, f, bounds as [number, number] | undefined)
      expect(closed.logZ).toBeCloseTo(q.logZ, 6)
      expect(closed.mean).toBeCloseTo(q.mean, 6)
      expect(closed.variance).toBeCloseTo(q.variance, 6)
    }
  })
  it('stays finite far in the tails', () => {
    const t = stepTilted(-40, 1)
    expect(t.mean).toBeGreaterThan(0)
    expect(t.variance).toBeGreaterThan(0)
    expect(Number.isFinite(t.logZ)).toBe(true)
  })
  it('broadcasts over tensors', () => {
    const t = probitTilted(tensor([0, 1, 2]), 1)
    expect(t.mean.shape).toEqual([3])
    expect(toFlat(t.mean)[1]).toBeCloseTo(probitTilted(1, 1).mean, 14)
  })
  it('quadrature tilt agrees with the closed form', () => {
    const q = tiltedByQuadrature(0.4, 2, (t) => Math.log(normalCdf(t) as number))
    const c = probitTilted(0.4, 2)
    expect(q.mean).toBeCloseTo(c.mean, 6)
    expect(q.variance).toBeCloseTo(c.variance, 6)
  })
})

describe('EP on the clutter problem against quadrature', () => {
  const problem = { weight: 0.25 }
  const x = Array.from(sampleClutter(stream(7), 20, 2, problem).data)
  // The exact posterior moments by adaptive quadrature, independent of clutterPosterior's grid.
  const logPost = (t: number) =>
    -0.5 * Math.log(2 * Math.PI * 100) - (t * t) / 200 + clutterLogLikelihood(t, x, problem)
  const top = logPost(2)
  const f = (t: number) => Math.exp(logPost(t) - top)
  const Z = panels(f, -40, 40)
  const mean = panels((t) => t * f(t), -40, 40) / Z
  const variance = panels((t) => (t - mean) ** 2 * f(t), -40, 40) / Z
  const logEvidence = top + Math.log(Z)
  it('the grid posterior matches adaptive quadrature', () => {
    const p = clutterPosterior(x, problem)
    expect(p.mean).toBeCloseTo(mean, 8)
    expect(p.variance).toBeCloseTo(variance, 8)
    expect(p.logEvidence).toBeCloseTo(logEvidence, 8)
  })
  it('EP converges close to the exact moments and evidence', () => {
    const s = run(expectationPropagationSteps, clutterEp(x, problem), 2000)
    expect(s.converged).toBe(true)
    expect(Math.abs(s.posterior.mean - mean)).toBeLessThan(0.05 * Math.sqrt(variance))
    expect(Math.abs(s.posterior.variance / variance - 1)).toBeLessThan(0.1)
    expect(Math.abs(epLogEvidence(s) - logEvidence)).toBeLessThan(0.05)
  })
  it('the first EP sweep is ADF', () => {
    const opts = clutterEp(x, problem)
    const ep = run(expectationPropagationSteps, opts, x.length)
    const adf = run(adfSteps, opts, x.length)
    expect(ep.posterior.mean).toBeCloseTo(adf.posterior.mean, 12)
    expect(ep.posterior.variance).toBeCloseTo(adf.posterior.variance, 12)
  })
  it('damped and power EP converge', () => {
    const damped = run(expectationPropagationSteps, clutterEp(x, problem, { damping: 0.5 }), 5000)
    expect(damped.converged).toBe(true)
    const power = run(expectationPropagationSteps, clutterEp(x, problem, { power: 0.5 }), 5000)
    expect(power.converged).toBe(true)
    expect(Math.abs(power.posterior.mean - mean)).toBeLessThan(0.2)
  })
  it('with Gaussian factors EP is exact, evidence included', () => {
    const xs = [0.5, 1.5, -0.2]
    const opts: EpOptions = {
      prior: { mean: 0, variance: 4 },
      factors: 3,
      tilted: (i, c) => {
        const variance = 1 / (1 / c.variance + 1)
        const mean = variance * (c.mean / c.variance + xs[i])
        const s2 = c.variance + 1
        return { logZ: -0.5 * Math.log(2 * Math.PI * s2) - (0.5 * (xs[i] - c.mean) ** 2) / s2, mean, variance }
      },
    }
    const s = run(expectationPropagationSteps, opts, 20)
    const post = 1 / (1 / 4 + 3)
    expect(s.posterior.variance).toBeCloseTo(post, 12)
    expect(s.posterior.mean).toBeCloseTo(post * (0.5 + 1.5 - 0.2), 12)
    // log N(x; 0, I + 4·11ᵀ) by quadrature over θ.
    const ev = integrate(
      (t) =>
        Math.exp(
          -0.5 * Math.log(8 * Math.PI) -
            (t * t) / 8 +
            xs.reduce((a, xi) => a - 0.5 * Math.log(2 * Math.PI) - 0.5 * (xi - t) ** 2, 0),
        ),
      -30,
      30,
      { rtol: 1e-12 },
    ).value
    expect(epLogEvidence(s)).toBeCloseTo(Math.log(ev), 10)
  })
})

describe('TrueSkill', () => {
  it('reproduces the published two-player updates (trueskill.org defaults)', () => {
    const r = { mean: 25, sd: 25 / 3 }
    const win = trueSkillUpdate(r, r, 'win')
    expect(win.player1.mean).toBeCloseTo(29.396, 3)
    expect(win.player1.sd).toBeCloseTo(7.171, 3)
    expect(win.player2.mean).toBeCloseTo(20.604, 3)
    const draw = trueSkillUpdate(r, r, 'draw')
    expect(draw.player1.mean).toBeCloseTo(25, 10)
    expect(draw.player1.sd).toBeCloseTo(6.458, 3)
  })
  it('matches the formulas v = φ/Φ, w = v(v + t) written out', () => {
    const r1 = { mean: 30, sd: 4 }
    const r2 = { mean: 22, sd: 6 }
    const beta = 4
    const eps = drawMargin(0.1, beta)
    const u = trueSkillUpdate(r1, r2, 'loss', { beta, tau: 0, drawMargin: eps })
    const c = Math.sqrt(2 * beta * beta + 16 + 36)
    const t = -(30 - 22) / c - eps / c
    const v = (normalPdf(t) as number) / (normalCdf(t) as number)
    const w = v * (v + t)
    expect(u.player1.mean).toBeCloseTo(30 - (16 * v) / c, 12)
    expect(u.player2.mean).toBeCloseTo(22 + (36 * v) / c, 12)
    expect(u.player1.sd).toBeCloseTo(Math.sqrt(16 * (1 - (16 / (c * c)) * w)), 12)
    expect(u.probability).toBeCloseTo(normalCdf(t) as number, 12)
  })
  it('EP on one match is the one-shot update; on a match set it converges', () => {
    const p = { mean: 25, sd: 25 / 3 }
    const one = run(trueSkillEpSteps, { players: [p, p], matches: [{ winner: 0, loser: 1 }] }, 10)
    const direct = trueSkillUpdate(p, p, 'win', { tau: 0 })
    expect(one.means.data[0]).toBeCloseTo(direct.player1.mean, 10)
    expect(one.sds.data[1]).toBeCloseTo(direct.player2.sd, 10)
    const matches = [
      { winner: 0, loser: 1 },
      { winner: 1, loser: 2 },
      { winner: 0, loser: 2 },
      { winner: 2, loser: 1, draw: true },
    ]
    const s = run(trueSkillEpSteps, { players: [p, p, p], matches }, 1000)
    expect(s.converged).toBe(true)
    expect(s.means.data[0]).toBeGreaterThan(s.means.data[1])
  })
})

describe('Bayes point machine', () => {
  const x = [
    [1, 2, 1],
    [2, 1, 1],
    [1.5, 1.5, 1],
    [-1, -2, 1],
    [-2, -0.5, 1],
    [-1.5, -1, 1],
  ]
  const y = [1, 1, 1, -1, -1, -1]
  it.each(['probit', 'step'] as const)('%s: converges and separates the training data', (likelihood) => {
    const s = run(bayesPointMachineSteps, { x, y, likelihood }, 1000)
    expect(s.converged).toBe(true)
    const p = toFlat(bayesPointMachinePredict(s, x))
    p.forEach((pi, i) => expect(y[i] > 0 ? pi > 0.5 : pi < 0.5).toBe(true))
  })
})

describe('protocol', () => {
  const problem = { weight: 0.25 }
  const x = Array.from(sampleClutter(stream(7), 8, 2, problem).data)
  const p = { mean: 25, sd: 25 / 3 }
  const cases: [string, Algorithm<unknown, unknown>, unknown, (s: never) => number][] = [
    [
      'ep',
      expectationPropagationSteps as never,
      clutterEp(x, problem),
      (s: { posterior: { mean: number } }) => s.posterior.mean,
    ],
    ['adf', adfSteps as never, clutterEp(x, problem), (s: { posterior: { mean: number } }) => s.posterior.mean],
    [
      'trueskill',
      trueSkillEpSteps as never,
      {
        players: [p, p, p],
        matches: [
          { winner: 0, loser: 1 },
          { winner: 1, loser: 2 },
        ],
      },
      (s: TrueSkillEpState) => s.means.data[0],
    ],
    [
      'bpm',
      bayesPointMachineSteps as never,
      {
        x: [
          [1, 1],
          [-1, -0.5],
        ],
        y: [1, -1],
      },
      (s: { mean: Tensor }) => s.mean.data[0],
    ],
  ]
  it.each(cases)('%s: same trace twice; seek equals run; extend equals a longer trace', (_, alg, opts, rec) => {
    const record = { v: rec as (s: unknown) => number }
    const a = trace(alg, opts, 6, { record, stream: stream(1), stopOnNonFinite: false })
    const b = trace(alg, opts, 6, { record, stream: stream(1), stopOnNonFinite: false })
    expect(toFlat(a.series.v)).toEqual(toFlat(b.series.v))
    expect(rec(seek(alg, opts, 3) as never)).toEqual(rec(run(alg, opts, 3) as never))
    const longer = extend(trace(alg, opts, 2, { record, stream: stream(1), stopOnNonFinite: false }), alg, opts, 4)
    expect(toFlat(longer.series.v)).toEqual(toFlat(a.series.v))
  })
})
