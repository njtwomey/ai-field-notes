import { gaussianTarget } from 'aifn/mcmc'
import { normal, normals, stream } from 'aifn/random'
import { toFlat, type Tensor } from 'aifn/tensor'
import { extend, run, seek, trace } from 'aifn/trace'
import {
  bbvi,
  caviGaussianMixture,
  caviNormalGamma,
  elbo,
  elboGradient,
  fullRankGaussian,
  gradientVariance,
  meanFieldGaussian,
  mixturePredictiveDensity,
  normalGammaPosterior,
  type GaussianFamily,
  type NormalGammaPrior,
} from 'aifn/vi'
import { describe, expect, it } from 'vitest'
import { fixture } from './fixtures'

const m = [1, -0.5]
const S = [
  [1, 0.6],
  [0.6, 0.8],
]
const target = gaussianTarget(m, S)
const P = toFlat(target.precision)

/** The exact ELBO of q = N(μ, C) against the normalised Gaussian target: E_q[log p] + H[q]. */
function exactElbo(family: GaussianFamily, lambda: number[]): number {
  const mu = toFlat(family.mean(lambda))
  const C = toFlat(family.covariance(lambda))
  let tr = 0
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) tr += P[i * 2 + j] * (C[j * 2 + i] + (mu[j] - m[j]) * (mu[i] - m[i]))
  const logNorm = target.logDensity(target.mean) as number
  return logNorm - 0.5 * tr + family.entropy(lambda)
}

function numericGrad(f: (l: number[]) => number, l: number[]) {
  return l.map((_, i) => {
    const up = [...l]
    const dn = [...l]
    up[i] += 1e-6
    dn[i] -= 1e-6
    return (f(up) - f(dn)) / 2e-6
  })
}

describe('families', () => {
  it.each([
    ['mean-field', meanFieldGaussian(2), [0.3, -0.2, 0.1, -0.4]],
    ['full-rank', fullRankGaussian(2), [0.3, -0.2, 0.1, 0.5, -0.4]],
  ] as [string, GaussianFamily, number[]][])('%s: score and path kernels match finite differences', (_, fam, l) => {
    const L = Float64Array.from(l)
    const eps = Float64Array.of(0.7, -1.1)
    const x = fam.kernels.transform(L, eps)
    const score = fam.kernels.score(L, eps)
    numericGrad((ll) => fam.logDensity(ll, x), l).forEach((g, i) => expect(score[i]).toBeCloseTo(g, 5))
    // Path: f(x) = a·x, ∇f = a.
    const a = Float64Array.of(0.4, -1.3)
    const path = fam.kernels.pathGrad(L, eps, a)
    const f = (ll: number[]) => {
      const y = fam.kernels.transform(Float64Array.from(ll), eps)
      return a[0] * y[0] + a[1] * y[1]
    }
    numericGrad(f, l).forEach((g, i) => expect(path[i]).toBeCloseTo(g, 5))
    const hg = fam.kernels.entropyGrad(L)
    numericGrad((ll) => fam.entropy(ll), l).forEach((g, i) => expect(hg[i]).toBeCloseTo(g, 5))
    // log q integrates to the right entropy (Monte Carlo).
    expect(elbo({ dim: 2, logDensity: () => 0 }, fam, l, stream(1), { samples: 20000 }).value).toBeCloseTo(
      fam.entropy(l),
      1,
    )
  })
})

describe('ELBO estimators', () => {
  const fam = meanFieldGaussian(2)
  const l = [0.2, 0.1, -0.3, 0.2]
  const truth = numericGrad((ll) => exactElbo(fam, ll), l)

  it('elboGradient returns a gradient per parameter, deterministic in its stream', () => {
    const a = elboGradient(target, fam, l, stream(9), { samples: 3 })
    const b = elboGradient(target, fam, l, stream(9), { samples: 3 })
    expect(a.grad.shape).toEqual([4])
    expect(a.draws.shape).toEqual([3, 2])
    expect(toFlat(a.grad)).toEqual(toFlat(b.grad))
    expect(() =>
      elboGradient(target, fam, l, stream(1), { estimator: 'score', samples: 2, baseline: 'control-variate' }),
    ).toThrow()
  })

  it('elbo estimate matches the exact value', () => {
    const e = elbo(target, fam, l, stream(1), { samples: 20000 })
    expect(Math.abs(e.value - exactElbo(fam, l))).toBeLessThan(4 * e.standardError + 1e-3)
  })

  it.each([
    ['reparameterisation', 'none', 1],
    ['score', 'none', 10],
    ['score', 'leave-one-out', 10],
    ['score', 'control-variate', 10],
  ] as const)('%s (%s) is unbiased', (estimator, baseline, samples) => {
    const v = gradientVariance(target, fam, l, stream(2), { estimator, baseline, samples, repeats: 3000 })
    const mean = toFlat(v.mean)
    const se = toFlat(v.variance).map((x) => Math.sqrt(x / v.repeats))
    truth.forEach((g, i) => expect(Math.abs(mean[i] - g)).toBeLessThan(5 * se[i] + 0.02))
  })

  it('reparameterisation has lower variance than the score function; baselines help', () => {
    const tv = (opts: Parameters<typeof gradientVariance>[4]) =>
      gradientVariance(target, fam, l, stream(3), { ...opts, samples: 10, repeats: 300 }).totalVariance
    const reparam = tv({ estimator: 'reparameterisation' })
    const plain = tv({ estimator: 'score', baseline: 'none' })
    const loo = tv({ estimator: 'score', baseline: 'leave-one-out' })
    expect(reparam).toBeLessThan(loo)
    expect(loo).toBeLessThan(plain)
  })

  it('full-rank gradients are unbiased', () => {
    const fr = fullRankGaussian(2)
    const lf = [0.2, 0.1, -0.3, 0.4, 0.2]
    const truthF = numericGrad((ll) => exactElbo(fr, ll), lf)
    const v = gradientVariance(target, fr, lf, stream(4), { samples: 4, repeats: 2000 })
    const mean = toFlat(v.mean)
    truthF.forEach((g, i) => expect(mean[i]).toBeCloseTo(g, 1))
  })
})

describe('bbvi', () => {
  it('mean field finds the mean and the conditional precisions (under-covers)', () => {
    const s = run(bbvi(target, { lr: (t) => 0.05 / (1 + t / 100), samples: 4 }), {}, 5000, { stream: stream(1) })
    const mu = toFlat(s.mean)
    const C = toFlat(s.covariance)
    expect(mu[0]).toBeCloseTo(1, 1)
    expect(mu[1]).toBeCloseTo(-0.5, 1)
    // KL(q‖p) optimum: σᵢ² = 1/Λᵢᵢ, below the marginal variance Σᵢᵢ.
    expect(C[0]).toBeCloseTo(1 / P[0], 1)
    expect(C[3]).toBeCloseTo(1 / P[3], 1)
    expect(C[0]).toBeLessThan(S[0][0])
    expect(s.objective).toBe('KL(q‖p)')
  })

  it('full rank recovers the covariance', () => {
    const s = run(bbvi(target, { family: 'full-rank', lr: (t) => 0.05 / (1 + t / 100), samples: 8 }), {}, 5000, {
      stream: stream(2),
    })
    const C = toFlat(s.covariance)
    expect(C[1]).toBeCloseTo(0.6, 1)
    expect(C[3]).toBeCloseTo(0.8, 1)
  })

  it('score estimator also converges', () => {
    const s = run(bbvi(target, { estimator: 'score', samples: 20, lr: 0.03 }), {}, 1500, { stream: stream(3) })
    expect(toFlat(s.mean)[0]).toBeCloseTo(1, 0)
  })

  it('protocol: same stream, seek, extend', () => {
    const alg = bbvi(target, { lr: 0.05 })
    const record = { elbo: (s: { elbo: number }) => s.elbo }
    const a = trace(alg, {}, 20, { record, stream: stream(5) })
    const b = trace(alg, {}, 20, { record, stream: stream(5) })
    expect(toFlat(a.series.elbo)).toEqual(toFlat(b.series.elbo))
    expect(toFlat(seek(alg, {}, 13, { stream: stream(5) }).lambda)).toEqual(toFlat(a.steps[13].lambda))
    const longer = extend(trace(alg, {}, 8, { record, stream: stream(5) }), alg, {}, 12)
    expect(toFlat(longer.series.elbo)).toEqual(toFlat(a.series.elbo))
  })
})

describe('CAVI', () => {
  const c = fixture<{ normal_gamma: { x: number[]; prior: NormalGammaPrior; log_evidence: number } }>('vi').normal_gamma

  it('normal-gamma exact posterior evidence matches quadrature', () => {
    expect(normalGammaPosterior(c.x, c.prior).logEvidence).toBeCloseTo(c.log_evidence, 7)
  })

  it('CAVI normal-gamma: ELBO rises, stays below log p(x), and the means match the truth', () => {
    const tr = trace(caviNormalGamma(c.x, c.prior), { expectedTau0: 0.1 }, 200, {
      record: { elbo: (s) => s.elbo },
    })
    const e = toFlat(tr.series.elbo)
    for (let i = 1; i < e.length; i++) expect(e[i]).toBeGreaterThanOrEqual(e[i - 1] - 1e-12)
    const last = tr.steps.at(-1)!
    expect(tr.meta.stopped).toBe('done')
    expect(last.kl).toBeGreaterThan(0)
    expect(last.kl).toBeLessThan(0.1)
    const exact = normalGammaPosterior(c.x, c.prior)
    expect(last.muMean).toBeCloseTo(exact.meanOfMu, 10)
    expect(1 / last.muPrecision).toBeLessThan(exact.varianceOfMu)
    expect(last.expectedTau).toBeCloseTo(exact.meanOfTau, 1)
  })

  it('CAVI Gaussian mixture: monotone ELBO, recovers well-separated components', () => {
    const s0 = stream('gmm')
    const x: number[][] = []
    const centres = [
      [-3, 0],
      [3, 1],
      [0, 4],
    ]
    for (let n = 0; n < 300; n++) {
      const c2 = centres[n % 3]
      const e = toFlat(normals(s0.child(n), 2))
      x.push([c2[0] + 0.6 * e[0], c2[1] + 0.6 * e[1]])
    }
    const alg = caviGaussianMixture(x, 6, { alpha0: 1e-3 })
    const tr = trace(alg, {}, 300, { stream: stream(1), record: { elbo: (s) => s.elbo } })
    const e = toFlat(tr.series.elbo)
    for (let i = 1; i < e.length; i++) expect(e[i]).toBeGreaterThanOrEqual(e[i - 1] - 1e-8 * Math.abs(e[i]))
    const last = tr.steps.at(-1)!
    const w = toFlat(last.weights)
    const used = w.filter((v) => v > 0.05).length
    expect(used).toBe(3)
    const means = toFlat(last.means)
    for (const c2 of centres) {
      const hit = w.some((v, k) => v > 0.05 && Math.hypot(means[2 * k] - c2[0], means[2 * k + 1] - c2[1]) < 0.3)
      expect(hit).toBe(true)
    }
    // The predictive density integrates to about 1 (1-D check below) and is positive.
    const dens = toFlat(mixturePredictiveDensity(last, [[-3, 0]]).density)
    expect(dens[0]).toBeGreaterThan(0)
  })

  it('1-D mixture predictive density integrates to 1', () => {
    const xs = Array.from({ length: 200 }, (_, n) => normal(stream('x').child(n), n % 2 ? 2 : -2, 0.7))
    const s = run(caviGaussianMixture(xs, 2), {}, 200, { stream: stream(1) })
    const grid = Array.from({ length: 2001 }, (_, i) => -10 + i * 0.01)
    const d = toFlat(mixturePredictiveDensity(s, grid).density as Tensor)
    expect(d.reduce((a, b) => a + b, 0) * 0.01).toBeCloseTo(1, 3)
  })
})
