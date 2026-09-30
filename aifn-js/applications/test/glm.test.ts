import { describe, expect, it } from 'vitest'
import { deviance, irls } from 'aifn-applied/learning/generalised'
import { family, link, poisson, type FamilyName, type LinkName } from 'aifn/probability/likelihoods'
import { glm, multinomialLogisticRegression, negativeBinomialRegression } from 'aifn-applied/learning/generalised/glm'
import { stream } from 'aifn/foundation/random'
import { fromData, tensor, toFlat, toRows } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'
import { gradCheck } from 'aifn/foundation/autodiff'
import { fixture } from './fixtures'

type Fit = {
  family: FamilyName
  link: LinkName
  theta: number
  y: number[]
  weights: number[] | null
  offset: number[] | null
  coef: number[]
  se: number[]
  p: number[]
  dispersion: number
  deviance: number
  pearson: number[]
  deviance_residuals: number[]
}
type Fixture = {
  x: number[][]
  fits: Record<string, Fit>
  logistic: { y: number[]; coef: number[] }
  nb_ml: { coef: number[]; theta: number }
  multinomial: { y: number[]; coef: number[][]; se: number[][] }
}
const F = fixture<Fixture>('glm')
const x = tensor(F.x)

function close(actual: number[], expected: number[], tol: number) {
  expect(actual.length).toBe(expected.length)
  actual.forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThan(tol * (1 + Math.abs(expected[i]))))
}

describe('links', () => {
  it('inverse undoes link and derivative is dμ/dη', () => {
    for (const name of [
      'identity',
      'log',
      'logit',
      'probit',
      'cloglog',
      'inverse',
      'inverse-squared',
      'sqrt',
    ] as const) {
      const l = link(name)
      const eta = name === 'inverse' || name === 'inverse-squared' || name === 'sqrt' ? 0.7 : -0.4
      expect(l.link(l.inverse(eta) as number) as number).toBeCloseTo(eta, 10)
      const report = gradCheck((e: number) => l.inverse(e), eta)
      expect(report.ok).toBe(true)
      const h = 1e-6
      const fd = ((l.inverse(eta + h) as number) - (l.inverse(eta - h) as number)) / (2 * h)
      expect(l.derivative(eta) as number).toBeCloseTo(fd, 6)
    }
  })
})

describe('glm', () => {
  for (const [key, f] of Object.entries(F.fits)) {
    it(`${key} (${f.family}, ${f.link}) matches the maximum-likelihood reference`, () => {
      const fam = family(f.family, { theta: f.theta })
      const model = glm({ family: fam, link: f.link, tol: 1e-12, maxIterations: 100 }).fit({
        x,
        y: tensor(f.y),
        weights: f.weights ? tensor(f.weights) : undefined,
        offset: f.offset ? tensor(f.offset) : undefined,
      })
      expect(model.converged).toBe(true)
      close(toFlat(model.coefficients), f.coef, 1e-6)
      close(toFlat(model.standardErrors), f.se, 1e-5)
      close(toFlat(model.pValues), f.p, 1e-4)
      expect(model.dispersion).toBeCloseTo(f.dispersion, 6)
      expect(model.deviance).toBeCloseTo(f.deviance, 6)
      close(toFlat(model.residuals('pearson')), f.pearson, 1e-5)
      close(toFlat(model.residuals('deviance')), f.deviance_residuals, 1e-5)
      expect(model.nullDeviance).toBeGreaterThanOrEqual(model.deviance - 1e-9)
    })
  }

  it('the Bernoulli GLM agrees with scikit-learn logistic regression', () => {
    const m = glm({ family: family('binomial'), tol: 1e-12 }).fit({ x, y: tensor(F.logistic.y) })
    close(toFlat(m.coefficients), F.logistic.coef, 1e-5)
  })

  it('predicts, samples and keeps the IRLS trace', () => {
    const f = F.fits.poisson
    const model = glm({ family: poisson() }).fit({ x, y: tensor(f.y), offset: tensor(f.offset!) })
    const draws = model.sample(stream('glm'), x, 3)
    expect(draws.shape).toEqual([3, F.x.length])
    expect(model.training.series.deviance.shape[0]).toBe(model.training.steps.length)
    expect(toFlat(model.expect(x))[0]).toBeCloseTo(Math.exp(toFlat(model.forward(x))[0]), 6)
  })

  it('satisfies the trace protocol', () => {
    const f = F.fits.gamma
    const n = F.x.length
    const design = fromData(Float64Array.from(F.x.flatMap((r) => [...r, 1])), [n, 3])
    const alg = irls({ design, y: tensor(f.y), family: family('gamma'), link: link('log') })
    const long = trace(alg, {}, 5)
    for (const i of [0, 2, 4]) expect(seek(alg, {}, i).deviance).toBe(run(alg, {}, i).deviance)
    expect(extend(trace(alg, {}, 2), alg, {}, 3).steps.at(-1)!.deviance).toBe(long.steps.at(-1)!.deviance)
    expect(deviance(family('gamma'), tensor(f.y), long.steps.at(-1)!.mu)).toBeCloseTo(long.steps.at(-1)!.deviance, 10)
  })
})

describe('negative binomial with θ estimated', () => {
  it('reaches the joint maximum-likelihood estimate', () => {
    const m = negativeBinomialRegression().fit({ x, y: tensor(F.fits['negative-binomial'].y) })
    expect(m.family.params.theta).toBeCloseTo(F.nb_ml.theta, 3)
    close(toFlat(m.coefficients), F.nb_ml.coef, 1e-4)
  })
})

describe('multinomial logistic regression', () => {
  it('contrasts against class 0 match the maximum-likelihood reference with standard errors', () => {
    const m = multinomialLogisticRegression({ tol: 1e-14 }).fit({ x, y: tensor(F.multinomial.y) })
    close(toRows(m.contrasts.coefficients).flat(), F.multinomial.coef.flat(), 1e-5)
    close(toRows(m.contrasts.standardErrors).flat(), F.multinomial.se.flat(), 1e-4)
  })
})
