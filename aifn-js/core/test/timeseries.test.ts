import { describe, expect, test } from 'vitest'
import { stream } from 'aifn/foundation/random'
import { add, get, mul, sin, stack, toFlat, toRows, type Tensor, type Value, type Vector } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'
import {
  armaAutocorrelation,
  armaAutocovariance,
  armaFitter,
  armaLogLikelihood,
  armaRoots,
  classicalDecomposition,
  difference,
  exponentialSmoothing,
  fitArma,
  forecastArma,
  garchFitter,
  garchLogLikelihood,
  garchProperties,
  isInvertible,
  isStationary,
  psiWeights,
  simulateArma,
  simulateGarch,
  smoothingFitter,
  stateSpaceEm,
  stl,
  undifference,
} from 'aifn-applied/timeseries'
import { burg, yuleWalker } from 'aifn/signal/statistical'
import {
  extendedKalmanFilter,
  kalmanFilter,
  rtsSmoother,
  simulateStateSpace,
  steadyStateKalman,
  unscentedKalmanFilter,
} from 'aifn/inference/filtering'
import { levinsonDurbin } from 'aifn/numerics/linalg'
import { sampleAcf, samplePacf } from 'aifn/probability/stats'
import { fixture } from './fixtures'

type Fx = {
  ar2: number[]
  acf: number[]
  acov: number[]
  pacf: number[]
  yuleWalker: { ar: number[]; sigma2: number }
  burg: { ar: number[]; sigma2: number }
  armaAcov: { spec: { ar: number[]; ma: number[]; sigma: number }; acov: number[] }
  armaExact: { x: number[]; spec: { ar: number[]; ma: number[]; sigma: number }; logLikelihood: number }
  kalman: {
    A: number[][]
    C: number[][]
    Q: number[][]
    R: number[][]
    m0: number[]
    P0: number[][]
    y: number[][]
    filteredMean: number[][]
    filteredCov: number[][][]
    smoothedMean: number[][]
    smoothedCov: number[][][]
    lagOneCov: number[][][]
    logLikelihood: number
  }
  holtWinters: {
    y: number[]
    alpha: number
    beta: number
    gamma: number
    phi: number
    fitted: number[]
    forecast: number[]
  }
  garch: { x: number[]; omega: number; alpha: number; beta: number; logLikelihood: number }
}
const fx = fixture<Fx>('timeseries')

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol = 1e-10) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol * (1 + Math.abs(b[i])))
}
const flat3 = (t: Tensor) => toFlat(t)

describe('autocorrelation and AR estimation', () => {
  test('sample ACF and PACF match numpy and Toeplitz solves', () => {
    const a = sampleAcf(fx.ar2, 12)
    close(toFlat(a.acf), fx.acf)
    expect(a.band).toBeCloseTo(1.959963984540054 / Math.sqrt(fx.ar2.length), 12)
    close(toFlat(samplePacf(fx.ar2, 10)), fx.pacf, 1e-9)
  })
  test('Levinson–Durbin solves the Yule–Walker Toeplitz system', () => {
    const ld = levinsonDurbin(fx.acov, 3)
    close(toFlat(ld.ar), fx.yuleWalker.ar, 1e-10)
    expect(ld.singular).toBe(false)
    const yw = yuleWalker(fx.ar2, 3)
    close(toFlat(yw.ar), fx.yuleWalker.ar, 1e-10)
    expect(yw.sigma2).toBeCloseTo(fx.yuleWalker.sigma2, 10)
  })
  test('Burg matches the reference recursion and recovers the AR(2)', () => {
    const b = burg(fx.ar2, 3)
    close(toFlat(b.ar), fx.burg.ar, 1e-10)
    expect(b.sigma2).toBeCloseTo(fx.burg.sigma2, 10)
    const b2 = toFlat(burg(fx.ar2, 2).ar)
    expect(b2[0]).toBeCloseTo(0.6, 1)
    expect(b2[1]).toBeCloseTo(-0.3, 1)
  })
})

describe('ARMA', () => {
  test('roots, stationarity and invertibility', () => {
    expect(isStationary([0.5])).toBe(true)
    expect(isStationary([1.2])).toBe(false)
    expect(isStationary([0.6, 0.5])).toBe(false) // 1 − 0.6z − 0.5z² has a root inside the unit circle
    expect(isInvertible([0.4])).toBe(true)
    expect(isInvertible([-2])).toBe(false)
    expect(armaRoots({ ar: [0.5] }).ar.minModulus).toBeCloseTo(2, 12)
  })
  test('ψ weights and the theoretical autocovariance', () => {
    close(toFlat(psiWeights({ ar: [0.5], ma: [0.4] }, 4)), [1, 0.9, 0.45, 0.225, 0.1125])
    close(toFlat(armaAutocovariance(fx.armaAcov.spec, 10)), fx.armaAcov.acov, 1e-10)
    // AR(1): ρ(k) = φ^k.
    close(toFlat(armaAutocorrelation({ ar: [0.7] }, 5)), [1, 0.7, 0.49, 0.343, 0.2401, 0.16807], 1e-12)
    expect(() => armaAutocovariance({ ar: [1.1] }, 3)).toThrow(/stationary/)
  })
  test('exact log-likelihood equals the dense Gaussian density', () => {
    const L = armaLogLikelihood(fx.armaExact.x, fx.armaExact.spec)
    expect(L.logLikelihood).toBeCloseTo(fx.armaExact.logLikelihood, 8)
    expect(armaLogLikelihood(fx.armaExact.x, { ar: [1.5] }).logLikelihood).toBe(-Infinity)
  })
  test('simulation is seeded, never clips and reports explosion', () => {
    const a = simulateArma(stream(3), { ar: [0.5] }, 50)
    expect(toFlat(a.x)).toEqual(toFlat(simulateArma(stream(3), { ar: [0.5] }, 50).x))
    expect(a.stationary).toBe(true)
    const big = simulateArma(stream(3), { ar: [1.9] }, 2000)
    expect(big.stationary).toBe(false)
    expect(big.diverged).toBe(true)
    expect(Math.max(...toFlat(big.x).slice(0, big.divergedAt).map(Math.abs))).toBeGreaterThan(1e6)
  })
  test('CSS and exact fits recover an ARMA(1,1)', () => {
    const sim = simulateArma(stream(8), { ar: [0.6], ma: [0.3], sigma: 1.5 }, 2000)
    for (const method of ['css', 'exact'] as const) {
      const fit = fitArma(sim.x, { p: 1, q: 1, method })
      expect(fit.converged).toBe(true)
      expect(toFlat(fit.ar)[0]).toBeCloseTo(0.6, 1)
      expect(toFlat(fit.ma)[0]).toBeCloseTo(0.3, 1)
      expect(Math.sqrt(fit.sigma2)).toBeCloseTo(1.5, 1)
    }
  })
  test('the fitter follows the trace protocol', () => {
    const x = simulateArma(stream(2), { ar: [0.5] }, 200).x
    const alg = armaFitter(x, { p: 1, q: 0, method: 'css' })
    const opts = {}
    expect(seek(alg, opts, 7).objective).toBe(run(alg, opts, 7).objective)
    const rec = { record: { f: (s: { objective: number }) => s.objective } }
    expect(toFlat(extend(trace(alg, opts, 5, rec), alg, opts, 5, rec).series.f)).toEqual(
      toFlat(trace(alg, opts, 10, rec).series.f),
    )
  })
  test('forecasts revert to the mean with widening intervals', () => {
    const f = forecastArma([0, 0, 0, 4], { ar: [0.5], mean: 0, sigma: 1 }, 3)
    close(toFlat(f.mean), [2, 1, 0.5])
    close(toFlat(f.se), [1, Math.sqrt(1.25), Math.sqrt(1.3125)])
  })
})

describe('transformations', () => {
  test('differencing and its inverse', () => {
    const x = [1, 4, 9, 16, 25, 36]
    close(toFlat(difference(x, { order: 2 })), [2, 2, 2, 2])
    close(toFlat(undifference(difference(x, { lag: 2 }), [1, 4], { lag: 2 })), x)
  })
  test('decompositions recover a clean seasonal pattern', () => {
    const pattern = [3, -1, -4, 2]
    const y = Array.from({ length: 48 }, (_, t) => 10 + 0.3 * t + pattern[t % 4])
    close(toFlat(classicalDecomposition(y, 4).pattern), pattern, 1e-9)
    const s = stl(y, 4)
    close(toFlat(s.pattern), pattern, 0.05)
    expect(Math.max(...toFlat(s.remainder).map(Math.abs))).toBeLessThan(0.1)
  })
})

describe('state space', () => {
  const k = fx.kalman
  const model = { A: k.A, C: k.C, Q: k.Q, R: k.R, m0: k.m0, P0: k.P0 }
  test('the Kalman filter equals exact Gaussian conditioning', () => {
    const f = kalmanFilter(model, k.y)
    close(toFlat(f.mean), k.filteredMean.flat(), 1e-9)
    close(flat3(f.cov), k.filteredCov.flat(2), 1e-9)
    expect(f.logLikelihood).toBeCloseTo(k.logLikelihood, 9)
  })
  test('the RTS smoother equals exact Gaussian conditioning, lag-one covariances included', () => {
    const s = rtsSmoother(model, k.y)
    close(toFlat(s.mean), k.smoothedMean.flat(), 1e-9)
    close(flat3(s.cov), k.smoothedCov.flat(2), 1e-9)
    close(toFlat(s.lagOneCov).slice(4), k.lagOneCov.flat(2), 1e-9)
  })
  test('local level model: the steady-state gain has its closed form', () => {
    // P = P + q − P²/(P + r) ⇒ P² − qP − qr = 0 for the predicted variance; gain K = P/(P + r).
    const q = 0.5
    const r = 2
    const P = (q + Math.sqrt(q * q + 4 * q * r)) / 2
    const ss = steadyStateKalman({ A: 1, C: 1, Q: q, R: r, m0: 0, P0: 1 })
    expect(ss.converged).toBe(true)
    expect(toFlat(ss.gain)[0]).toBeCloseTo(P / (P + r), 10)
  })
  test('singular noise (a zero diagonal) filters and simulates without NaN', () => {
    const m = {
      A: [
        [1, 1],
        [0, 1],
      ],
      C: [[1, 0]],
      Q: [
        [0, 0],
        [0, 0.1],
      ],
      R: [[0.2]],
      m0: [0, 0],
      P0: [
        [0, 0],
        [0, 1],
      ],
    }
    const sim = simulateStateSpace(stream(5), m, 30)
    expect(toFlat(sim.states).every(Number.isFinite)).toBe(true)
    const f = kalmanFilter(m, sim.observations)
    expect(toFlat(f.mean).every(Number.isFinite)).toBe(true)
    expect(Number.isFinite(f.logLikelihood)).toBe(true)
    expect(toFlat(rtsSmoother(m, sim.observations).mean).every(Number.isFinite)).toBe(true)
  })
  test('missing observations are predicted through', () => {
    const y = k.y.map((r, t) => (t === 4 ? [NaN] : r))
    const f = kalmanFilter(model, y)
    const rows = toRows(f.mean)
    close(rows[4], toRows(f.predictedMean)[4])
  })
  test('EM never decreases the log-likelihood and follows the trace protocol', () => {
    const truth = { A: [[0.9]], C: [[1]], Q: [[0.3]], R: [[0.5]], m0: [0], P0: [[1]] }
    const y = simulateStateSpace(stream(4), truth, 300).observations
    const alg = stateSpaceEm(y, { A: 0.5, C: 1, Q: 1, R: 1, m0: 0, P0: 1 }, { estimate: { C: false } })
    const t = trace(alg, {}, 60, { record: { ll: (s) => s.logLikelihood } })
    const ll = toFlat(t.series.ll)
    for (let i = 1; i < ll.length; i++) expect(ll[i]).toBeGreaterThanOrEqual(ll[i - 1] - 1e-8)
    expect(toFlat(t.steps.at(-1)!.model.A)[0]).toBeCloseTo(0.9, 1)
    expect(seek(alg, {}, 5).logLikelihood).toBe(run(alg, {}, 5).logLikelihood)
  })
  test('EKF and UKF reduce to the Kalman filter for a linear model', () => {
    const lin = {
      f: (z: Vector): Value => stack([add(get(z, 0), get(z, 1)), get(z, 1)]),
      h: (z: Vector): Value => stack([get(z, 0)]),
      Q: k.Q,
      R: k.R,
      m0: k.m0,
      P0: k.P0,
    }
    const ref = kalmanFilter(model, k.y)
    close(toFlat(extendedKalmanFilter(lin, k.y).mean), toFlat(ref.mean), 1e-9)
    close(toFlat(unscentedKalmanFilter(lin, k.y).mean), toFlat(ref.mean), 1e-9)
    // A nonlinear pendulum runs and stays finite.
    const pend = {
      f: (z: Vector): Value => stack([add(get(z, 0), mul(0.1, get(z, 1))), add(get(z, 1), mul(-0.1, sin(get(z, 0))))]),
      h: (z: Vector): Value => stack([sin(get(z, 0))]),
      Q: [
        [1e-4, 0],
        [0, 1e-4],
      ],
      R: [[0.05]],
      m0: [1.2, 0],
      P0: [
        [0.1, 0],
        [0, 0.1],
      ],
    }
    const ys = Array.from({ length: 40 }, (_, t) => [Math.sin(1.2 * Math.cos(0.1 * t))])
    expect(toFlat(extendedKalmanFilter(pend, ys).mean).every(Number.isFinite)).toBe(true)
    expect(toFlat(unscentedKalmanFilter(pend, ys).mean).every(Number.isFinite)).toBe(true)
  })
})

describe('exponential smoothing', () => {
  const h = fx.holtWinters
  test('additive damped Holt–Winters matches the reference recursion', () => {
    const r = exponentialSmoothing(
      h.y,
      { trend: 'damped', seasonal: 'additive', period: 4, alpha: h.alpha, beta: h.beta, gamma: h.gamma, phi: h.phi },
      { horizon: 8 },
    )
    close(toFlat(r.fitted), h.fitted, 1e-10)
    close(toFlat(r.forecast), h.forecast, 1e-10)
    const lo = toFlat(r.lower)
    const hi = toFlat(r.upper)
    for (let i = 1; i < 8; i++) expect(hi[i] - lo[i]).toBeGreaterThanOrEqual(hi[i - 1] - lo[i - 1])
  })
  test('simple smoothing: forecasts equal the last level', () => {
    const r = exponentialSmoothing([1, 2, 3, 2, 1], { alpha: 0.5 }, { horizon: 2 })
    const lv = toFlat(r.level)
    close(toFlat(r.forecast), [lv[4], lv[4]])
  })
  test('the fitter lowers the SSE', () => {
    const alg = smoothingFitter(h.y, { trend: 'additive', seasonal: 'additive', period: 4 })
    const s0 = alg.init({})
    const s = run(alg, {}, 500)
    expect(s.objective).toBeLessThan(s0.objective)
  })
})

describe('GARCH', () => {
  test('log-likelihood matches the reference recursion', () => {
    const g = fx.garch
    expect(garchLogLikelihood(g.x, g).logLikelihood).toBeCloseTo(g.logLikelihood, 9)
  })
  test('simulated returns have the unconditional variance and heavy tails', () => {
    const spec = { omega: 0.05, alpha: 0.08, beta: 0.9 }
    const p = garchProperties(spec)
    const r = toFlat(simulateGarch(stream(1), spec, 40000).returns)
    const m2 = r.reduce((a, v) => a + v * v, 0) / r.length
    const m4 = r.reduce((a, v) => a + v ** 4, 0) / r.length
    expect(m2).toBeCloseTo(p.unconditionalVariance, 0)
    expect(Math.abs(m2 / p.unconditionalVariance - 1)).toBeLessThan(0.15)
    expect(m4 / (m2 * m2)).toBeGreaterThan(3.3)
  })
  test('the fitter recovers persistence', () => {
    const spec = { omega: 0.05, alpha: 0.1, beta: 0.85 }
    const r = simulateGarch(stream(9), spec, 4000).returns
    const s = run(garchFitter(r), {}, 3000)
    expect(s.params.alpha + s.params.beta).toBeCloseTo(0.95, 1)
  })
})
