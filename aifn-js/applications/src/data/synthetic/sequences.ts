/**
 * Seeded sequences: draws from a discrete hidden Markov model (and the occasionally dishonest casino), and synthetic
 * time series (autoregressive, seasonal with trend, random walk).
 */

import { normal, type Stream } from 'aifn/foundation/random'
import { fromData, type Tensor } from 'aifn/foundation/tensor'
import { hmm, sampleHmm } from 'aifn-applied/inference/sequence-models'
import { checkCount, labels, matrix, vector, type DatasetMeta } from '../types'

/** A discrete hidden Markov model: initial distribution (k), transition matrix (k × k) and emission matrix (k × m). */
export interface DiscreteHmm {
  initial: Tensor
  transition: Tensor
  emission: Tensor
}

/** A draw from a hidden Markov model: observations `x` and hidden states `z` (int32, length n). */
export interface HmmSample {
  x: Tensor
  z: Tensor
  model: DiscreteHmm
  meta: DatasetMeta
}

function rows(m: readonly (readonly number[])[], what: string): number[][] {
  return m.map((r, i) => {
    const total = r.reduce((a, b) => a + b, 0)
    if (Math.abs(total - 1) > 1e-9) throw new RangeError(`${what}: row ${i} sums to ${total}, not 1`)
    return [...r]
  })
}

/**
 * n steps of a discrete hidden Markov model: z₁ ~ initial, z_t | z_{t−1} ~ transition[z_{t−1}], x_t | z_t ~
 * emission[z_t]. Observation symbols are 0, …, m − 1. Drawn by `aifn-applied/inference/sequence-models`' `sampleHmm`
 * (step t from `child(s, 'step', t)`).
 */
export function hmmSample(
  s: Stream,
  model: {
    initial: readonly number[]
    transition: readonly (readonly number[])[]
    emission: readonly (readonly number[])[]
  },
  n: number,
): HmmSample {
  checkCount(n, 'hmmSample')
  const [initial] = rows([model.initial], 'hmmSample initial')
  const A = rows(model.transition, 'hmmSample transition')
  const B = rows(model.emission, 'hmmSample emission')
  const k = initial.length
  const m = B[0].length
  const drawn = sampleHmm(s, hmm(initial, A, B), n)
  return {
    x: drawn.observations,
    z: drawn.states,
    model: {
      initial: vector(initial),
      transition: matrix(Float64Array.from(A.flat()), k, k),
      emission: matrix(Float64Array.from(B.flat()), k, m),
    },
    meta: {
      name: 'hidden Markov model',
      description: `${n} steps of a ${k}-state hidden Markov model with ${m} symbols.`,
      task: 'sequence',
      featureNames: ['symbol'],
      labelNames: Array.from({ length: k }, (_, j) => `state ${j}`),
      key: s.key,
    },
  }
}

/** Options for `casino`. */
export interface CasinoOptions {
  /** Rolls. Default 300. */
  n?: number
  /** P(fair → loaded) per roll. Default 0.05. */
  toLoaded?: number
  /** P(loaded → fair) per roll. Default 0.1. */
  toFair?: number
  /** P(six | loaded); the other five faces share the rest equally. Default 0.5. */
  loadedSix?: number
}

/**
 * The occasionally dishonest casino (Durbin, Eddy, Krogh and Mitchison, 1998, "Biological Sequence Analysis", §3.2): a
 * fair die (state 0) and a loaded die (state 1) that shows six half the time. The chain starts from its stationary
 * distribution. Observations `x` are faces 1–6 (symbols 0–5 plus one); `model.emission` is indexed by face − 1.
 */
export function casino(s: Stream, options: CasinoOptions = {}): HmmSample {
  const { n = 300, toLoaded = 0.05, toFair = 0.1, loadedSix = 0.5 } = options
  const loaded = toLoaded / (toLoaded + toFair)
  const other = (1 - loadedSix) / 5
  const sample = hmmSample(
    s,
    {
      initial: [1 - loaded, loaded],
      transition: [
        [1 - toLoaded, toLoaded],
        [toFair, 1 - toFair],
      ],
      emission: [Array<number>(6).fill(1 / 6), [other, other, other, other, other, loadedSix]],
    },
    n,
  )
  const faces = Int32Array.from(sample.x.data, (v) => v + 1)
  return {
    ...sample,
    x: labels(faces),
    meta: {
      name: 'occasionally dishonest casino',
      description: `${n} rolls of a casino die that switches from fair to loaded with probability ${toLoaded} and back with probability ${toFair}; the loaded die shows six with probability ${loadedSix}.`,
      task: 'sequence',
      featureNames: ['face'],
      labelNames: ['fair', 'loaded'],
      source: 'Durbin, Eddy, Krogh and Mitchison (1998), Biological Sequence Analysis, §3.2',
      key: s.key,
    },
  }
}

/** A time series: times `t` and values `y` (length n), with a flag for divergence. */
export interface TimeSeries {
  t: Tensor
  y: Tensor
  /** True when the series overflowed to a non-finite value (e.g. a non-stationary AR model); values are kept as drawn. */
  diverged: boolean
  meta: DatasetMeta
}

function series(y: Float64Array, meta: DatasetMeta): TimeSeries {
  return {
    t: fromData(Float64Array.from({ length: y.length }, (_, i) => i)),
    y: vector(y),
    diverged: y.some((v) => !Number.isFinite(v)),
    meta,
  }
}

/** Options for `arSeries`. */
export interface ArOptions {
  /** AR coefficients a₁, …, a_p in x_t = c + Σ a_k x_{t−k} + e_t. */
  coefficients: readonly number[]
  n?: number
  /** Standard deviation of the driving noise e_t. Default 1. */
  sd?: number
  /** Constant c. Default 0. */
  constant?: number
  /** Initial steps drawn and discarded so that the series starts near stationarity. Default 500. */
  burn?: number
}

/**
 * An autoregressive AR(p) series x_t = c + Σ_k a_k x_{t−k} + e_t, e_t ~ N(0, sd²), started from zeros with a burn-in.
 * No clipping: an explosive model sets `diverged`.
 */
export function arSeries(s: Stream, options: ArOptions): TimeSeries {
  const { coefficients: a, n = 200, sd = 1, constant = 0, burn = 500 } = options
  checkCount(n, 'arSeries')
  const total = n + burn
  const x = new Float64Array(total)
  for (let t = 0; t < total; t++) {
    let v = constant + sd * normal(s)
    for (let k = 0; k < a.length; k++) if (t - k - 1 >= 0) v += a[k] * x[t - k - 1]
    x[t] = v
  }
  return series(x.slice(burn), {
    name: `AR(${a.length})`,
    description: `${n} steps of an AR(${a.length}) process with coefficients (${a.join(', ')}) and noise sd ${sd}.`,
    task: 'sequence',
    featureNames: ['x'],
    key: s.key,
  })
}

/** Options for `seasonalSeries`. */
export interface SeasonalOptions {
  n?: number
  /** Period of the seasonal cycle in steps. Default 12. */
  period?: number
  /** Amplitude of each harmonic of the season, first harmonic first. Default [1]. */
  amplitudes?: readonly number[]
  /** Level at t = 0 and slope per step. Defaults 0 and 0.02. */
  level?: number
  trend?: number
  /** Standard deviation of white observation noise. Default 0.3. */
  noise?: number
  /** Coefficient of AR(1) noise instead of white noise (0 for white). Default 0. */
  persistence?: number
}

/**
 * A trend plus a seasonal cycle plus noise: y_t = level + trend·t + Σ_h A_h sin(2π h t / period) + u_t, where u_t is
 * white or AR(1) noise with marginal standard deviation `noise`.
 */
export function seasonalSeries(s: Stream, options: SeasonalOptions = {}): TimeSeries {
  const { n = 120, period = 12, amplitudes = [1], level = 0, trend = 0.02, noise = 0.3, persistence = 0 } = options
  checkCount(n, 'seasonalSeries')
  const y = new Float64Array(n)
  const innovation = noise * Math.sqrt(1 - persistence * persistence)
  let u = noise * normal(s)
  for (let t = 0; t < n; t++) {
    if (t > 0) u = persistence * u + innovation * normal(s)
    let season = 0
    amplitudes.forEach((amp, h) => (season += amp * Math.sin((2 * Math.PI * (h + 1) * t) / period)))
    y[t] = level + trend * t + season + u
  }
  return series(y, {
    name: 'seasonal series',
    description: `${n} steps of a trend (slope ${trend}) plus a season of period ${period} plus noise of sd ${noise}.`,
    task: 'sequence',
    featureNames: ['y'],
    key: s.key,
  })
}

/** A Gaussian random walk y_t = y_{t−1} + drift + sd·z_t from y_0 = start. */
export function randomWalk(
  s: Stream,
  options: { n?: number; sd?: number; drift?: number; start?: number } = {},
): TimeSeries {
  const { n = 200, sd = 1, drift = 0, start = 0 } = options
  checkCount(n, 'randomWalk')
  const y = new Float64Array(n)
  let v = start
  for (let t = 0; t < n; t++) {
    if (t > 0) v += drift + sd * normal(s)
    y[t] = v
  }
  return series(y, {
    name: 'random walk',
    description: `${n} steps of a Gaussian random walk with step sd ${sd} and drift ${drift}.`,
    task: 'sequence',
    featureNames: ['y'],
    key: s.key,
  })
}
