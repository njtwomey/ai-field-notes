/**
 * Samplers: from noise to data by reversing the forward process with a noise predictor (see `predictor.ts`). Each is
 * a traceable `Algorithm` whose state holds every particle, so a trace of `x` gives the sampling paths.
 *
 * - `ddpmSampler`: ancestral sampling of DDPM (Ho et al., 2020, Algorithm 2), one step per noise level.
 * - `ddimSampler`: DDIM (Song, Meng & Ermon, 2021, eq. 12) on a subsequence of levels; η = 0 is deterministic.
 * - `reverseSdeSampler`: Euler–Maruyama on the reverse-time SDE dx = [f x − g²∇log p_t] dt + g dw̄ (Anderson, 1982;
 *   Song et al., 2021, eq. 6).
 * - `probabilityFlowSampler`: the probability-flow ODE dx/dt = f x − ½g²∇log p_t (Song et al., 2021, eq. 13),
 *   integrated with classical fourth-order Runge–Kutta (or Euler), deterministic after the initial draw.
 *
 * Step k draws from `stream.child('step', k)`, so every step is a pure function of its state.
 */

import { normals, stream, type Stream } from 'aifn/random'
import { add, mul, sub, type Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { predictClean, predictNoise, type NoisePredictor } from './predictor'
import { alphaBarAt, betaAt, type ForwardSde, type NoiseSchedule } from './schedules'

/** Where sampling starts: given points, or n fresh draws from the prior in `dimension` dimensions. */
export type SamplerStart = { x: Tensor } | { n: number; dimension: number }

/** The state of every sampler. */
export type SamplerState = {
  /** Steps taken. */
  k: number
  /** The current noise level: a discrete step t (DDPM, DDIM; 0 is clean) or a continuous time (SDE, ODE). */
  time: number
  /** The particles, shape [n, d]. */
  x: Tensor
  /** The predicted noise ε̂ at the previous point, and the clean point x̂₀ it implies (null at the start). */
  noise: Tensor | null
  clean: Tensor | null
  /** Noise-predictor evaluations so far. */
  evaluations: number
  stream: Stream
}

function start(opts: SamplerStart, s: Stream | undefined, name: string, priorStd: number, time: number): SamplerState {
  const str = s ?? stream(name)
  const x = 'x' in opts ? opts.x : mul(normals(str.child('prior'), [opts.n, opts.dimension]), priorStd)
  return { k: 0, time, x, noise: null, clean: null, evaluations: 0, stream: str }
}

/** Options of `ddpmSampler`. */
export type DdpmOptions = {
  /**
   * The reverse-step variance σₜ²: `beta` (βₜ, the default) or `posterior` (β̃ₜ), the two choices Ho et al. found to
   * give similar samples (§3.2).
   */
  variance?: 'beta' | 'posterior'
}

/**
 * DDPM ancestral sampling (Ho et al., 2020, Algorithm 2): from x_T ~ N(0, I), for t = T … 1,
 * x_{t−1} = (x_t − βₜ/√(1 − ᾱₜ)·ε̂(x_t, ᾱₜ))/√αₜ + σₜz, with no noise on the last step. T steps.
 */
export function ddpmSampler(
  predictor: NoisePredictor,
  schedule: NoiseSchedule,
  { variance = 'beta' }: DdpmOptions = {},
): Algorithm<SamplerStart, SamplerState> {
  return {
    name: 'ddpm',
    init: (opts, s) => start(opts, s, 'ddpm', 1, schedule.steps),
    step: (st) => {
      const t = st.time
      const beta = betaAt(schedule, t)
      const ab = alphaBarAt(schedule, t)
      const noise = predictor(st.x, ab)
      let x = mul(sub(st.x, mul(noise, beta / Math.sqrt(1 - ab))), 1 / Math.sqrt(1 - beta))
      if (t > 1) {
        const sigma2 = variance === 'beta' ? beta : ((1 - alphaBarAt(schedule, t - 1)) / (1 - ab)) * beta
        x = add(x, mul(normals(st.stream.child('step', st.k), st.x.shape), Math.sqrt(sigma2)))
      }
      return {
        ...st,
        k: st.k + 1,
        time: t - 1,
        x,
        noise,
        clean: predictClean(noise, st.x, Math.sqrt(ab), Math.sqrt(1 - ab)),
        evaluations: st.evaluations + 1,
      }
    },
    done: (st) => st.time <= 0,
  }
}

/** Options of `ddimSampler`. */
export type DdimOptions = {
  /** Number of sampling steps S ≤ T; the levels used are S evenly spaced steps from T down to 1. Default 50. */
  steps?: number
  /** η ∈ [0, 1]: 0 is the deterministic DDIM, 1 matches DDPM's posterior variance. Default 0. */
  eta?: number
}

/** S evenly spaced steps from T down to 1 (distinct integers), then 0. */
export function ddimTimesteps(T: number, S: number): number[] {
  const n = Math.max(1, Math.min(S, T))
  const out = new Set<number>()
  for (let i = n - 1; i >= 0; i--) out.add(n === 1 ? T : Math.round(1 + (i * (T - 1)) / (n - 1)))
  return [...out, 0]
}

/**
 * DDIM (Song, Meng & Ermon, 2021, eq. 12) from level t to the next level t′ < t of the subsequence:
 * x̂₀ = (x_t − √(1 − ᾱₜ)ε̂)/√ᾱₜ, x_{t′} = √ᾱ_{t′}x̂₀ + √(1 − ᾱ_{t′} − σ²)·ε̂ + σz, with
 * σ = η√((1 − ᾱ_{t′})/(1 − ᾱₜ))·√(1 − ᾱₜ/ᾱ_{t′}). With η = 0 the map from x_T to x₀ is deterministic.
 */
export function ddimSampler(
  predictor: NoisePredictor,
  schedule: NoiseSchedule,
  { steps = 50, eta = 0 }: DdimOptions = {},
): Algorithm<SamplerStart, SamplerState> {
  const levels = ddimTimesteps(schedule.steps, steps)
  return {
    name: 'ddim',
    init: (opts, s) => start(opts, s, 'ddim', 1, levels[0]),
    step: (st) => {
      const t = st.time
      const next = levels[st.k + 1]
      const ab = alphaBarAt(schedule, t)
      const abNext = alphaBarAt(schedule, next)
      const noise = predictor(st.x, ab)
      const clean = predictClean(noise, st.x, Math.sqrt(ab), Math.sqrt(1 - ab))
      const sigma = eta * Math.sqrt((1 - abNext) / (1 - ab)) * Math.sqrt(1 - ab / abNext)
      let x = add(mul(clean, Math.sqrt(abNext)), mul(noise, Math.sqrt(Math.max(0, 1 - abNext - sigma * sigma))))
      if (sigma > 0) x = add(x, mul(normals(st.stream.child('step', st.k), st.x.shape), sigma))
      return { ...st, k: st.k + 1, time: next, x, noise, clean, evaluations: st.evaluations + 1 }
    },
    done: (st) => st.time <= 0,
  }
}

/** Options of the continuous-time samplers. */
export type ContinuousOptions = {
  /** Number of steps from t = 1 to `end`. Default 500 for the SDE, 100 for the ODE. */
  steps?: number
  /** The final time ε > 0 (the score is singular at t = 0). Default 1e-3. */
  end?: number
}

/**
 * The reverse-time SDE by Euler–Maruyama (Song et al., 2021, eq. 6 and appendix D): with h = (1 − ε)/N and time
 * falling from 1, x ← x − [f(t)x − g(t)²∇log p_t(x)]h + g(t)√h·z.
 */
export function reverseSdeSampler(
  predictor: NoisePredictor,
  sde: ForwardSde,
  { steps = 500, end = 1e-3 }: ContinuousOptions = {},
): Algorithm<SamplerStart, SamplerState> {
  const h = (1 - end) / steps
  return {
    name: 'reverse-sde',
    init: (opts, s) => start(opts, s, 'reverse-sde', sde.priorStd, 1),
    step: (st) => {
      const t = st.time
      const m = sde.meanScale(t)
      const sd = sde.std(t)
      const g = sde.diffusion(t)
      const noise = predictNoise(predictor, st.x, m, sd)
      const score = mul(noise, -1 / sd)
      const drift = sub(mul(st.x, sde.drift(t)), mul(score, g * g))
      const z = normals(st.stream.child('step', st.k), st.x.shape)
      const x = add(sub(st.x, mul(drift, h)), mul(z, g * Math.sqrt(h)))
      const time = st.k + 1 === steps ? end : 1 - (st.k + 1) * h
      return {
        ...st,
        k: st.k + 1,
        time,
        x,
        noise,
        clean: predictClean(noise, st.x, m, sd),
        evaluations: st.evaluations + 1,
      }
    },
    done: (st) => st.k >= steps,
  }
}

/** Options of `probabilityFlowSampler`. */
export type ProbabilityFlowOptions = ContinuousOptions & {
  /** `rk4` (default; four predictor calls per step) or `euler` (one). */
  method?: 'rk4' | 'euler'
}

/**
 * The probability-flow ODE dx/dt = f(t)x − ½g(t)²∇log p_t(x) (Song et al., 2021, eq. 13) from t = 1 to ε, whose
 * solutions have the same marginals p_t as the SDE but are deterministic given x_T. Integrated with the classical
 * fourth-order Runge–Kutta method (Kutta, 1901), time falling in N equal steps. (A private integrator: `aifn/ode`
 * was not available when this module was written.)
 */
export function probabilityFlowSampler(
  predictor: NoisePredictor,
  sde: ForwardSde,
  { steps = 100, end = 1e-3, method = 'rk4' }: ProbabilityFlowOptions = {},
): Algorithm<SamplerStart, SamplerState> {
  const h = (1 - end) / steps
  // The ODE's velocity at (x, t), and the noise prediction it came from.
  const flow = (x: Tensor, t: number) => {
    const g = sde.diffusion(t)
    const sd = sde.std(t)
    const noise = predictNoise(predictor, x, sde.meanScale(t), sd)
    return { v: sub(mul(x, sde.drift(t)), mul(noise, (-0.5 * g * g) / sd)), noise }
  }
  const velocity = (x: Tensor, t: number) => flow(x, t).v
  return {
    name: 'probability-flow',
    init: (opts, s) => start(opts, s, 'probability-flow', sde.priorStd, 1),
    step: (st) => {
      const t = st.time
      const dt = -h
      const first = flow(st.x, t)
      let x: Tensor
      let evaluations = st.evaluations + 1
      if (method === 'euler') x = add(st.x, mul(first.v, dt))
      else {
        const k1 = first.v
        const k2 = velocity(add(st.x, mul(k1, dt / 2)), t + dt / 2)
        const k3 = velocity(add(st.x, mul(k2, dt / 2)), t + dt / 2)
        const k4 = velocity(add(st.x, mul(k3, dt)), t + dt)
        x = add(st.x, mul(add(add(k1, mul(k2, 2)), add(mul(k3, 2), k4)), dt / 6))
        evaluations += 3
      }
      const time = st.k + 1 === steps ? end : 1 - (st.k + 1) * h
      const clean = predictClean(first.noise, st.x, sde.meanScale(t), sde.std(t))
      return { ...st, k: st.k + 1, time, x, noise: first.noise, clean, evaluations }
    },
    done: (st) => st.k >= steps,
  }
}
