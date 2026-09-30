/**
 * `aifn/sde`: stochastic differential equations dX = a(t, X) dt + b(t, X) dW with diagonal noise, integrated for a
 * whole cloud of paths at once. Every scheme is a traceable `Algorithm`; the Brownian increments of step k come from
 * `stream.child('step', k)` in row-major order, so path i's increments depend only on the stream, the step and i:
 * raising the number of paths adds paths without changing the existing ones, and two schemes (or an exact solution)
 * given the same stream see the same Brownian path.
 *
 * - Schemes: `eulerMaruyama` (strong order ½, weak order 1), `milstein` (strong order 1; the derivative ∂b/∂x by
 *   autodiff unless given), `stochasticRungeKutta` (Platen's derivative-free strong order 1 scheme).
 * - Exact solutions: `ornsteinUhlenbeck` and `geometricBrownianMotion` (moments, transition laws and exact samplers
 *   on the same streams), and `brownianMotion`.
 * - `paths` extracts a path matrix from a trace; `densityEvolution` solves the Fokker–Planck equation of a scalar SDE
 *   with `aifn/pde`.
 *
 * References: Kloeden & Platen (1992), "Numerical Solution of Stochastic Differential Equations", §9–11; Higham
 * (2001), "An algorithmic introduction to numerical simulation of stochastic differential equations", SIAM Review 43.
 */

import { grad } from 'aifn/autodiff'
import { fokkerPlanck, type Grid1, type PdeState, type Profile, type TimeScheme } from 'aifn/pde'
import { normals, type Stream } from 'aifn/random'
import { fromData, isTensor, mul, sub, sum, toFlat, type Matrix, type Tensor, type Value } from 'aifn/tensor'
import type { Algorithm, Trace } from 'aifn/trace'

type F64 = Float64Array<ArrayBuffer>

/**
 * An SDE with diagonal noise, evaluated on a whole cloud of states: `drift` and `diffusion` take the time and the
 * states x (any shape, e.g. [N] for N paths of a scalar SDE or [N, d]) and return a value of the same shape (or a
 * number, broadcast). Written with `aifn/tensor` primitives they apply elementwise and can be differentiated.
 */
export type Sde = {
  drift: (t: number, x: Tensor) => Value
  diffusion: (t: number, x: Tensor) => Value
  /** ∂b/∂x elementwise (for Milstein); by autodiff of Σ b when omitted, which is right for diagonal noise. */
  diffusionDerivative?: (t: number, x: Tensor) => Value
}

/** The initial cloud: `x0` (a number, or an array/tensor of one state), repeated for `paths` paths. */
export type SdeInitial = { x0: number | ArrayLike<number> | Tensor; paths?: number; t0?: number }

/** The state of an SDE scheme. */
export type SdeState = {
  t: number
  /** The states of all paths: [paths] for a scalar SDE, [paths, d] otherwise. */
  x: Tensor
  step: number
  h: number
  /** The Brownian increments ΔW of the last step (same shape as x; zeros at t₀). */
  dW: Tensor
  /** The stream the increments are drawn from (step k uses `stream.child('step', k)`). */
  stream: Stream
  /** Evaluations of the drift and diffusion (each over the whole cloud) so far. */
  evaluations: number
  diverged: boolean
  /** The number of paths whose state is not finite. */
  nonFinite: number
}

/** Options common to the schemes. */
export type SdeOptions = {
  /** The step size (positive). */
  h: number
  /** Stop on reaching this time (the last step is shortened). */
  tEnd?: number
}

function cloud(value: Value, shape: readonly number[], where: string): F64 {
  const size = shape.reduce((a, b) => a * b, 1)
  if (typeof value === 'number') return new Float64Array(size).fill(value)
  const flat = isTensor(value)
    ? Float64Array.from(toFlat(value))
    : Array.isArray(value) || ArrayBuffer.isView(value)
      ? Float64Array.from(value as ArrayLike<number>)
      : null
  if (!flat) throw new Error(`${where}: expected a number, tensor or array`)
  if (flat.length === 1) return new Float64Array(size).fill(flat[0])
  if (flat.length !== size) throw new Error(`${where}: returned ${flat.length} values for ${size} states`)
  return flat
}

function initial({ x0, paths = 1, t0 = 0 }: SdeInitial, s: Stream | undefined, where: string): SdeState {
  if (!s) throw new Error(`${where}: pass a stream (trace(alg, opts, n, { stream }))`)
  const one = typeof x0 === 'number' ? [x0] : isTensor(x0) ? toFlat(x0) : Array.from(x0)
  const scalar = typeof x0 === 'number' || one.length === 1
  const d = one.length
  const shape = scalar ? [paths] : [paths, d]
  const x = new Float64Array(paths * d)
  for (let i = 0; i < paths; i++) x.set(one, i * d)
  return {
    t: t0,
    x: fromData(x, shape),
    step: 0,
    h: 0,
    dW: fromData(new Float64Array(paths * d), shape),
    stream: s,
    evaluations: 0,
    diverged: !one.every(Number.isFinite),
    nonFinite: one.every(Number.isFinite) ? 0 : paths,
  }
}

/** The Brownian increments of step k: N(0, |h|) for every element, from `stream.child('step', k)`. */
export function increments(s: Stream, step: number, shape: readonly number[], h: number): Tensor {
  return fromData(Float64Array.from(toFlat(normals(s.child('step', step), [...shape], 0, Math.sqrt(Math.abs(h))))), [
    ...shape,
  ])
}

type Update = (sde: Sde, t: number, x: Tensor, h: number, dW: F64) => { next: F64; evaluations: number }

function scheme(name: string, sde: Sde, { h, tEnd }: SdeOptions, update: Update): Algorithm<SdeInitial, SdeState> {
  if (!(h > 0)) throw new Error(`${name}: the step size h must be positive`)
  return {
    name,
    init: (opts, s) => initial(opts, s, name),
    step: (s) => {
      const hk = tEnd !== undefined && tEnd - s.t < h ? tEnd - s.t : h
      const dW = increments(s.stream, s.step, s.x.shape, hk)
      const { next, evaluations } = update(sde, s.t, s.x, hk, dW.data as F64)
      let nonFinite = 0
      for (const v of next) if (!Number.isFinite(v)) nonFinite++
      return {
        ...s,
        t: s.t + hk,
        x: fromData(next, s.x.shape),
        step: s.step + 1,
        h: hk,
        dW,
        evaluations: s.evaluations + evaluations,
        nonFinite,
        diverged: nonFinite === s.x.data.length,
      }
    },
    done: (s) => tEnd !== undefined && s.t >= tEnd - 1e-12 * Math.max(1, Math.abs(tEnd)),
  }
}

/**
 * The Euler–Maruyama scheme X_{n+1} = X_n + a(t_n, X_n) h + b(t_n, X_n) ΔW_n with ΔW_n ~ N(0, h): strong order ½
 * (pathwise error O(h^½)) and weak order 1 (error in expectations O(h)). `init` takes `{ x0, paths, t0 }` and a
 * stream. A path that becomes non-finite is counted in `nonFinite`; the run is `diverged` only when all have.
 */
export function eulerMaruyama(sde: Sde, options: SdeOptions): Algorithm<SdeInitial, SdeState> {
  return scheme('euler-maruyama', sde, options, (p, t, x, h, dW) => {
    const a = cloud(p.drift(t, x), x.shape, 'drift')
    const b = cloud(p.diffusion(t, x), x.shape, 'diffusion')
    const xs = x.data as F64
    return { next: Float64Array.from(xs, (v, i) => v + a[i] * h + b[i] * dW[i]), evaluations: 1 }
  })
}

/**
 * The Milstein scheme: Euler–Maruyama plus the Itô correction ½ b b′ (ΔW² − h), which raises the strong order to 1
 * for diagonal noise. The derivative b′ = ∂b/∂x comes from `diffusionDerivative` or by autodiff of Σ b(t, x).
 */
export function milstein(sde: Sde, options: SdeOptions): Algorithm<SdeInitial, SdeState> {
  const db =
    sde.diffusionDerivative ??
    ((t: number, x: Tensor) => grad((y: Value) => sum(sde.diffusion(t, y as Tensor) as Tensor) as Value)(x) as Value)
  return scheme('milstein', sde, options, (p, t, x, h, dW) => {
    const a = cloud(p.drift(t, x), x.shape, 'drift')
    const b = cloud(p.diffusion(t, x), x.shape, 'diffusion')
    const bp = cloud(db(t, x), x.shape, 'diffusionDerivative')
    const xs = x.data as F64
    return {
      next: Float64Array.from(xs, (v, i) => v + a[i] * h + b[i] * dW[i] + 0.5 * b[i] * bp[i] * (dW[i] * dW[i] - h)),
      evaluations: 2,
    }
  })
}

/**
 * Platen's derivative-free explicit scheme of strong order 1 (Kloeden & Platen, 1992, §11.1, eq. 11.1.3): with the
 * supporting value X̂ = X + a h + b √h, X_{n+1} = X + a h + b ΔW + (b(X̂) − b(X))(ΔW² − h)/(2√h). It replaces
 * Milstein's b′ by a finite difference, so the diffusion need not be differentiable.
 */
export function stochasticRungeKutta(sde: Sde, options: SdeOptions): Algorithm<SdeInitial, SdeState> {
  return scheme('stochastic-runge-kutta', sde, options, (p, t, x, h, dW) => {
    const a = cloud(p.drift(t, x), x.shape, 'drift')
    const b = cloud(p.diffusion(t, x), x.shape, 'diffusion')
    const xs = x.data as F64
    const sq = Math.sqrt(h)
    const support = fromData(
      Float64Array.from(xs, (v, i) => v + a[i] * h + b[i] * sq),
      x.shape,
    )
    const bs = cloud(p.diffusion(t, support), x.shape, 'diffusion')
    return {
      next: Float64Array.from(
        xs,
        (v, i) => v + a[i] * h + b[i] * dW[i] + ((bs[i] - b[i]) * (dW[i] * dW[i] - h)) / (2 * sq),
      ),
      evaluations: 3,
    }
  })
}

/**
 * The path matrix of a traced SDE run: entry [k, i] is path i at kept step k (for a d-dimensional SDE, its
 * `component`). Rows align with `trace.index` and `times`.
 */
export function paths(tr: Trace<SdeState>, component = 0): { times: Tensor; values: Matrix } {
  const first = tr.steps[0].x
  const n = first.shape[0]
  const d = first.shape.length > 1 ? first.shape[1] : 1
  const out = new Float64Array(tr.steps.length * n)
  tr.steps.forEach((s, k) => {
    const x = s.x.data as F64
    for (let i = 0; i < n; i++) out[k * n + i] = x[i * d + component]
  })
  return {
    times: fromData(
      Float64Array.from(tr.steps, (s) => s.t),
      [tr.steps.length],
    ),
    values: fromData(out, [tr.steps.length, n]),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Exact solutions

/** An SDE with a known exact solution: its law at time t given x₀, and an exact sampler on the scheme's streams. */
export type ExactSde = {
  sde: Sde
  /** E[X_t | X_0 = x₀]. */
  mean: (t: number, x0: number) => number
  /** Var[X_t | X_0 = x₀]. */
  variance: (t: number, x0: number) => number
  /** Exact transitions on a grid of step h, drawing the same Brownian increments as the schemes. */
  exact: (options: SdeOptions) => Algorithm<SdeInitial, SdeState>
}

function exactScheme(
  name: string,
  { h, tEnd }: SdeOptions,
  update: (x: number, t: number, h: number, dW: number) => number,
): Algorithm<SdeInitial, SdeState> {
  return scheme(name, { drift: () => 0, diffusion: () => 0 }, { h, tEnd }, (_p, t, x, hk, dW) => ({
    next: Float64Array.from(x.data as F64, (v, i) => update(v, t, hk, dW[i])),
    evaluations: 0,
  }))
}

/** Standard Brownian motion scaled by σ: dX = σ dW, with X_t ~ N(x₀, σ²t). */
export function brownianMotion(sigma = 1): ExactSde {
  return {
    sde: { drift: () => 0, diffusion: () => sigma },
    mean: (_t, x0) => x0,
    variance: (t) => sigma * sigma * t,
    exact: (o) => exactScheme('brownian-motion', o, (x, _t, _h, dW) => x + sigma * dW),
  }
}

/**
 * The Ornstein–Uhlenbeck process dX = θ(μ − X) dt + σ dW (θ > 0): X_t | x₀ ~ N(μ + (x₀ − μ)e^{−θt},
 * σ²(1 − e^{−2θt})/(2θ)), stationary N(μ, σ²/(2θ)). The exact sampler uses the transition over each step with the
 * step's increment rescaled to the right variance, X_{t+h} = μ + (X − μ)e^{−θh} + σ√((1 − e^{−2θh})/(2θh)) ΔW, so it
 * shares the schemes' streams (though not their exact Brownian path).
 */
export function ornsteinUhlenbeck({ theta, mu = 0, sigma }: { theta: number; mu?: number; sigma: number }): ExactSde {
  if (!(theta > 0)) throw new Error('ornsteinUhlenbeck: θ must be positive')
  return {
    sde: {
      drift: (_t, x) => mul(theta, sub(mu, x)),
      diffusion: () => sigma,
      diffusionDerivative: () => 0,
    },
    mean: (t, x0) => mu + (x0 - mu) * Math.exp(-theta * t),
    variance: (t) => (sigma * sigma * -Math.expm1(-2 * theta * t)) / (2 * theta),
    exact: (o) =>
      exactScheme('ornstein-uhlenbeck-exact', o, (x, _t, h, dW) => {
        const decay = Math.exp(-theta * h)
        const scale = sigma * Math.sqrt(-Math.expm1(-2 * theta * h) / (2 * theta * h))
        return mu + (x - mu) * decay + scale * dW
      }),
  }
}

/**
 * Geometric Brownian motion dX = μX dt + σX dW: X_t = x₀ exp((μ − σ²/2)t + σW_t), with E[X_t] = x₀e^{μt} and
 * Var[X_t] = x₀²e^{2μt}(e^{σ²t} − 1). The exact sampler applies the solution step by step with the same increments
 * the schemes draw, so it gives the true path that a scheme's path approximates (for strong-error measurements).
 */
export function geometricBrownianMotion({ mu, sigma }: { mu: number; sigma: number }): ExactSde {
  const scaled = (c: number) => (_t: number, x: Tensor) => mul(c, x)
  return {
    sde: { drift: scaled(mu), diffusion: scaled(sigma), diffusionDerivative: () => sigma },
    mean: (t, x0) => x0 * Math.exp(mu * t),
    variance: (t, x0) => x0 * x0 * Math.exp(2 * mu * t) * Math.expm1(sigma * sigma * t),
    exact: (o) =>
      exactScheme('gbm-exact', o, (x, _t, h, dW) => x * Math.exp((mu - 0.5 * sigma * sigma) * h + sigma * dW)),
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Densities

/**
 * The density of a scalar SDE dX = a(X) dt + σ(X) dW evolved by its Fokker–Planck equation p_t = −(ap)_x +
 * ½(σ²p)_xx on a grid (`aifn/pde`'s conservative solver with reflecting ends, default implicit steps). The solver's
 * state reports the mass and the stability number. `init` takes `{ u0 }`, the initial density.
 */
export function densityEvolution({
  drift,
  sigma,
  grid,
  dt,
  scheme: timeScheme = 'implicit',
  tEnd,
}: {
  drift: (x: number) => number
  sigma: (x: number) => number
  grid: Grid1
  dt: number
  scheme?: TimeScheme
  tEnd?: number
}): Algorithm<{ u0: Profile }, PdeState> {
  return fokkerPlanck({ drift, diffusion: (x) => 0.5 * sigma(x) ** 2, grid, dt, scheme: timeScheme, tEnd })
}
