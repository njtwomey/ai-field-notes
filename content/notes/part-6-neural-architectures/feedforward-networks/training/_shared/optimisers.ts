/** First-order optimisers on two-dimensional test surfaces, for the optimiser notes in this category. */

export type Vec = [number, number]
export type LossAndGrad = (p: Vec) => { loss: number; grad: Vec }

export type SurfaceId = 'valley' | 'rotated' | 'rosenbrock'

export type Surface = {
  id: SurfaceId
  label: string
  f: LossAndGrad
  xRange: Vec
  yRange: Vec
  start: Vec
  minimum: Vec
  /** Default step size for SGD, momentum and Nesterov. */
  eta: number
  /** Default step size for AdaGrad, RMSProp and Adam. */
  alpha: number
}

/** Condition number of the quadratic valleys: curvature 1 along one axis, 25 along the other. */
export const KAPPA = 25

const valley: LossAndGrad = ([x, y]) => ({ loss: 0.5 * (x * x + KAPPA * y * y), grad: [x, KAPPA * y] })

const C = Math.SQRT1_2
/** The valley rotated by 45°: evaluate at the rotated point u = Rᵀp, rotate the gradient back. */
const rotated: LossAndGrad = ([x, y]) => {
  const u = C * x + C * y
  const v = -C * x + C * y
  const { loss, grad } = valley([u, v])
  return { loss, grad: [C * grad[0] - C * grad[1], C * grad[0] + C * grad[1]] }
}

/** Rosenbrock's function with the curved valley's steepness 10 rather than 100. */
const rosenbrock: LossAndGrad = ([x, y]) => {
  const r = y - x * x
  return { loss: (1 - x) ** 2 + 10 * r * r, grad: [-2 * (1 - x) - 40 * x * r, 20 * r] }
}

export const SURFACES: Surface[] = [
  {
    id: 'valley',
    label: 'valley',
    f: valley,
    xRange: [-3, 3],
    yRange: [-2, 2],
    start: [-2.5, 1],
    minimum: [0, 0],
    eta: 0.03,
    alpha: 0.1,
  },
  {
    id: 'rotated',
    label: 'rotated valley',
    f: rotated,
    xRange: [-3, 3],
    yRange: [-2, 2],
    // The axis-aligned start rotated by 45°, so the SGD family traces a rotated copy of its valley path.
    start: [-2.47, -1.06],
    minimum: [0, 0],
    eta: 0.03,
    alpha: 0.1,
  },
  {
    id: 'rosenbrock',
    label: 'Rosenbrock',
    f: rosenbrock,
    xRange: [-2, 2],
    yRange: [-1, 3],
    start: [-1.2, 1.5],
    minimum: [1, 1],
    eta: 0.01,
    alpha: 0.1,
  },
]

export type OptimiserId = 'sgd' | 'momentum' | 'nesterov' | 'adagrad' | 'rmsprop' | 'adam'

export const OPTIMISERS: { id: OptimiserId; label: string; adaptive: boolean }[] = [
  { id: 'sgd', label: 'SGD', adaptive: false },
  { id: 'momentum', label: 'momentum', adaptive: false },
  { id: 'nesterov', label: 'Nesterov', adaptive: false },
  { id: 'adagrad', label: 'AdaGrad', adaptive: true },
  { id: 'rmsprop', label: 'RMSProp', adaptive: true },
  { id: 'adam', label: 'Adam', adaptive: true },
]

export const BETA = 0.9
export const RHO = 0.9
export const BETA2 = 0.999
const EPS = 1e-8
const DIVERGED = 1e8

export type Run = { path: Vec[]; losses: number[]; diverged: boolean }

export function optimise(f: LossAndGrad, start: Vec, opt: OptimiserId, lr: number, steps: number): Run {
  let p: Vec = [...start]
  let v: Vec = [0, 0]
  let m: Vec = [0, 0]
  let s: Vec = [0, 0]
  const path: Vec[] = [p]
  const losses = [f(p).loss]
  for (let t = 1; t <= steps; t++) {
    // Nesterov evaluates the gradient at the look-ahead point p + βv.
    const at: Vec = opt === 'nesterov' ? [p[0] + BETA * v[0], p[1] + BETA * v[1]] : p
    const g = f(at).grad
    let step: Vec
    switch (opt) {
      case 'sgd':
        step = [-lr * g[0], -lr * g[1]]
        break
      case 'momentum':
      case 'nesterov':
        v = [BETA * v[0] - lr * g[0], BETA * v[1] - lr * g[1]]
        step = v
        break
      case 'adagrad':
        s = [s[0] + g[0] ** 2, s[1] + g[1] ** 2]
        step = [(-lr * g[0]) / (Math.sqrt(s[0]) + EPS), (-lr * g[1]) / (Math.sqrt(s[1]) + EPS)]
        break
      case 'rmsprop':
        s = [RHO * s[0] + (1 - RHO) * g[0] ** 2, RHO * s[1] + (1 - RHO) * g[1] ** 2]
        step = [(-lr * g[0]) / (Math.sqrt(s[0]) + EPS), (-lr * g[1]) / (Math.sqrt(s[1]) + EPS)]
        break
      case 'adam': {
        m = [BETA * m[0] + (1 - BETA) * g[0], BETA * m[1] + (1 - BETA) * g[1]]
        s = [BETA2 * s[0] + (1 - BETA2) * g[0] ** 2, BETA2 * s[1] + (1 - BETA2) * g[1] ** 2]
        const c1 = 1 - BETA ** t
        const c2 = 1 - BETA2 ** t
        step = [(-lr * (m[0] / c1)) / (Math.sqrt(s[0] / c2) + EPS), (-lr * (m[1] / c1)) / (Math.sqrt(s[1] / c2) + EPS)]
        break
      }
    }
    p = [p[0] + step[0], p[1] + step[1]]
    const loss = f(p).loss
    if (!Number.isFinite(loss) || loss > DIVERGED) return { path, losses, diverged: true }
    path.push(p)
    losses.push(loss)
  }
  return { path, losses, diverged: false }
}

/** First step at which the loss falls below `tol`, or undefined if it never does. */
export function stepsTo(losses: number[], tol: number): number | undefined {
  const i = losses.findIndex((l) => l < tol)
  return i === -1 ? undefined : i
}
