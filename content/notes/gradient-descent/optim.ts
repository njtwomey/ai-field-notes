/** Heavy-ball gradient descent for the widgets in this note. Mirrors python/mlc/examples/gradient_descent/model.py. */

export type Vec = [number, number]
export type LossAndGrad = (theta: Vec) => { loss: number; grad: Vec }

export type Trajectory = { path: Vec[]; losses: number[]; diverged: boolean }

/** Loss above which a run stops and is reported as diverged. */
export const DIVERGED = 1e8

/** v ← βv − η∇L(θ), θ ← θ + v. With β = 0 this is plain gradient descent. */
export function descend(f: LossAndGrad, start: Vec, lr: number, steps: number, beta = 0): Trajectory {
  let theta: Vec = [...start]
  let v: Vec = [0, 0]
  const path: Vec[] = [theta]
  const losses: number[] = []
  for (let t = 0; t < steps; t++) {
    const { loss, grad } = f(theta)
    losses.push(loss)
    if (!Number.isFinite(loss) || loss > DIVERGED) return { path, losses, diverged: true }
    v = [beta * v[0] - lr * grad[0], beta * v[1] - lr * grad[1]]
    theta = [theta[0] + v[0], theta[1] + v[1]]
    path.push(theta)
  }
  losses.push(f(theta).loss)
  return { path, losses, diverged: false }
}

/** L − L*, floored so that a log axis stays finite once a run has converged. */
export const excess = (losses: number[], optimum: number) => losses.map((l) => Math.max(l - optimum, 1e-12))

/** Round a clicked grid coordinate to the slider precision. */
export const snap = (v: number) => Math.round(v * 100) / 100
