import { describe, expect, it } from 'vitest'
import {
  adagrad,
  adam,
  adamw,
  backtracking,
  bfgs,
  cmaEs,
  conjugateGradient,
  coordinateDescent,
  fista,
  gaussNewton,
  gradientDescent,
  ista,
  lbfgs,
  leastSquares,
  levenbergMarquardt,
  linearConjugateGradient,
  minimize,
  momentum,
  nelderMead,
  nesterov,
  newton,
  projectBall,
  projectBox,
  projectSimplex,
  projectedGradient,
  proxL1,
  proxL2,
  proxSquaredL2,
  rmsprop,
  simulatedAnnealing,
  solveConjugateGradient,
  strongWolfe,
  trustRegion,
  type IterateState,
  type Method,
  type Objective,
  type StartOptions,
} from 'aifn/optim'
import { beale, himmelblau, quadraticBowl, rastrigin, rosenbrock } from 'aifn/datasets'
import { stream } from 'aifn/random'
import { tensor, toFlat } from 'aifn/tensor'
import { extend, run, seek, trace, type Algorithm } from 'aifn/trace'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol)
}

const rosen = rosenbrock()

describe('test functions', () => {
  it('gradients and Hessians match finite differences', () => {
    const fns = [
      rosenbrock(),
      rosenbrock({ n: 4 }),
      himmelblau(),
      beale(),
      quadraticBowl({ condition: 30, angle: 0.4 }),
      rastrigin(),
    ]
    for (const fn of fns) {
      const x = Array.from({ length: fn.dimension }, (_, i) => 0.3 + 0.17 * i)
      const { grad } = fn.objective(tensor(x))
      const H = toFlat(fn.hessian(tensor(x)))
      const g = toFlat(grad as never)
      const h = 1e-6
      for (let i = 0; i < x.length; i++) {
        const up = [...x]
        const down = [...x]
        up[i] += h
        down[i] -= h
        expect((fn.value(tensor(up)) - fn.value(tensor(down))) / (2 * h)).toBeCloseTo(g[i], 4)
        const gu = toFlat(fn.objective(tensor(up)).grad as never)
        const gd = toFlat(fn.objective(tensor(down)).grad as never)
        for (let j = 0; j < x.length; j++) expect((gu[j] - gd[j]) / (2 * h)).toBeCloseTo(H[i * x.length + j], 3)
      }
      for (const m of fn.minima) expect(fn.value(m)).toBeCloseTo(fn.minimumValue, 10)
    }
  })
})

describe('line searches', () => {
  it('backtracking satisfies Armijo and strong Wolfe satisfies both conditions', () => {
    const x = [-1.2, 1]
    const { grad } = rosen.objective(tensor(x))
    const p = toFlat(grad as never).map((v) => -v)
    const b = backtracking(rosen.objective, x, p)
    expect(b.converged && b.armijo).toBe(true)
    expect(b.trials.length).toBeGreaterThan(1)
    const w = strongWolfe(rosen.objective, x, p, { c2: 0.1 })
    expect(w.converged && w.armijo && w.curvature).toBe(true)
    expect(Math.abs(w.trials.at(-1)!.slope)).toBeLessThanOrEqual(0.1 * Math.abs(w.initialSlope))
  })
  it('fails visibly on an ascent direction', () => {
    const r = backtracking(rosen.objective, [0, 0], [-1, 0])
    expect(r.converged).toBe(false)
    expect(r.alpha).toBe(0)
  })
})

describe('first-order methods', () => {
  const bowl = quadraticBowl({ condition: 10 })
  it('gradient descent on a quadratic converges linearly at rate max|1 − ηλ|', () => {
    const lr = 0.1
    const t = trace(gradientDescent(bowl.objective, { lr, tolerance: 0 }), { x0: bowl.start }, 40, {
      record: { value: (s) => s.value },
    })
    const v = toFlat(t.series.value)
    // f ∝ (1 − ηλ_min)^{2t} asymptotically: λ_min = 1 gives ratio 0.81.
    expect(v[40] / v[39]).toBeCloseTo(0.81, 3)
  })
  it('every first-order method decreases a convex bowl', () => {
    const algs = [
      momentum(bowl.objective, { lr: 0.05 }),
      nesterov(bowl.objective, { lr: 0.05 }),
      adagrad(bowl.objective, { lr: 0.5 }),
      rmsprop(bowl.objective, { lr: 0.05 }),
      adam(bowl.objective, { lr: 0.1 }),
      adamw(bowl.objective, { lr: 0.1, weightDecay: 0 }),
      gradientDescent(bowl.objective, { lr: 1, lineSearch: 'backtracking' }),
    ]
    for (const alg of algs) {
      const s = run(alg, { x0: bowl.start }, 500)
      expect(s.value).toBeLessThan(1e-3)
      expect(s.diverged).toBe(false)
    }
  })
  it('flags divergence when the step is too large', () => {
    const s = run(gradientDescent(bowl.objective, { lr: 0.25 }), { x0: bowl.start }, 10000)
    expect(s.diverged).toBe(true)
  })
})

describe('second-order and quasi-Newton methods reach the Rosenbrock minimum', () => {
  it.each([
    ['newton', () => run(newton(rosen.objective, { hessian: rosen.hessian }), { x0: rosen.start }, 200)],
    ['trust region', () => run(trustRegion(rosen.objective, { hessian: rosen.hessian }), { x0: rosen.start }, 500)],
    ['bfgs', () => run(bfgs(rosen.objective), { x0: rosen.start }, 500)],
    ['lbfgs', () => run(lbfgs(rosen.objective), { x0: rosen.start }, 500)],
    [
      'fletcher-reeves',
      () => run(conjugateGradient(rosen.objective, { variant: 'fletcher-reeves' }), { x0: rosen.start }, 5000),
    ],
    ['polak-ribiere', () => run(conjugateGradient(rosen.objective), { x0: rosen.start }, 2000)],
  ])('%s', (_, go) => {
    const s = go()
    expect(s.converged).toBe(true)
    close(toFlat(s.x), [1, 1], 1e-4)
  })

  it('pure Newton converges quadratically near the minimum', () => {
    const t = trace(
      newton(rosen.objective, { hessian: rosen.hessian, lineSearch: 'none', tolerance: 0 }),
      { x0: [1.1, 1.2] },
      8,
      {
        record: { error: (s) => Math.hypot(toFlat(s.x)[0] - 1, toFlat(s.x)[1] - 1) },
      },
    )
    const e = toFlat(t.series.error)
    // e_{k+1} ≤ C e_k² with a moderate C once close.
    expect(e[5] / e[4] ** 2).toBeLessThan(1e2)
    expect(e[6] / e[5] ** 2).toBeLessThan(1e2)
    expect(e[6]).toBeLessThan(1e-10)
  })
})

describe('least squares', () => {
  // Fit y = a·exp(b·t) to exact data from a = 2, b = −0.5.
  const ts = [0, 0.5, 1, 1.5, 2, 3]
  const residuals = (x: Parameters<Objective>[0]) => {
    const [a, b] = toFlat(x)
    return {
      residuals: ts.map((t) => a * Math.exp(b * t) - 2 * Math.exp(-0.5 * t)),
      jacobian: ts.map((t) => [Math.exp(b * t), a * t * Math.exp(b * t)]),
    }
  }
  it('Gauss–Newton and Levenberg–Marquardt recover the parameters', () => {
    for (const method of ['gauss-newton', 'levenberg-marquardt'] as const) {
      const r = leastSquares(residuals, [1, 0], { method, tolerance: 1e-12 })
      expect(r.converged).toBe(true)
      close(toFlat(r.x), [2, -0.5], 1e-8)
    }
    expect(run(gaussNewton(residuals), { x0: [1, 0] }, 50).converged).toBe(true)
    expect(run(levenbergMarquardt(residuals, { scaling: 'marquardt' }), { x0: [1, 0] }, 200).converged).toBe(true)
  })
})

describe('linear conjugate gradients', () => {
  it('solves an SPD system in n steps', () => {
    const A = [
      [4, 1, 0],
      [1, 3, 1],
      [0, 1, 2],
    ]
    const b = [1, 2, 3]
    const r = solveConjugateGradient(A, b)
    expect(r.converged).toBe(true)
    expect(r.steps).toBeLessThanOrEqual(3)
    const x = toFlat(r.x)
    close(
      A.map((row) => row.reduce((s, a, j) => s + a * x[j], 0)),
      b,
      1e-10,
    )
    expect(
      run(
        linearConjugateGradient((v) => v, [1, 2]),
        {},
        5,
      ).converged,
    ).toBe(true)
  })
})

describe('derivative-free methods', () => {
  it('Nelder–Mead finds the Rosenbrock minimum', () => {
    const s = run(nelderMead(rosen.value), { x0: rosen.start }, 2000)
    expect(s.converged).toBe(true)
    close(toFlat(s.x), [1, 1], 1e-4)
  })
  it('coordinate descent with Newton coordinate steps solves a quadratic', () => {
    const bowl = quadraticBowl({ condition: 5, angle: 0.5 })
    for (const rule of ['cyclic', 'random', 'greedy'] as const) {
      const s = run(coordinateDescent(bowl.objective, { rule, hessian: bowl.hessian }), { x0: bowl.start }, 2000, {
        stream: stream(3),
      })
      expect(s.converged).toBe(true)
    }
  })
  it('CMA-ES finds the Rosenbrock minimum, and simulated annealing lowers Rastrigin', () => {
    const s = run(cmaEs(rosen.value), { x0: rosen.start }, 1000, { stream: stream(1) })
    expect(s.converged).toBe(true)
    close(toFlat(s.x), [1, 1], 1e-4)
    const r = rastrigin()
    const a = run(simulatedAnnealing(r.value, { temperature: 5 }), { x0: r.start }, 3000, { stream: stream(2) })
    expect(a.bestValue).toBeLessThan(r.value(r.start))
  })
})

describe('proximal and projected methods', () => {
  // Lasso: ½‖x − c‖² + λ‖x‖₁ has the soft-threshold solution.
  const c = [3, -0.5, 1.2]
  const f: Objective = (x) => {
    const v = toFlat(x)
    const d = v.map((vi, i) => vi - c[i])
    return { value: 0.5 * d.reduce((s, di) => s + di * di, 0), grad: d }
  }
  it('ISTA and FISTA reach the soft-threshold solution', () => {
    for (const alg of [ista(f, proxL1(1), { lr: 0.5 }), fista(f, proxL1(1), { lr: 0.5, backtracking: true })]) {
      const s = run(alg, { x0: [0, 0, 0] }, 500)
      expect(s.converged).toBe(true)
      close(toFlat(s.x), [2, 0, 0.2], 1e-6)
    }
    expect(toFlat(run(ista(f, proxSquaredL2(1), { lr: 0.5 }), { x0: [0, 0, 0] }, 500).x)[0]).toBeCloseTo(1.5, 5)
    expect(run(ista(f, proxL2(1), { lr: 0.5 }), { x0: [0, 0, 0] }, 500).converged).toBe(true)
  })
  it('projected gradient stays in the set', () => {
    const s = run(projectedGradient(f, projectBox(0, 1), { lr: 0.5 }), { x0: [0.5, 0.5, 0.5] }, 200)
    close(toFlat(s.x), [1, 0, 1], 1e-6)
    close(toFlat(projectSimplex()(tensor([0.5, 0.8, -1]))), [0.35, 0.65, 0], 1e-12)
    expect(Math.hypot(...toFlat(projectBall(1)(tensor([3, 4]))))).toBeCloseTo(1, 12)
  })
})

describe('minimize', () => {
  it('runs every method by name', () => {
    const bowl = quadraticBowl({ condition: 4 })
    const methods: Method[] = [
      'gradient-descent',
      'momentum',
      'nesterov',
      'adam',
      'bfgs',
      'lbfgs',
      'conjugate-gradient',
      'nelder-mead',
    ]
    for (const method of methods) {
      const r = minimize(bowl.objective, bowl.start, { method, maxSteps: 3000, lr: 0.1 } as never)
      expect(r.value).toBeLessThan(1e-6)
    }
    expect(minimize(bowl.objective, bowl.start, { method: 'newton', hessian: bowl.hessian }).converged).toBe(true)
  })
})

describe('protocol', () => {
  const algs: Record<string, Algorithm<StartOptions, IterateState>> = {
    lbfgs: lbfgs(rosen.objective),
    nelderMead: nelderMead(rosen.value),
    cmaEs: cmaEs(rosen.value),
    annealing: simulatedAnnealing(rosen.value),
  }
  it.each(Object.entries(algs))(
    '%s: same stream → same trace; seek equals run; extend equals a longer trace',
    (_, alg) => {
      const opts = { x0: rosen.start }
      const record = { value: (s: { value: number }) => s.value }
      const a = trace(alg, opts, 30, { record, stream: stream(5) })
      const b = trace(alg, opts, 30, { record, stream: stream(5) })
      expect(toFlat(a.series.value)).toEqual(toFlat(b.series.value))
      const r = run(alg, opts, 17, { stream: stream(5) })
      expect(toFlat(seek(alg, opts, 17, { stream: stream(5) }).x)).toEqual(toFlat(r.x))
      const short = trace(alg, opts, 12, { record, stream: stream(5) })
      const longer = extend(short, alg, opts, 18)
      expect(toFlat(longer.series.value)).toEqual(toFlat(a.series.value))
    },
  )
})
