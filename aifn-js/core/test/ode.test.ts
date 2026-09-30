import { describe, expect, it } from 'vitest'
import {
  amplification,
  bdf,
  boundaryLocus,
  dormandPrince,
  hamiltonianSystem,
  implicitEuler,
  linearFlow,
  rungeKutta,
  solveIvp,
  stabilityRegion,
  symplectic,
  trapezoid,
  withEvents,
  type Rhs,
} from 'aifn/dynamics/ode'
import { eig, expm } from 'aifn/numerics/linalg'
import { get, mul, neg, stack, sub, sum, square, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'
import { fixture } from './fixtures'

type Fixture = {
  rk45: Record<string, { rtol: number; atol: number; t: number[]; x: number[][]; nfev: number }>
  lotka_volterra_x10: number[]
  expm: Record<string, { a: number[][]; expm: number[][] }>
  eig: Record<string, { a: number[][]; real: number[]; imag: number[] }>
}
const fx = fixture<Fixture>('ode')

const decay: Rhs = (_t, x) => neg(x)
// x′ = cos t · x, x(0) = 1: x(t) = exp(sin t). Time-dependent, so stage times are exercised.
const timeDependent: Rhs = (t, x) => mul(Math.cos(t), x)
const lotkaVolterra: Rhs = (_t, x) => {
  const [a, b] = [get(x, 0), get(x, 1)]
  return stack([sub(mul(1.5, a), mul(a, b)), sub(mul(a, b), mul(3, b))])
}

function observedOrder(errorAt: (h: number) => number, h = 0.1): number {
  return Math.log2(errorAt(h) / errorAt(h / 2))
}

describe('explicit Runge–Kutta', () => {
  it.each([
    ['euler', 1],
    ['heun', 2],
    ['midpoint', 2],
    ['rk4', 4],
  ] as const)('%s converges at order %i', (method, order) => {
    const err = (h: number) => {
      const s = run(rungeKutta(timeDependent, method, { h, tEnd: 2 }), { x0: [1] }, 10_000)
      return Math.abs(toFlat(s.x)[0] - Math.exp(Math.sin(2)))
    }
    expect(observedOrder(err, 0.05)).toBeCloseTo(order, 0)
  })

  it('counts evaluations and lands exactly on tEnd', () => {
    const s = run(rungeKutta(decay, 'rk4', { h: 0.3, tEnd: 1 }), { x0: [1] }, 100)
    expect(s.t).toBeCloseTo(1, 14)
    expect(s.step).toBe(4)
    expect(s.evaluations).toBe(16)
  })

  it('flags divergence instead of returning Infinity silently', () => {
    const tr = trace(
      rungeKutta((_t, x) => mul(x, x), 'euler', { h: 0.5 }),
      { x0: [2] },
      50,
    )
    expect(tr.meta.stopped).toBe('diverged')
    expect(tr.steps.at(-1)!.failure).toBe('not finite')
  })
})

describe('Dormand–Prince', () => {
  it.each(Object.entries(fx.rk45))('matches scipy RK45 step for step (rtol %s)', (_k, ref) => {
    const sol = solveIvp(lotkaVolterra, [0, 10], [10, 5], { rtol: ref.rtol, atol: ref.atol })
    expect(sol.t.shape[0]).toBe(ref.t.length)
    const t = toFlat(sol.t)
    ref.t.forEach((v, i) => expect(t[i]).toBeCloseTo(v, 9))
    const x = toRows(sol.x)
    ref.x.forEach((row, i) => row.forEach((v, j) => expect(x[i][j]).toBeCloseTo(v, 7)))
    expect(sol.evaluations).toBe(ref.nfev)
  })

  it('reaches the reference solution within its tolerance', () => {
    const sol = solveIvp(lotkaVolterra, [0, 10], [10, 5], { rtol: 1e-9, atol: 1e-12 })
    const end = toRows(sol.x).at(-1)!
    fx.lotka_volterra_x10.forEach((v, i) => expect(Math.abs(end[i] - v) / Math.abs(v)).toBeLessThan(1e-7))
  })

  it('rejects steps with large errors and records the attempts', () => {
    const tr = trace(dormandPrince(lotkaVolterra, { tEnd: 10, h0: 2 }), { x0: [10, 5] }, 1000)
    const first = tr.steps[1]
    expect(first.attempts.length).toBeGreaterThan(1)
    expect(first.attempts.at(-1)!.accepted).toBe(true)
    expect(first.attempts.slice(0, -1).every((a) => !a.accepted && a.error > 1)).toBe(true)
    expect(tr.steps.at(-1)!.rejected).toBeGreaterThan(0)
  })

  it('integrates backwards in time', () => {
    const sol = solveIvp(decay, [1, 0], [Math.exp(-1)], { rtol: 1e-10, atol: 1e-12 })
    expect(toFlat(sol.x).at(-1)!).toBeCloseTo(1, 8)
  })
})

describe('implicit methods', () => {
  it.each([
    ['implicit-euler', 1],
    ['trapezoid', 2],
    ['bdf2', 2],
    ['bdf3', 3],
  ] as const)('%s converges at order %i', (method, order) => {
    const err = (h: number) => {
      const sol = solveIvp(timeDependent, [0, 2], [1], { method, h })
      return Math.abs(toFlat(sol.x).at(-1)! - Math.exp(Math.sin(2)))
    }
    expect(observedOrder(err, 0.02)).toBeCloseTo(order, 0)
  })

  it('stays stable on a stiff decay where explicit Euler blows up', () => {
    const stiff: Rhs = (_t, x) => mul(-1000, x)
    const explicit = trace(rungeKutta(stiff, 'euler', { h: 0.01, tEnd: 10 }), { x0: [1] }, 1000)
    expect(explicit.meta.stopped).toBe('diverged')
    const implicit = run(implicitEuler(stiff, { h: 0.01, tEnd: 1 }), { x0: [1] }, 200)
    expect(Math.abs(toFlat(implicit.x)[0])).toBeLessThan(1e-10)
    // The trapezoid rule is A-stable but not L-stable: the stiff mode flips sign each step and barely decays.
    const trap = trace(trapezoid(stiff, { h: 0.1, tEnd: 0.5 }), { x0: [1] }, 10, { record: { x: (s) => s.x } })
    const xs = toFlat(trap.series.x)
    expect(xs[1] * xs[2]).toBeLessThan(0)
    expect(Math.abs(xs[5])).toBeGreaterThan(0.5)
  })

  it('gives the same answer with autodiff and finite-difference Jacobians', () => {
    const vanDerPol: Rhs = (_t, x) => {
      const [a, b] = [get(x, 0), get(x, 1)]
      return stack([b, sub(mul(mul(10, sub(1, mul(a, a))), b), a)])
    }
    const a = run(bdf(vanDerPol, 2, { h: 0.01, tEnd: 1 }), { x0: [2, 0] }, 1000)
    const b = run(bdf(vanDerPol, 2, { h: 0.01, tEnd: 1, jacobian: 'finite-difference' }), { x0: [2, 0] }, 1000)
    toFlat(a.x).forEach((v, i) => expect(v).toBeCloseTo(toFlat(b.x)[i], 6))
    expect(a.newtonConverged).toBe(true)
    expect(a.jacobianEvaluations).toBeGreaterThan(0)
  })
})

describe('symplectic integrators', () => {
  it.each(['symplectic-euler', 'leapfrog', 'velocity-verlet'] as const)(
    '%s keeps the energy error bounded over long times',
    (method) => {
      const H = { potential: (q: Tensor) => mul(0.5, sum(square(q))) } // harmonic oscillator
      const tr = trace(symplectic(H, method, { h: 0.1 }), { q0: [1], p0: [0] }, 20_000, {
        every: 100,
        record: { error: (s) => s.energyError },
      })
      const err = toFlat(tr.series.error).map(Math.abs)
      const early = Math.max(...err.slice(0, 20))
      const late = Math.max(...err.slice(-20))
      expect(late).toBeLessThan(1.5 * early + 1e-12)
    },
  )

  it('RK4 energy drifts steadily where leapfrog does not', () => {
    const H = { potential: (q: Tensor) => mul(0.5, sum(square(q))) }
    const { rhs, energy } = hamiltonianSystem(H)
    const rk = run(rungeKutta(rhs, 'rk4', { h: 0.5 }), { x0: [1, 0] }, 2000)
    const lf = run(symplectic(H, 'leapfrog', { h: 0.5 }), { q0: [1], p0: [0] }, 2000)
    expect(Math.abs(energy(rk.x) - 0.5)).toBeGreaterThan(0.05)
    expect(Math.abs(lf.energyError)).toBeLessThan(0.05)
  })

  it('velocity Verlet is second order and uses one force evaluation per step', () => {
    const H = { potential: (q: Tensor) => mul(0.5, sum(square(q))), potentialGradient: (q: Tensor) => q }
    const err = (h: number) => {
      const s = run(symplectic(H, 'velocity-verlet', { h, tEnd: 1 }), { q0: [1], p0: [0] }, 10_000)
      return Math.abs(toFlat(s.q)[0] - Math.cos(1))
    }
    expect(observedOrder(err, 0.05)).toBeCloseTo(2, 0)
    const s = run(symplectic(H, 'velocity-verlet', { h: 0.1 }), { q0: [1], p0: [0] }, 10)
    expect(s.evaluations).toBe(1 + 2 * 10)
  })
})

describe('events', () => {
  it('locates the zero crossings of an oscillator and stops at a terminal event', () => {
    const osc: Rhs = (_t, x) => stack([get(x, 1), neg(get(x, 0))])
    const sol = solveIvp(osc, [0, 10], [1, 0], {
      rtol: 1e-8,
      atol: 1e-10,
      events: [
        { name: 'x = 0', g: (_t, x) => toFlat(x)[0] },
        { name: 'v up', g: (_t, x) => toFlat(x)[1], direction: 1, terminal: true },
      ],
    })
    // x = cos t crosses zero at π/2, 3π/2; v = −sin t crosses zero upwards at π, which ends the run.
    expect(sol.events.map((e) => e.name)).toEqual(['x = 0', 'v up'])
    expect(sol.events[0].t).toBeCloseTo(Math.PI / 2, 7)
    expect(sol.events[1].t).toBeCloseTo(Math.PI, 7)
    expect(toFlat(sol.t).at(-1)!).toBeCloseTo(Math.PI, 7)
    expect(sol.stopped).toBe('done')
  })

  it('wraps a fixed-step solver as well', () => {
    const alg = withEvents(rungeKutta(decay, 'rk4', { h: 0.1, tEnd: 3 }), decay, [
      { g: (_t, x) => toFlat(x)[0] - 0.5, terminal: true },
    ])
    const s = run(alg, { x0: [1] }, 100)
    expect(s.t).toBeCloseTo(Math.LN2, 5)
    expect(s.terminated).toBe(true)
  })
})

describe('linear systems', () => {
  it.each(Object.entries(fx.expm))('expm matches scipy (%s)', (_k, { a, expm: ref }) => {
    const e = toRows(expm(a).value)
    const scale = Math.max(...ref.flat().map(Math.abs))
    ref.forEach((row, i) => row.forEach((v, j) => expect(Math.abs(e[i][j] - v) / scale).toBeLessThan(1e-12)))
  })

  it.each(Object.entries(fx.eig))('eig matches numpy (%s)', (_k, { a, real, imag }) => {
    const e = eig(a)
    expect(e.converged).toBe(true)
    toFlat(e.real).forEach((v, i) => expect(v).toBeCloseTo(real[i], 9))
    toFlat(e.imag).forEach((v, i) => expect(v).toBeCloseTo(imag[i], 9))
    // A v = λ v for every pair.
    const n = a.length
    const vr = toRows(e.vectorsReal)
    const vi = toRows(e.vectorsImag)
    for (let k = 0; k < n; k++) {
      const [lr, li] = [toFlat(e.real)[k], toFlat(e.imag)[k]]
      for (let i = 0; i < n; i++) {
        let re = 0
        let im = 0
        for (let j = 0; j < n; j++) {
          re += a[i][j] * vr[j][k]
          im += a[i][j] * vi[j][k]
        }
        expect(re).toBeCloseTo(lr * vr[i][k] - li * vi[i][k], 7)
        expect(im).toBeCloseTo(lr * vi[i][k] + li * vr[i][k], 7)
      }
    }
  })

  it('linearFlow solves x′ = Ax', () => {
    const X = toRows(
      linearFlow(
        [
          [0, 1],
          [-1, 0],
        ],
        [1, 0],
        [0, Math.PI / 2, Math.PI],
      ),
    )
    expect(X[1][0]).toBeCloseTo(0, 12)
    expect(X[1][1]).toBeCloseTo(-1, 12)
    expect(X[2][0]).toBeCloseTo(-1, 12)
  })
})

describe('stability', () => {
  it('gives the known amplification factors', () => {
    expect(amplification('euler', -2, 0)).toBeCloseTo(1, 14)
    expect(amplification('implicit-euler', -100, 0)).toBeCloseTo(1 / 101, 14)
    expect(amplification('trapezoid', 0, 3)).toBeCloseTo(1, 14)
    // RK4 on the real axis: stable up to z ≈ −2.785.
    expect(amplification('rk4', -2.78, 0)).toBeLessThan(1)
    expect(amplification('rk4', -2.8, 0)).toBeGreaterThan(1)
    // BDF2 is A-stable: stable on the imaginary axis, unstable just right of 0 on the real axis.
    expect(amplification('bdf2', 0, 5)).toBeLessThanOrEqual(1 + 1e-9)
    expect(amplification('bdf2', 0.1, 0)).toBeGreaterThan(1)
  })

  it('stabilityRegion and boundaryLocus have their shapes', () => {
    const r = stabilityRegion('euler', { nx: 5, ny: 3, real: [-2, 0], imag: [-1, 1] })
    expect(r.amplification.shape).toEqual([3, 5])
    expect(toRows(r.amplification)[1][2]).toBeCloseTo(0, 14) // z = −1
    const locus = boundaryLocus('bdf1', 100)
    toFlat(locus.real).forEach((re, i) => expect(Math.hypot(re - 1, toFlat(locus.imag)[i])).toBeCloseTo(1, 12))
  })
})

describe('trace protocol', () => {
  const algs = {
    rk4: () => rungeKutta(lotkaVolterra, 'rk4', { h: 0.05 }),
    'dormand-prince': () => dormandPrince(lotkaVolterra, { tEnd: 50 }),
    bdf3: () => bdf(lotkaVolterra, 3, { h: 0.05 }),
  }
  it.each(Object.entries(algs))('%s: seek equals run, extend equals a longer trace', (_k, make) => {
    const opts = { x0: [10, 5] }
    const long = trace(make(), opts, 30, { checkpointEvery: 5 })
    const a = seek(make(), opts, 17, long)
    const b = run(make(), opts, 17)
    expect(toFlat(a.x)).toEqual(toFlat(b.x))
    const short = trace(make(), opts, 12, { checkpointEvery: 5 })
    const extended = extend(short, make(), opts, 18)
    expect(toFlat(extended.steps.at(-1)!.x)).toEqual(toFlat(long.steps.at(-1)!.x))
    expect(extended.index).toEqual(long.index)
  })

  it('symplectic: seek equals run', () => {
    const H = { potential: (q: Tensor) => mul(0.5, sum(square(q))) }
    const long = trace(symplectic(H, 'leapfrog', { h: 0.1 }), { q0: [1], p0: [0] }, 40, { checkpointEvery: 10 })
    const a = seek(symplectic(H, 'leapfrog', { h: 0.1 }), { q0: [1], p0: [0] }, 23, long)
    const b = run(symplectic(H, 'leapfrog', { h: 0.1 }), { q0: [1], p0: [0] }, 23)
    expect(a.energyError).toBe(b.energyError)
  })
})
