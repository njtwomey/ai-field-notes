import { describe, expect, it } from 'vitest'
import {
  bisection,
  brent,
  broyden,
  continuation,
  findRoot,
  fixedPoint,
  newtonHomotopy,
  newtonRoot,
  newtonSystem,
  polynomialRoots,
  polynomialValue,
  regulaFalsi,
  secant,
  solveSystem,
  type SystemWithJacobian,
} from 'aifn/solve'
import { tensor, toFlat } from 'aifn/tensor'
import { extend, run, seek, trace } from 'aifn/trace'

const cubic = (x: number) => x * x * x - 2 * x - 5 // Newton's own example; root 2.0945514815423265
const ROOT = 2.0945514815423265

describe('scalar roots', () => {
  it('every method finds the root of x³ − 2x − 5', () => {
    const bracket = { lo: 2, hi: 3 }
    expect(run(bisection(cubic), bracket, 100).x).toBeCloseTo(ROOT, 11)
    expect(run(regulaFalsi(cubic), bracket, 100).x).toBeCloseTo(ROOT, 12)
    expect(run(brent(cubic), bracket, 100).x).toBeCloseTo(ROOT, 12)
    expect(run(secant(cubic), { x0: 2, x1: 3 }, 100).x).toBeCloseTo(ROOT, 12)
    const d = (x: number) => ({ value: cubic(x), derivative: 3 * x * x - 2 })
    expect(run(newtonRoot(d), { x0: 2 }, 100).x).toBeCloseTo(ROOT, 12)
    expect(run(newtonRoot(d, { damped: true }), { x0: 0.5 }, 100).x).toBeCloseTo(ROOT, 12)
  })
  it('Brent uses far fewer evaluations than bisection and reports its step kinds', () => {
    const b = trace(brent(cubic), { lo: 0, hi: 5 }, 100)
    const s = b.steps.at(-1)!
    expect(s.converged).toBe(true)
    expect(s.evaluations).toBeLessThan(15)
    expect(new Set(b.steps.map((q) => q.method)).has('inverse-quadratic')).toBe(true)
  })
  it('reports a bracket without a sign change', () => {
    const r = findRoot((x) => x * x + 1, [-1, 1])
    expect(r.converged).toBe(false)
    expect(r.failure).toBe('no sign change')
  })
  it('protocol: seek equals run; extend equals a longer trace', () => {
    const alg = bisection(cubic)
    const record = { x: (s: { x: number }) => s.x }
    const a = trace(alg, { lo: 2, hi: 3 }, 20, { record })
    expect(seek(alg, { lo: 2, hi: 3 }, 9).x).toBe(run(alg, { lo: 2, hi: 3 }, 9).x)
    expect(toFlat(extend(trace(alg, { lo: 2, hi: 3 }, 8, { record }), alg, { lo: 2, hi: 3 }, 12).series.x)).toEqual(
      toFlat(a.series.x),
    )
  })
})

describe('systems', () => {
  // x² + y² = 4, xy = 1.
  const F: SystemWithJacobian = (v) => {
    const [x, y] = toFlat(v)
    return {
      value: [x * x + y * y - 4, x * y - 1],
      jacobian: [
        [2 * x, 2 * y],
        [y, x],
      ],
    }
  }
  const check = (x: number[]) => {
    expect(Math.abs(x[0] ** 2 + x[1] ** 2 - 4)).toBeLessThan(1e-9)
    expect(Math.abs(x[0] * x[1] - 1)).toBeLessThan(1e-9)
  }
  it('Newton, damped Newton and Broyden solve it', () => {
    check(toFlat(run(newtonSystem(F), { x0: [2, 0.3] }, 50).x))
    check(toFlat(solveSystem(F, [2, 0.3]).x))
    const values = (v: Parameters<SystemWithJacobian>[0]) => F(v).value
    const b = solveSystem(values, [2, 0.3], { method: 'broyden' })
    expect(b.converged).toBe(true)
    check(toFlat(b.x))
    expect(run(broyden(values), { x0: [2, 0.3] }, 100).converged).toBe(true)
  })
  it('fixed-point iteration on cos converges with contraction ≈ |sin x*|', () => {
    const s = run(
      fixedPoint((x) => toFlat(x).map(Math.cos)),
      { x0: [1] },
      200,
    )
    expect(s.converged).toBe(true)
    expect(toFlat(s.x)[0]).toBeCloseTo(0.7390851332151607, 11)
    const t = run(
      fixedPoint((x) => toFlat(x).map(Math.cos), { ftol: 0 }),
      { x0: [1] },
      30,
    )
    expect(t.contraction).toBeCloseTo(Math.sin(0.7390851332151607), 3)
    const d = run(
      fixedPoint((x) => toFlat(x).map((v) => 2 * v + 1), { patience: 5 }),
      { x0: [1] },
      100,
    )
    expect(d.failure).toBe('diverging')
  })
  it('continuation follows the Newton homotopy to a root', () => {
    const s = run(continuation(newtonHomotopy(F, [3, 0.5])), { x0: [3, 0.5] }, 200)
    expect(s.converged).toBe(true)
    check(toFlat(s.x))
  })
})

describe('polynomial roots', () => {
  it('finds real and complex roots', () => {
    // (x − 1)(x − 2)(x − 3)(x² + 1) = x⁵ − 6x⁴ + 12x³ − 12x² + 11x − 6
    const r = polynomialRoots([1, -6, 12, -12, 11, -6])
    expect(r.converged).toBe(true)
    const re = toFlat(r.real)
    const im = toFlat(r.imag)
    ;[3, 2, 1, 0, 0].forEach((v, i) => expect(re[i]).toBeCloseTo(v, 10))
    ;[0, 0, 0, 1, -1].forEach((v, i) => expect(im[i]).toBeCloseTo(v, 10))
    expect(toFlat(polynomialRoots([0, 2, -2, 0]).real)).toEqual([1, 0])
    expect(polynomialValue([1, -6, 12, -12, 11, -6], 3)).toBeCloseTo(0, 12)
    expect(toFlat(polynomialValue([1, 0, -1], tensor([2, 3])))).toEqual([3, 8])
  })
  it('handles a degree-20 Wilkinson-type polynomial to modest accuracy', () => {
    let c = [1]
    for (let k = 1; k <= 10; k++) c = [...c, 0].map((v, i) => v - k * (i > 0 ? c[i - 1] : 0))
    const re = toFlat(polynomialRoots(c).real)
    re.forEach((v, i) => expect(v).toBeCloseTo(10 - i, 5))
  })
})
