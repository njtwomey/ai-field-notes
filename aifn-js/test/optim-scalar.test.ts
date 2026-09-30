import { describe, expect, it } from 'vitest'
import { minimizeScalar } from 'aifn/optim'

describe('minimizeScalar', () => {
  const quartic = (x: number) => (x - 2) ** 4 + (x - 2) ** 2 + 1

  it('Brent finds a smooth minimum from a bracket, expanding downhill past it', () => {
    const r = minimizeScalar(quartic)
    expect(r.converged).toBe(true)
    expect(r.bracketed).toBe(true)
    expect(r.x).toBeCloseTo(2, 7)
    expect(r.value).toBeCloseTo(1, 12)
    expect(minimizeScalar((x) => Math.cos(x), { bracket: [3, 3.5] }).x).toBeCloseTo(Math.PI, 7)
  })

  it('golden section converges too, in more steps', () => {
    const brent = minimizeScalar(quartic, { bounds: [0, 5] })
    const golden = minimizeScalar(quartic, { bounds: [0, 5], method: 'golden' })
    expect(golden.converged).toBe(true)
    expect(golden.x).toBeCloseTo(2, 6)
    expect(golden.steps).toBeGreaterThan(brent.steps)
  })

  it('stays within bounds and returns the endpoint for a monotone function', () => {
    const r = minimizeScalar((x) => x, { bounds: [1, 4] })
    expect(r.x).toBeGreaterThanOrEqual(1)
    expect(r.x).toBeCloseTo(1, 6)
    expect(() => minimizeScalar((x) => x, { bounds: [4, 1] })).toThrow(/bounds/)
  })

  it('reports a function that decreases without bound, and a step budget that runs out', () => {
    const r = minimizeScalar((x) => -x)
    expect(r.bracketed).toBe(false)
    expect(r.converged).toBe(false)
    expect(minimizeScalar(quartic, { bounds: [0, 5], maxSteps: 3 }).converged).toBe(false)
  })

  it('matches scipy.optimize.minimize_scalar (Brent) on a standard problem', () => {
    // scipy: minimize_scalar(lambda x: (x - 2) * x * (x + 2) ** 2) → x = 1.28077640…, f = −9.91484…
    const r = minimizeScalar((x) => (x - 2) * x * (x + 2) ** 2)
    expect(r.x).toBeCloseTo(1.2807764064044151, 7)
    expect(r.value).toBeCloseTo(-9.914949590828147, 10)
  })
})
