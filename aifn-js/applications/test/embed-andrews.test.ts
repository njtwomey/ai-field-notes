import { describe, expect, it } from 'vitest'
import { andrewsCurves } from 'aifn-applied/unsupervised/embedding/linear'
import { tensor, toRows } from 'aifn/foundation/tensor'

describe('andrewsCurves', () => {
  it('evaluates x1/√2 + x2 sin t + x3 cos t + x4 sin 2t + x5 cos 2t', () => {
    const x = tensor([
      [1, 2, 3, 4, 5],
      [0, 0, 0, 0, 1],
    ])
    const t = [0, 0.3, -1.2, Math.PI]
    const { curves } = andrewsCurves(x, t)
    const f = (r: number[], s: number) =>
      r[0] / Math.SQRT2 + r[1] * Math.sin(s) + r[2] * Math.cos(s) + r[3] * Math.sin(2 * s) + r[4] * Math.cos(2 * s)
    const rows = toRows(curves) as number[][]
    expect(curves.shape).toEqual([2, 4])
    t.forEach((s, j) => {
      expect(rows[0][j]).toBeCloseTo(f([1, 2, 3, 4, 5], s), 12)
      expect(rows[1][j]).toBeCloseTo(f([0, 0, 0, 0, 1], s), 12)
    })
  })

  it('preserves distances: ∫(f_x − f_y)² dt = π‖x − y‖² on [−π, π]', () => {
    const x = tensor([
      [0.5, -1, 2, 0.3],
      [1.5, 0.2, -0.4, 1],
    ])
    const m = 4000
    // Midpoint rule on a full period is exact for trigonometric polynomials of low degree.
    const t = Array.from({ length: m }, (_, j) => -Math.PI + (2 * Math.PI * (j + 0.5)) / m)
    const rows = toRows(andrewsCurves(x, t).curves) as number[][]
    const integral = rows[0].reduce((s, a, j) => s + (a - rows[1][j]) ** 2, 0) * ((2 * Math.PI) / m)
    const d2 = [1, -1.2, 2.4, -0.7].reduce((s, v) => s + v * v, 0)
    expect(integral).toBeCloseTo(Math.PI * d2, 9)
  })

  it('defaults to 101 points on [−π, π] and rejects a vector', () => {
    const { t, curves } = andrewsCurves(tensor([[1, 1]]))
    expect(t.length).toBe(101)
    expect(t[0]).toBeCloseTo(-Math.PI, 12)
    expect(t[100]).toBeCloseTo(Math.PI, 12)
    expect(curves.shape).toEqual([1, 101])
    expect(() => andrewsCurves(tensor([1, 2]))).toThrow(/matrix/)
  })
})
