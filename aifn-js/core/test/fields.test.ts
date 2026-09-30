import { describe, expect, it } from 'vitest'
import {
  classifyLinear,
  contour,
  curl,
  directionField,
  divergence,
  fixedPoints,
  flowMap,
  gradientField,
  hamiltonianField,
  invariantManifolds,
  jacobianAt,
  linearise,
  nullclines,
  sampleField,
  streamline,
  transportDensity,
  transportDensityFrames,
  pushForwardFrames,
  type VectorField,
} from 'aifn/dynamics/fields'
import { limitCycle, lyapunovCheck, poincareSection } from 'aifn-applied/dynamics/nonlinear'
import {
  add,
  cos,
  fromRows,
  get,
  mul,
  neg,
  sin,
  square,
  stack,
  sub,
  sum,
  tensor,
  toFlat,
  toRows,
  type Tensor,
} from 'aifn/foundation/tensor'

const pendulum: VectorField = (x) => stack([get(x, 1), neg(sin(get(x, 0)))])
const vanDerPol: VectorField = (x) => {
  const [a, b] = [get(x, 0), get(x, 1)]
  return stack([b, sub(mul(sub(1, mul(a, a)), b), a)])
}
const rotation: VectorField = (x) => stack([neg(get(x, 1)), get(x, 0)])

describe('calculus', () => {
  it('computes Jacobian, divergence and curl', () => {
    const f: VectorField = (x) => stack([mul(get(x, 0), get(x, 1)), add(square(get(x, 0)), mul(3, get(x, 1)))])
    expect(toRows(jacobianAt(f, [2, 5]))).toEqual([
      [5, 2],
      [4, 3],
    ])
    expect(divergence(f, [2, 5])).toBeCloseTo(8, 12)
    expect(curl(f, [2, 5])).toBeCloseTo(4 - 2, 12)
    expect(curl(rotation, [0.3, 0.4])).toBeCloseTo(2, 12)
    expect(divergence(rotation, [0.3, 0.4])).toBeCloseTo(0, 12)
  })

  it('gradient fields are curl-free and Hamiltonian fields divergence-free', () => {
    const V = (x: Tensor) => add(mul(0.5, sum(square(x))), mul(get(x, 0), get(x, 1)))
    const g = gradientField(V)
    expect(toFlat(g(tensor([1, 2])) as Tensor)).toEqual([-3, -3])
    expect(curl(g, [0.7, -0.2])).toBeCloseTo(0, 12)
    const H = hamiltonianField((x: Tensor) => sub(mul(0.5, square(get(x, 1))), cos(get(x, 0))))
    expect(divergence(H, [0.4, 1.1])).toBeCloseTo(0, 12)
    const hv = toFlat(H(tensor([0.4, 1.1])) as Tensor)
    expect(hv[0]).toBeCloseTo(1.1, 14)
    expect(hv[1]).toBeCloseTo(-Math.sin(0.4), 14)
  })
})

describe('grids and level sets', () => {
  it('samples a field and draws a direction field', () => {
    const s = sampleField(rotation, { x: [-1, 1], y: [-1, 1], nx: 3, ny: 3 })
    expect(toRows(s.u)[0]).toEqual([1, 1, 1]) // u = −y at y = −1
    const d = directionField(rotation, { x: [-1, 1], y: [-1, 1], nx: 4, ny: 4 })
    expect(d.start.shape).toEqual([16, 2])
    const len = toRows(d.end).map((e, k) => Math.hypot(e[0] - toRows(d.start)[k][0], e[1] - toRows(d.start)[k][1]))
    len.forEach((l) => expect(l).toBeCloseTo(0.7 * 0.5, 12))
  })

  it('contours a circle and finds nullclines', () => {
    const n = 41
    const xs = Array.from({ length: n }, (_, i) => -2 + (4 * i) / (n - 1))
    const z = xs.map((y) => xs.map((x) => x * x + y * y))
    const c = contour(xs, xs, fromRows(z), 1)
    toRows(c.start).forEach(([x, y]) => expect(Math.hypot(x, y)).toBeCloseTo(1, 1))
    const [nx, ny] = nullclines(pendulum, { x: [-4, 4], y: [-2, 2], nx: 41, ny: 41 })
    // ẋ = y vanishes on y = 0; ẏ = −sin x on x = 0, ±π.
    toRows(nx.start).forEach(([, y]) => expect(Math.abs(y)).toBeLessThan(1e-12))
    toRows(ny.start).forEach(([x]) => expect(Math.min(Math.abs(x), Math.abs(Math.abs(x) - Math.PI))).toBeLessThan(0.01))
  })
})

describe('fixed points', () => {
  it('classifies canonical linear systems', () => {
    expect(
      classifyLinear([
        [-1, 0],
        [0, -2],
      ]).kind,
    ).toBe('stable node')
    expect(
      classifyLinear([
        [1, 0],
        [0, -2],
      ]).kind,
    ).toBe('saddle')
    expect(
      classifyLinear([
        [-0.1, 1],
        [-1, -0.1],
      ]).kind,
    ).toBe('stable spiral')
    expect(
      classifyLinear([
        [0.1, 1],
        [-1, 0.1],
      ]).kind,
    ).toBe('unstable spiral')
    expect(
      classifyLinear([
        [0, 1],
        [-1, 0],
      ]).kind,
    ).toBe('centre')
    expect(
      classifyLinear([
        [-1, 0],
        [0, -1],
      ]).kind,
    ).toBe('star')
    expect(
      classifyLinear([
        [-1, 1],
        [0, -1],
      ]).kind,
    ).toBe('stable degenerate node')
    expect(
      classifyLinear([
        [0, 0],
        [0, -1],
      ]).kind,
    ).toBe('non-hyperbolic')
    const three = classifyLinear([
      [-1, 0, 0],
      [0, -2, 1],
      [0, -1, -2],
    ])
    expect(three.kind).toBe('stable spiral')
    expect(three.determinant).toBeCloseTo(-5, 10)
  })

  it('finds the pendulum’s equilibria', () => {
    const fps = fixedPoints(pendulum, [
      [-4, 4],
      [-2, 2],
    ])
    expect(fps.map((p) => toFlat(p.point)[0])).toEqual([
      expect.closeTo(-Math.PI, 10),
      expect.closeTo(0, 10),
      expect.closeTo(Math.PI, 10),
    ])
    expect(fps.map((p) => p.kind)).toEqual(['saddle', 'centre', 'saddle'])
  })

  it('traces a saddle’s manifolds along its eigenvectors', () => {
    const f: VectorField = (x) => stack([get(x, 0), neg(get(x, 1))])
    const m = invariantManifolds(f, linearise(f, [0, 0]), {
      t: 15,
      bounds: [
        [-2, 2],
        [-2, 2],
      ],
    })
    expect(m.unstable).toHaveLength(2)
    expect(m.stable).toHaveLength(2)
    m.unstable.forEach((c) => toRows(c).forEach(([, y]) => expect(Math.abs(y)).toBeLessThan(1e-12)))
    m.stable.forEach((c) => toRows(c).forEach(([x]) => expect(Math.abs(x)).toBeLessThan(1e-12)))
    // Stable branches run from the fixed point outwards.
    expect(toRows(m.stable[0])[0]).toEqual([0, 0])
    expect(Math.abs(toRows(m.stable[0]).at(-1)![1])).toBeGreaterThan(1)
  })

  it('checks a Lyapunov function', () => {
    const damped: VectorField = (x) => stack([get(x, 1), sub(neg(get(x, 0)), mul(0.5, get(x, 1)))])
    const V = (x: Tensor) => sum(square(x))
    const r = lyapunovCheck(V, damped, [0, 0], { x: [-1, 1], y: [-1, 1], nx: 11, ny: 11 })
    expect(r.positiveDefinite).toBe(true)
    expect(r.nonIncreasing).toBe(true)
    expect(r.decreasing).toBe(false) // V̇ = −y² vanishes on the x axis
    const bad = lyapunovCheck(V, (x) => neg(damped(x) as Tensor), [0, 0], { x: [-1, 1], y: [-1, 1], nx: 11, ny: 11 })
    expect(bad.nonIncreasing).toBe(false)
  })
})

describe('flows', () => {
  it('flow maps and streamlines follow the rotation', () => {
    const x = toFlat(flowMap(rotation, [1, 0], Math.PI / 2))
    expect(x[0]).toBeCloseTo(0, 7)
    expect(x[1]).toBeCloseTo(1, 7)
    const s = streamline(rotation, [1, 0], { t: 2 * Math.PI, steps: 400 })
    toRows(s).forEach(([a, b]) => expect(Math.hypot(a, b)).toBeCloseTo(1, 6))
    const bounded = streamline((p) => p, [0.1, 0], {
      direction: 'forward',
      bounds: [
        [-1, 1],
        [-1, 1],
      ],
    })
    expect(toRows(bounded).at(-1)![0]).toBeGreaterThan(1)
    expect(toRows(bounded).at(-2)![0]).toBeLessThanOrEqual(1)
  })

  it('finds the Van der Pol limit cycle and its period', () => {
    const section = { point: [0, 0], normal: [1, 0] }
    const cycle = limitCycle(vanDerPol, [0.5, 0.5], section)
    expect(cycle.converged).toBe(true)
    expect(cycle.period).toBeCloseTo(6.6632868593, 5) // μ = 1
    expect(Math.abs(cycle.multiplier)).toBeLessThan(1)
    const r = poincareSection(vanDerPol, [0.5, 0.5], section, { crossings: 5 })
    expect(r.points.shape).toEqual([5, 2])
    toRows(r.points).forEach(([x]) => expect(Math.abs(x)).toBeLessThan(1e-7))
  })

  it('transports a density by Liouville’s equation', () => {
    const a = 0.7
    const f: VectorField = (x) => mul(a, x)
    const rho0 = (x: Tensor) => Math.exp(-0.5 * toFlat(x)[0] ** 2) / Math.sqrt(2 * Math.PI)
    const pts = [[-1], [0], [0.5], [2]]
    const t = 1.3
    const rho = toFlat(transportDensity(f, rho0, pts, t))
    pts.forEach(([x], k) => {
      const exact = rho0(tensor([x * Math.exp(-a * t)])) * Math.exp(-a * t)
      expect(rho[k]).toBeCloseTo(exact, 8)
    })
  })

  it('transports a density at every frame, matching a single-time transport', () => {
    const a = 0.7
    const f: VectorField = (x) => mul(a, x)
    const rho0 = (x: Tensor) => Math.exp(-0.5 * toFlat(x)[0] ** 2) / Math.sqrt(2 * Math.PI)
    const pts = [[-1], [0.5], [2]]
    const frames = toRows(transportDensityFrames(f, rho0, pts, 1.2, { frames: 4, steps: 10, divergence: () => a }))
    expect(frames.length).toBe(5)
    frames.forEach((row, j) => {
      const t = (1.2 * j) / 4
      pts.forEach(([x], k) => expect(row[k]).toBeCloseTo(rho0(tensor([x * Math.exp(-a * t)])) * Math.exp(-a * t), 7))
    })
  })

  it('pushes samples forward at every frame with their log-volume change', () => {
    const f: VectorField = (x) => mul(-0.5, x)
    const r = pushForwardFrames(f, [[1, 2]], 2, { frames: 2, steps: 20, divergence: () => -1 })
    const x = toFlat(r.x)
    expect(x[4]).toBeCloseTo(Math.exp(-1), 8)
    expect(x[5]).toBeCloseTo(2 * Math.exp(-1), 8)
    expect(toFlat(r.logVolume)).toEqual([0, -1, -2].map((v) => expect.closeTo(v, 12)))
  })
})
