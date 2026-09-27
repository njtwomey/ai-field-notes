/** Small ODE helpers shared by the differential-equations widgets. Light enough to run on every slider drag. */
import type { Segment } from '@/components/viz'

export type Vec = number[]
/** Right-hand side of ẋ = f(t, x). */
export type Rhs = (t: number, x: Vec) => Vec

const axpy = (a: number, x: Vec, y: Vec): Vec => y.map((v, i) => v + a * x[i])

/** One classical fourth-order Runge–Kutta step. */
export function rk4Step(f: Rhs, t: number, x: Vec, h: number): Vec {
  const k1 = f(t, x)
  const k2 = f(t + h / 2, axpy(h / 2, k1, x))
  const k3 = f(t + h / 2, axpy(h / 2, k2, x))
  const k4 = f(t + h, axpy(h, k3, x))
  return x.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]))
}

/**
 * Integrate from (t0, x0) to t1 with `n` RK4 steps (t1 may be earlier than t0). Returns the times and states,
 * stopping early if the state leaves the box |x_i| ≤ `bound`, so blow-up does not produce Infinity.
 */
export function integrate(f: Rhs, x0: Vec, t0: number, t1: number, n: number, bound = 1e3) {
  const h = (t1 - t0) / n
  const ts = [t0]
  const xs = [x0]
  let x = x0
  for (let k = 0; k < n; k++) {
    x = rk4Step(f, t0 + k * h, x, h)
    if (!x.every((v) => Number.isFinite(v) && Math.abs(v) <= bound)) break
    ts.push(t0 + (k + 1) * h)
    xs.push(x)
  }
  return { ts, xs }
}

/**
 * A direction field: at each point of an nx × ny grid, a short segment along f with a small arrowhead. Lengths are
 * normalised so the field shows direction only. `scale` is the segment length as a fraction of the grid spacing.
 * For a slope field of ẋ = g(t, x), pass f = (t, x) ↦ [1, g(t, x)] and `arrows: false`.
 */
export function directionField(
  f: (x: number, y: number) => [number, number],
  [x0, x1]: [number, number],
  [y0, y1]: [number, number],
  nx: number,
  ny: number,
  { scale = 0.7, arrows = true }: { scale?: number; arrows?: boolean } = {},
): Segment[] {
  const dx = (x1 - x0) / nx
  const dy = (y1 - y0) / ny
  const out: Segment[] = []
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const cx = x0 + (i + 0.5) * dx
      const cy = y0 + (j + 0.5) * dy
      const [u, v] = f(cx, cy)
      // Normalise in grid units so the segments look the same length whatever the axis ranges.
      const gu = u / dx
      const gv = v / dy
      const m = Math.hypot(gu, gv)
      if (!(m > 1e-12)) continue
      const hx = ((gu / m) * dx * scale) / 2
      const hy = ((gv / m) * dy * scale) / 2
      const tip: [number, number] = [cx + hx, cy + hy]
      out.push({ from: [cx - hx, cy - hy], to: tip })
      if (arrows) {
        // Two short barbs at the tip, turned ±150° from the direction, in grid units.
        for (const s of [1, -1]) {
          const a = (s * (5 * Math.PI)) / 6
          const bu = (Math.cos(a) * gu - Math.sin(a) * gv) / m
          const bv = (Math.sin(a) * gu + Math.cos(a) * gv) / m
          out.push({ from: tip, to: [tip[0] + bu * dx * scale * 0.3, tip[1] + bv * dy * scale * 0.3] })
        }
      }
    }
  }
  return out
}

export type Mat2 = [[number, number], [number, number]]

/** Eigenvalues of a 2×2 matrix as (re, im) pairs, and the name of the phase portrait they give. */
export function classify2(m: Mat2) {
  const [[a, b], [c, d]] = m
  const tr = a + d
  const det = a * d - b * c
  const disc = (tr * tr) / 4 - det
  const eps = 1e-9
  const values: [number, number][] =
    disc >= 0
      ? [
          [tr / 2 + Math.sqrt(disc), 0],
          [tr / 2 - Math.sqrt(disc), 0],
        ]
      : [
          [tr / 2, Math.sqrt(-disc)],
          [tr / 2, -Math.sqrt(-disc)],
        ]
  let kind: string
  if (Math.abs(det) < eps) kind = 'degenerate (a zero eigenvalue)'
  else if (det < 0) kind = 'saddle'
  else if (disc < 0) kind = Math.abs(tr) < eps ? 'centre' : tr < 0 ? 'stable spiral' : 'unstable spiral'
  else if (Math.abs(tr) < eps) kind = 'centre'
  else kind = tr < 0 ? 'stable node' : 'unstable node'
  return { tr, det, values, kind }
}

/** Format an eigenvalue pair (re, im) as "a ± bi" or "a". */
export function formatEig([re, im]: [number, number], fmt: (v: number) => string) {
  return Math.abs(im) < 1e-9 ? fmt(re) : `${fmt(re)} ${im < 0 ? '−' : '+'} ${fmt(Math.abs(im))}i`
}

/** Histogram of samples on fixed bins, scaled to a density so it is comparable with a pdf. */
export function histogramDensity(samples: ArrayLike<number>, lo: number, hi: number, bins: number) {
  const w = (hi - lo) / bins
  const counts = new Array<number>(bins).fill(0)
  for (let i = 0; i < samples.length; i++) {
    const k = Math.floor((samples[i] - lo) / w)
    if (k >= 0 && k < bins) counts[k]++
  }
  const x = counts.map((_, k) => lo + (k + 0.5) * w)
  const y = counts.map((c) => c / (samples.length * w))
  return { x, y }
}

/**
 * e^{At} for a 2×2 matrix in closed form. With s = tr/2 and N = A − sI, N² = (s² − det)·I, so the power series of
 * e^{Nt} sums to cosh/sinh (real eigenvalues), cos/sin (complex) or I + Nt (repeated).
 */
export function expm2(m: Mat2, t: number): Mat2 {
  const [[a, b], [c, d]] = m
  const s = (a + d) / 2
  const disc = s * s - (a * d - b * c)
  const q = Math.sqrt(Math.abs(disc))
  let ch: number
  let sh: number
  if (q * Math.abs(t) < 1e-8) {
    ch = 1
    sh = t
  } else if (disc > 0) {
    ch = Math.cosh(q * t)
    sh = Math.sinh(q * t) / q
  } else {
    ch = Math.cos(q * t)
    sh = Math.sin(q * t) / q
  }
  const e = Math.exp(s * t)
  return [
    [e * (ch + sh * (a - s)), e * sh * b],
    [e * sh * c, e * (ch + sh * (d - s))],
  ]
}
