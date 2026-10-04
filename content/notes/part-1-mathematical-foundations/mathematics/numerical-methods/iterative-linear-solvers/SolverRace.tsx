import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal as drawNormal, stream } from 'aifn/foundation/random'

const ITERS = 300
const FLOOR = 1e-12
const STEPS = Array.from({ length: ITERS + 1 }, (_, k) => k)

/** y = A x for the 5-point Laplacian on an m × m grid with zero boundary values (diagonal 4, neighbours −1). */
function laplacian(m: number, x: Float64Array, y: Float64Array) {
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      const k = i * m + j
      let v = 4 * x[k]
      if (i > 0) v -= x[k - m]
      if (i < m - 1) v -= x[k + m]
      if (j > 0) v -= x[k - 1]
      if (j < m - 1) v -= x[k + 1]
      y[k] = v
    }
  }
}

const dot = (a: Float64Array, b: Float64Array) => a.reduce((s, v, i) => s + v * b[i], 0)

/** Conjugate gradient from x = 0; returns every iterate's copy only through the callback, to keep memory flat. */
function conjugateGradient(m: number, b: Float64Array, iters: number, visit: (x: Float64Array) => void) {
  const n = m * m
  const x = new Float64Array(n)
  const r = Float64Array.from(b)
  const p = Float64Array.from(b)
  const Ap = new Float64Array(n)
  let rr = dot(r, r)
  visit(x)
  for (let k = 0; k < iters; k++) {
    if (rr === 0) {
      visit(x)
      continue
    }
    laplacian(m, p, Ap)
    const alpha = rr / dot(p, Ap)
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i]
      r[i] -= alpha * Ap[i]
    }
    const next = dot(r, r)
    const beta = next / rr
    rr = next
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i]
    visit(x)
  }
}

/** Error of Jacobi, Gauss–Seidel and conjugate gradient on the 2-D Poisson problem, in the energy norm. */
export function SolverRace() {
  const state = useFigureState({
    m: int(20, { min: 6, max: 40, step: 2, label: 'grid side m' }),
  })

  const result = useMemo(() => {
    const M = state.m
    const n = M * M
    const g = stream(3)
    const b = Float64Array.from({ length: n }, () => drawNormal(g))
    // Reference solution: CG run far past convergence (it terminates in at most n steps in exact arithmetic).
    let xStar = new Float64Array(n)
    conjugateGradient(M, b, 4 * M + 400, (x) => (xStar = Float64Array.from(x)))
    const tmp = new Float64Array(n)
    const energy = (x: Float64Array) => {
      const e = x.map((v, i) => v - xStar[i])
      laplacian(M, e, tmp)
      return Math.sqrt(Math.max(dot(e, tmp), 0))
    }
    const e0 = energy(new Float64Array(n))
    const rel = (x: Float64Array) => Math.max(FLOOR, energy(x) / e0)

    const cg: number[] = []
    conjugateGradient(M, b, ITERS, (x) => cg.push(rel(x)))

    const jac: number[] = []
    const gs: number[] = []
    let xj = new Float64Array(n)
    const xg = new Float64Array(n)
    const Ax = new Float64Array(n)
    jac.push(1)
    gs.push(1)
    for (let k = 0; k < ITERS; k++) {
      // Jacobi: every unknown updated from the previous iterate.
      laplacian(M, xj, Ax)
      xj = xj.map((v, i) => v + (b[i] - Ax[i]) / 4)
      jac.push(rel(xj))
      // Gauss–Seidel: each unknown updated in place, using the newest neighbours.
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
          const q = i * M + j
          let s = b[q]
          if (i > 0) s += xg[q - M]
          if (i < M - 1) s += xg[q + M]
          if (j > 0) s += xg[q - 1]
          if (j < M - 1) s += xg[q + 1]
          xg[q] = s / 4
        }
      }
      gs.push(rel(xg))
    }

    // Extreme eigenvalues of the 5-point Laplacian give κ and the CG bound 2((√κ − 1)/(√κ + 1))^k.
    const s = Math.sin(Math.PI / (2 * (M + 1))) ** 2
    const kappa = (1 - s) / s
    const q = (Math.sqrt(kappa) - 1) / (Math.sqrt(kappa) + 1)
    const bound = STEPS.map((k) => Math.max(FLOOR, Math.min(1, 2 * q ** k)))
    const cgIters = cg.findIndex((v) => v < 1e-6)
    return { jac, gs, cg, bound, kappa, n, cgIters }
  }, [state.m])

  const series = [
    { name: 'Jacobi', x: STEPS, y: result.jac, slot: 0 },
    { name: 'Gauss–Seidel', x: STEPS, y: result.gs, slot: 1 },
    { name: 'conjugate gradient', x: STEPS, y: result.cg, slot: 2 },
    { name: 'CG bound 2((√κ−1)/(√κ+1))ᵏ', x: STEPS, y: result.bound, slot: 2, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'iteration k', range: [0, ITERS] })
  const yAxis = useAxis({ label: '‖x − xₖ‖_A / ‖x‖_A', range: [FLOOR, 2], log: true })
  return (
    <Figure
      title="Stationary iterations against conjugate gradient"
      state={state}
      caption="The 2-D Poisson equation on an m × m grid (n = m² unknowns), solved from x = 0. The curves show the relative error in the energy norm. Jacobi and Gauss–Seidel shrink the error by a factor near 1 − c/κ per step and stall as the grid grows; conjugate gradient shrinks it at the √κ rate of its bound, and faster in practice."

      readouts={
        <>
          <Readout label="unknowns n" value={formatNumber(result.n)} />
          <Readout label="κ(A)" value={formatNumber(result.kappa)} />
          <Readout label="CG steps to 10⁻⁶" value={result.cgIters < 0 ? `> ${ITERS}` : String(result.cgIters)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
      </Plot>
    </Figure>
  )
}
