import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import {
  addScaled,
  cholesky,
  cholSolve,
  crossprod,
  crossprodY,
  diffMatrix,
  dot,
  eye,
  gram,
  psplineRow,
  type Matrix,
} from '../_shared/terms-psplines'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const N = 60
const K = 20
const NOISE = 0.25
const GRID = toFlat(linspace(0, 1, 121))
const truth = (x: number) => 2.5 * x ** 3
// pyGAM uses 1e9; a smaller weight keeps the dense solve well conditioned and leaves violations below TOL.
const CONSTRAINT_WEIGHT = 1e6
const RIDGE = 1e-3
const TOL = 1e-4
const D1 = diffMatrix(K, 1)
const D2 = diffMatrix(K, 2)
const S = gram(D2)
const I = eye(K)

type Kind = 'none' | 'monotone' | 'convex'

/**
 * pyGAM's scheme: add a huge penalty on exactly those differences that the current coefficients violate, refit, and
 * repeat until the coefficients stop changing.
 */
function fitShape(BtB: Matrix, Bty: number[], lambda: number, kind: Kind) {
  const base = addScaled(BtB, [lambda, S], [1e-8, I])
  let beta = cholSolve(cholesky(base), Bty)
  let iterations = 0
  if (kind === 'none') return { beta, iterations }
  const D = kind === 'monotone' ? D1 : D2
  while (iterations < 50) {
    const violated = D.filter((row) => dot(row, beta) < 0)
    if (violated.length === 0) break
    const C = addScaled(gram(violated), [RIDGE / CONSTRAINT_WEIGHT, I])
    const next = cholSolve(cholesky(addScaled(base, [CONSTRAINT_WEIGHT, C])), Bty)
    const change = Math.hypot(...next.map((v, j) => v - beta[j])) / Math.hypot(...next)
    beta = next
    iterations++
    if (change < 1e-8) break
  }
  return { beta, iterations }
}

const violations = (beta: number[], D: Matrix) => D.filter((row) => dot(row, beta) < -TOL).length

export function ShapeConstrained() {
  const state = useFigureState({
    logLambda: float(0, { min: -3, max: 3, step: 0.1, label: 'log₁₀ λ', format: (v) => v.toFixed(1) }),
    seed: int(5, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => {
    const r = stream(state.seed)
    const x = Array.from({ length: N }, () => uniform(r))
    const y = x.map((xi) => truth(xi) + NOISE * normal(r))
    const B = x.map((xi) => psplineRow(xi, 0, 1, K))
    return { x, y, BtB: crossprod(B), Bty: crossprodY(B, y) }
  }, [state.seed])

  const lambda = 10 ** state.logLambda
  const fits = useMemo(
    () => (['none', 'monotone', 'convex'] as const).map((kind) => fitShape(data.BtB, data.Bty, lambda, kind)),
    [data, lambda],
  )
  const Bgrid = useMemo(() => GRID.map((g) => psplineRow(g, 0, 1, K)), [])
  const curve = (beta: number[]) => Bgrid.map((row) => dot(row, beta))

  const series = [
    { name: 'data', x: data.x, y: data.y, muted: true },
    { name: 'true f', x: GRID, y: GRID.map(truth), muted: true, dashed: true },
    { name: 'unconstrained', x: GRID, y: curve(fits[0].beta), slot: 0 },
    { name: 'monotone increasing', x: GRID, y: curve(fits[1].beta), slot: 1 },
    { name: 'convex', x: GRID, y: curve(fits[2].beta), slot: 2 },
  ] as const
  const count = (D: Matrix) => fits.map((f) => violations(f.beta, D)).join(' / ')

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [-0.8, 3.2] })
  return (
    <Figure
      title="Monotone and convex P-splines"
      state={state}
      caption={
        <>
          Sixty noisy points from the increasing, convex curve f(x) = 2.5x³ (dashed), fitted with 20 cubic B-splines and
          a second-difference penalty. The monotone fit adds a very large penalty on every first difference βⱼ₊₁ − βⱼ
          that is negative, the convex fit on every negative second difference, and each refits until the set of
          penalised differences stops changing. This is how pyGAM enforces its constraints. The readouts count the
          differences below −10⁻⁴ in the unconstrained, monotone and convex fits. With little smoothing the
          unconstrained fit dips and wiggles on the flat left part. The convex fit is not monotone: it may still fall at
          the left edge, because convexity says nothing about the sign of the slope.
        </>
      }

      readouts={
        <>
          <Readout label="Δβ < 0 (none / mono / convex)" value={count(D1)} />
          <Readout label="Δ²β < 0 (none / mono / convex)" value={count(D2)} />
          <Readout label="constrained refits (mono, convex)" value={`${fits[1].iterations}, ${fits[2].iterations}`} />
          <Readout
            label="min Δβ, monotone fit"
            value={formatNumber(Math.min(...D1.map((row) => dot(row, fits[1].beta))))}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Points {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Curve {...series[4]} />
      </Plot>
    </Figure>
  )
}
