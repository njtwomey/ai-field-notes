import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Raster,
  Readout,
  setting,
  useAxis,
  useFigureState,
  type AxisModel,
} from 'aifn-render'
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
  kron,
  psplineRow,
  rowKron,
  traceSolve,
} from '../_shared/terms-psplines'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const mean = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

const N = 400
const NOISE = 0.3
const G = toFlat(linspace(0, 1, 31))
// Wiggly in x, linear in z, with an interaction: the x-effect grows with z.
const truth = (x: number, z: number) => Math.sin(2 * Math.PI * x) * (0.5 + z) + z

/** Functional ANOVA of a surface on the plotting grid: overall mean, two main effects and the interaction. */
function anova(F: number[][]) {
  const m = mean(F.flat())
  // F[i][j] is the value at (x = G[j], z = G[i]).
  const fx = G.map((_, j) => mean(F.map((row) => row[j])) - m)
  const fz = F.map((row) => mean(row) - m)
  const fxz = F.map((row, i) => row.map((v, j) => v - m - fx[j] - fz[i]))
  return { fx, fz, fxz }
}

export function TensorSmooth() {
  const state = useFigureState({
    logL1: float(-1.5, { min: -3, max: 4, step: 0.1, label: 'log₁₀ λ₁ (x)', format: (v) => v.toFixed(1) }),
    logL2: float(1.5, {
      min: -3,
      max: 4,
      step: 0.1,
      label: 'log₁₀ λ₂ (z)',
      format: (v) => v.toFixed(1),
      when: (v) => !v.tie,
    }),
    k1: int(10, { min: 4, max: 12, step: 1, label: 'k₁ (x basis)' }),
    k2: int(10, { min: 4, max: 12, step: 1, label: 'k₂ (z basis)' }),
    tie: setting(false, 'tie λ₂ = λ₁'),
    seed: int(3, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => {
    const r = stream(state.seed)
    const x = Array.from({ length: N }, () => uniform(r))
    const z = Array.from({ length: N }, () => uniform(r))
    const y = x.map((xi, i) => truth(xi, z[i]) + NOISE * normal(r))
    return { x, z, y }
  }, [state.seed])

  const model = useMemo(() => {
    const X = data.x.map((xi, i) => rowKron(psplineRow(xi, 0, 1, state.k1), psplineRow(data.z[i], 0, 1, state.k2)))
    return {
      X,
      XtX: crossprod(X),
      Xty: crossprodY(X, data.y),
      // Penalties on the row-major coefficient matrix: S₁ ⊗ I smooths along x, I ⊗ S₂ along z.
      S1: kron(gram(diffMatrix(state.k1, 2)), eye(state.k2)),
      S2: kron(eye(state.k1), gram(diffMatrix(state.k2, 2))),
    }
  }, [data, state.k1, state.k2])

  const l1 = 10 ** state.logL1
  const l2 = 10 ** (state.tie ? state.logL1 : state.logL2)

  const fit = useMemo(() => {
    const L = cholesky(addScaled(model.XtX, [l1, model.S1], [l2, model.S2]))
    const beta = cholSolve(L, model.Xty)
    const edf = traceSolve(L, model.XtX)
    const rss = model.X.reduce((s, row, i) => s + (data.y[i] - dot(row, beta)) ** 2, 0)
    const Ax = G.map((g) => psplineRow(g, 0, 1, state.k1))
    const Az = G.map((g) => psplineRow(g, 0, 1, state.k2))
    const F = Az.map((bz) => Ax.map((ax) => dot(rowKron(ax, bz), beta)))
    return { F, edf, gcv: (N * rss) / (N - edf) ** 2 }
  }, [model, data, l1, l2, state.k1, state.k2])

  const trueF = useMemo(() => G.map((z) => G.map((x) => truth(x, z))), [])
  const rmse = Math.sqrt(mean(fit.F.flat().map((v, i) => (v - trueF.flat()[i]) ** 2)))
  const fitA = anova(fit.F)
  const trueA = useMemo(() => anova(trueF), [trueF])

  const mainAxes: Record<'x' | 'z', [AxisModel, AxisModel]> = {
    x: [useAxis({ label: 'x', range: [0, 1] }), useAxis({ label: 'f₁(x)', range: [-1.3, 1.3] })],
    z: [useAxis({ label: 'z', range: [0, 1] }), useAxis({ label: 'f₂(z)', range: [-1.3, 1.3] })],
  }
  const main = (which: 'x' | 'z') => (
    <Plot x={mainAxes[which][0]} y={mainAxes[which][1]} height={220}>
      <Curve name="true" x={G} y={which === 'x' ? trueA.fx : trueA.fz} slot={2} dashed />
      <Curve name="fitted" x={G} y={which === 'x' ? fitA.fx : fitA.fz} slot={1} />
    </Plot>
  )

  const xAxis = useAxis({ label: 'x' })
  const yAxis = useAxis({ label: 'z' })
  const xAxis2 = useAxis({ label: 'x' })
  const yAxis2 = useAxis({ label: 'z' })
  return (
    <Figure
      title="A tensor-product smooth with one λ per margin"
      state={state}
      caption={
        <>
          Four hundred noisy points from f(x, z) = sin(2πx)(½ + z) + z, fitted with a tensor product of two cubic
          P-spline bases of sizes k₁ and k₂. The penalty is λ₁ βᵀ(S₁ ⊗ I)β + λ₂ βᵀ(I ⊗ S₂)β, with second-difference
          matrices S₁ and S₂. The truth is wiggly in x and linear in z, so the best fit has a small λ₁ and a large λ₂.
          Tie the two parameters together and no single value does as well: the best tied fit spends about twice the
          effective degrees of freedom and still has a larger error against the true surface. The other three panels
          split the fitted surface (first panel) into its functional ANOVA parts by averaging over the plotted grid: the
          main effects f₁ and f₂ and the interaction f₁₂, each against its true counterpart.
        </>
      }
      readouts={
        <>
          <Readout label="coefficients k₁k₂" value={String(state.k1 * state.k2)} />
          <Readout label="effective df" value={formatNumber(fit.edf)} />
          <Readout label="GCV" value={formatNumber(fit.gcv)} />
          <Readout label="RMSE vs truth" value={formatNumber(rmse)} />
        </>
      }
    >
      <div className="grid gap-2 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300} ariaLabel={'Fitted surface'}>
          <Raster x={G} y={G} z={fit.F} scale={'diverging'} range={[-2.5, 2.5]} valueLabel={'f(x, z)'} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300} ariaLabel={'Fitted interaction'}>
          <Raster x={G} y={G} z={fitA.fxz} scale={'diverging'} range={[-0.8, 0.8]} valueLabel={'f₁₂(x, z)'} />
        </Plot>
        {main('x')}
        {main('z')}
      </div>
    </Figure>
  )
}
