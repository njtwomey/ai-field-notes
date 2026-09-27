import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
} from '@/components/viz'
import { linspace, mean, rng } from '@/lib/math'
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

const N = 400
const NOISE = 0.3
const G = linspace(0, 1, 31)
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
  const [k1, setK1] = useState(10)
  const [k2, setK2] = useState(10)
  const [logL1, setLogL1] = useState(-1.5)
  const [logL2, setLogL2] = useState(1.5)
  const [tie, setTie] = useState(false)
  const [seed, setSeed] = useState(3)

  const data = useMemo(() => {
    const r = rng(seed)
    const x = Array.from({ length: N }, () => r.uniform())
    const z = Array.from({ length: N }, () => r.uniform())
    const y = x.map((xi, i) => truth(xi, z[i]) + NOISE * r.normal())
    return { x, z, y }
  }, [seed])

  const model = useMemo(() => {
    const X = data.x.map((xi, i) => rowKron(psplineRow(xi, 0, 1, k1), psplineRow(data.z[i], 0, 1, k2)))
    return {
      X,
      XtX: crossprod(X),
      Xty: crossprodY(X, data.y),
      // Penalties on the row-major coefficient matrix: S₁ ⊗ I smooths along x, I ⊗ S₂ along z.
      S1: kron(gram(diffMatrix(k1, 2)), eye(k2)),
      S2: kron(eye(k1), gram(diffMatrix(k2, 2))),
    }
  }, [data, k1, k2])

  const l1 = 10 ** logL1
  const l2 = 10 ** (tie ? logL1 : logL2)

  const fit = useMemo(() => {
    const L = cholesky(addScaled(model.XtX, [l1, model.S1], [l2, model.S2]))
    const beta = cholSolve(L, model.Xty)
    const edf = traceSolve(L, model.XtX)
    const rss = model.X.reduce((s, row, i) => s + (data.y[i] - dot(row, beta)) ** 2, 0)
    const Ax = G.map((g) => psplineRow(g, 0, 1, k1))
    const Az = G.map((g) => psplineRow(g, 0, 1, k2))
    const F = Az.map((bz) => Ax.map((ax) => dot(rowKron(ax, bz), beta)))
    return { F, edf, gcv: (N * rss) / (N - edf) ** 2 }
  }, [model, data, l1, l2, k1, k2])

  const trueF = useMemo(() => G.map((z) => G.map((x) => truth(x, z))), [])
  const rmse = Math.sqrt(mean(fit.F.flat().map((v, i) => (v - trueF.flat()[i]) ** 2)))
  const fitA = anova(fit.F)
  const trueA = useMemo(() => anova(trueF), [trueF])

  const main = (which: 'x' | 'z') => {
    const series: XYSeries[] = [
      { name: 'true', type: 'line', x: G, y: which === 'x' ? trueA.fx : trueA.fz, slot: 2, dashed: true },
      { name: 'fitted', type: 'line', x: G, y: which === 'x' ? fitA.fx : fitA.fz, slot: 1 },
    ]
    return (
      <XYChart
        series={series}
        xRange={[0, 1]}
        yRange={[-1.3, 1.3]}
        xLabel={which}
        yLabel={which === 'x' ? 'f₁(x)' : 'f₂(z)'}
        height={220}
      />
    )
  }

  return (
    <Interactive
      title="A tensor-product smooth with one λ per margin"
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
      controls={
        <>
          <ParamSlider
            label="log₁₀ λ₁ (x)"
            value={logL1}
            onChange={setLogL1}
            min={-3}
            max={4}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          <ParamSlider
            label="log₁₀ λ₂ (z)"
            value={tie ? logL1 : logL2}
            onChange={setLogL2}
            min={-3}
            max={4}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          <ParamSlider label="k₁ (x basis)" value={k1} onChange={setK1} min={4} max={12} step={1} withArrows />
          <ParamSlider label="k₂ (z basis)" value={k2} onChange={setK2} min={4} max={12} step={1} withArrows />
          <ParamSwitch label="tie λ₂ = λ₁" checked={tie} onChange={setTie} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="coefficients k₁k₂" value={String(k1 * k2)} />
          <Readout label="effective df" value={formatNumber(fit.edf)} />
          <Readout label="GCV" value={formatNumber(fit.gcv)} />
          <Readout label="RMSE vs truth" value={formatNumber(rmse)} />
        </>
      }
    >
      <div className="grid gap-2 md:grid-cols-2">
        <Heatmap
          x={G}
          y={G}
          z={fit.F}
          xLabel="x"
          yLabel="z"
          scale="diverging"
          range={[-2.5, 2.5]}
          valueLabel="f(x, z)"
          height={300}
          ariaLabel="Fitted surface"
        />
        <Heatmap
          x={G}
          y={G}
          z={fitA.fxz}
          xLabel="x"
          yLabel="z"
          scale="diverging"
          range={[-0.8, 0.8]}
          valueLabel="f₁₂(x, z)"
          height={300}
          ariaLabel="Fitted interaction"
        />
        {main('x')}
        {main('z')}
      </div>
    </Interactive>
  )
}
