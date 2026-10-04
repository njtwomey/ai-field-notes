import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  type AxisModel,
} from 'aifn-render'
import {
  evaluate,
  makeBasis,
  penalise,
  smooth,
  type Smoother,
} from '../../regression/nonlinear-regression/_shared/splines'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'
import { normalCdf } from 'aifn/numerics/special'

const mean = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

const N = 200
const NOISE = 0.4
const MAX_SWEEPS = 30
const KNOTS = Array.from({ length: 10 }, (_, i) => (i + 1) / 11)
const GRID = toFlat(linspace(0, 1, 101))
const TRUE: [(x: number) => number, (x: number) => number] = [
  (x) => Math.sin(2 * Math.PI * x),
  (x) => 4 * (x - 0.5) ** 2,
]

type Sweep = { coef: [number[], number[]]; shift: [number, number]; change: number }

/** Backfitting for y = α + f₁(x₁) + f₂(x₂) + ε with one penalised spline smoother per term. Records every sweep. */
function backfit(y: number[], S: [Smoother, Smoother]): Sweep[] {
  const alpha = mean(y)
  const f: [number[], number[]] = [Array(N).fill(0), Array(N).fill(0)]
  const coef: [number[], number[]] = [[], []]
  const shift: [number, number] = [0, 0]
  const sweeps: Sweep[] = []
  for (let s = 0; s < MAX_SWEEPS; s++) {
    let change = 0
    for (const j of [0, 1] as const) {
      const other = f[1 - j]
      const partial = y.map((yi, i) => yi - alpha - other[i])
      const fit = smooth(S[j], partial)
      // Centre each smooth to mean zero over the data so that the intercept is identifiable.
      const m = mean(fit.fitted)
      const next = fit.fitted.map((v) => v - m)
      change += next.reduce((acc, v, i) => acc + (v - f[j][i]) ** 2, 0)
      f[j] = next
      coef[j] = fit.coef
      shift[j] = m
    }
    sweeps.push({ coef: [[...coef[0]], [...coef[1]]], shift: [shift[0], shift[1]], change: Math.sqrt(change / N) })
  }
  return sweeps
}

export function AdditiveFit({ initialRho = 0 }: { initialRho?: number }) {
  const state = useFigureState({
    rho: slider(0, 0.95, initialRho, { step: 0.05, label: 'input correlation ρ' }),
    logLambda: slider(-8, 0, -3, { step: 0.1, label: 'log₁₀ λ', format: (v) => v.toFixed(1) }),
    sweep: slider(1, MAX_SWEEPS, MAX_SWEEPS, { step: 1, label: 'sweeps' }),
    seed: int(2, { ge: 0, label: 'seed' }),
  })
  const { rho, logLambda, sweep, seed } = state

  const data = useMemo(() => {
    const r = stream(seed)
    const z = Array.from({ length: N }, () => [normal(r), normal(r)])
    // Correlated latent normals mapped to [0, 1]: ρ controls how far x₂ is predictable from x₁ (concurvity).
    const x1 = z.map(([a]) => normalCdf(a))
    const x2 = z.map(([a, b]) => normalCdf(rho * a + Math.sqrt(1 - rho * rho) * b))
    const eps = Array.from({ length: N }, () => normal(r))
    const y = x1.map((a, i) => 1 + TRUE[0](a) + TRUE[1](x2[i]) + NOISE * eps[i])
    return { x: [x1, x2] as [number[], number[]], y }
  }, [rho, seed])

  const bases = useMemo(() => data.x.map((x) => makeBasis(x, KNOTS, 0, 1)), [data])
  const S = useMemo(() => bases.map((b) => penalise(b, 10 ** logLambda)) as [Smoother, Smoother], [bases, logLambda])
  const sweeps = useMemo(() => backfit(data.y, S), [data, S])
  const k = Math.min(sweep, MAX_SWEEPS) - 1
  const current = sweeps[k]
  const converged = sweeps.findIndex((s) => s.change < 1e-4) + 1
  const alpha = mean(data.y)

  const fitted = (j: 0 | 1, xs: number[]) => evaluate(S[j], current.coef[j], xs).map((v) => v - current.shift[j])
  const fAtData: [number[], number[]] = [fitted(0, data.x[0]), fitted(1, data.x[1])]
  const r2 = (() => {
    const pred = data.y.map((_, i) => alpha + fAtData[0][i] + fAtData[1][i])
    const rss = data.y.reduce((s, y, i) => s + (y - pred[i]) ** 2, 0)
    const tss = data.y.reduce((s, y) => s + (y - alpha) ** 2, 0)
    return 1 - rss / tss
  })()
  const xCorr = (() => {
    const [a, b] = data.x
    const ma = mean(a)
    const mb = mean(b)
    const sab = a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0)
    const saa = a.reduce((s, v) => s + (v - ma) ** 2, 0)
    const sbb = b.reduce((s, v) => s + (v - mb) ** 2, 0)
    return sab / Math.sqrt(saa * sbb)
  })()

  const axes: [AxisModel, AxisModel][] = [
    [useAxis({ label: 'x₁', range: [0, 1] }), useAxis({ label: 'f₁(x₁)', range: [-2.5, 2.5] })],
    [useAxis({ label: 'x₂', range: [0, 1] }), useAxis({ label: 'f₂(x₂)', range: [-2.5, 2.5] })],
  ]
  const panel = (j: 0 | 1) => {
    const other = fAtData[1 - j]
    const partial = data.y.map((y, i) => y - alpha - other[i])
    const trueMean = mean(data.x[j].map(TRUE[j]))
    return (
      <Plot x={axes[j][0]} y={axes[j][1]} height={260}>
        <Points name="partial residuals" x={data.x[j]} y={partial} muted />
        <Curve name="true effect" x={GRID} y={GRID.map((g) => TRUE[j](g) - trueMean)} slot={2} dashed />
        <Curve name={`f${j === 0 ? '₁' : '₂'}`} x={GRID} y={fitted(j, GRID)} slot={1} />
      </Plot>
    )
  }

  return (
    <Figure
      title="Backfitting an additive model"
      caption={
        <>
          Two hundred points from y = 1 + sin(2πx₁) + 4(x₂ − ½)² + noise. Each panel shows one fitted smooth, centred to
          mean zero, over its partial residuals: y minus the intercept and the other smooth. The sweep slider shows the
          fit after that many backfitting cycles. With independent inputs the first sweep is already close; as the
          correlation ρ between the inputs grows (concurvity), backfitting needs more sweeps and the two smooths trade
          signal between them.
        </>
      }
      state={state}
      readouts={
        <>
          <Readout label="corr(x₁, x₂)" value={formatNumber(xCorr)} />
          <Readout label="edf f₁, f₂" value={`${formatNumber(S[0].edf)}, ${formatNumber(S[1].edf)}`} />
          <Readout label="change in this sweep" value={formatNumber(current.change)} />
          <Readout label="sweeps to converge" value={converged > 0 ? String(converged) : `> ${MAX_SWEEPS}`} />
          <Readout label="R²" value={formatNumber(r2)} />
        </>
      }
    >
      <div className="grid gap-2 md:grid-cols-2">
        {panel(0)}
        {panel(1)}
      </div>
    </Figure>
  )
}
