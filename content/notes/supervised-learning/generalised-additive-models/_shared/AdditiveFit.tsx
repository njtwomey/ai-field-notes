import { useMemo, useState } from 'react'
import { Interactive, ParamButton, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { linspace, mean, rng } from '@/lib/math'
import { normalCdf } from '@/lib/math/special'
import {
  evaluate,
  makeBasis,
  penalise,
  smooth,
  type Smoother,
} from '../../regression/nonlinear-regression/_shared/splines'

const N = 200
const NOISE = 0.4
const MAX_SWEEPS = 30
const KNOTS = Array.from({ length: 10 }, (_, i) => (i + 1) / 11)
const GRID = linspace(0, 1, 101)
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
  const [rho, setRho] = useState(initialRho)
  const [logLambda, setLogLambda] = useState(-3)
  const [sweep, setSweep] = useState(MAX_SWEEPS)
  const [seed, setSeed] = useState(2)

  const data = useMemo(() => {
    const r = rng(seed)
    const z = Array.from({ length: N }, () => [r.normal(), r.normal()])
    // Correlated latent normals mapped to [0, 1]: ρ controls how far x₂ is predictable from x₁ (concurvity).
    const x1 = z.map(([a]) => normalCdf(a))
    const x2 = z.map(([a, b]) => normalCdf(rho * a + Math.sqrt(1 - rho * rho) * b))
    const eps = Array.from({ length: N }, () => r.normal())
    const y = x1.map((a, i) => 1 + TRUE[0](a) + TRUE[1](x2[i]) + NOISE * eps[i])
    return { x: [x1, x2] as [number[], number[]], y }
  }, [rho, seed])

  const bases = useMemo(() => data.x.map((x) => makeBasis(x, KNOTS, 0, 1)), [data])
  const S = useMemo(() => bases.map((b) => penalise(b, 10 ** logLambda)) as [Smoother, Smoother], [bases, logLambda])
  const sweeps = useMemo(() => backfit(data.y, S), [data, S])
  const k = Math.min(sweep, MAX_SWEEPS) - 1
  const state = sweeps[k]
  const converged = sweeps.findIndex((s) => s.change < 1e-4) + 1
  const alpha = mean(data.y)

  const fitted = (j: 0 | 1, xs: number[]) => evaluate(S[j], state.coef[j], xs).map((v) => v - state.shift[j])
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

  const panel = (j: 0 | 1) => {
    const other = fAtData[1 - j]
    const partial = data.y.map((y, i) => y - alpha - other[i])
    const trueMean = mean(data.x[j].map(TRUE[j]))
    const series: XYSeries[] = [
      { name: 'partial residuals', type: 'scatter', x: data.x[j], y: partial, muted: true },
      { name: 'true effect', type: 'line', x: GRID, y: GRID.map((g) => TRUE[j](g) - trueMean), slot: 2, dashed: true },
      { name: `f${j === 0 ? '₁' : '₂'}`, type: 'line', x: GRID, y: fitted(j, GRID), slot: 1 },
    ]
    return (
      <XYChart
        series={series}
        xRange={[0, 1]}
        yRange={[-2.5, 2.5]}
        xLabel={j === 0 ? 'x₁' : 'x₂'}
        yLabel={j === 0 ? 'f₁(x₁)' : 'f₂(x₂)'}
        height={260}
      />
    )
  }

  return (
    <Interactive
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
      controls={
        <>
          <ParamSlider label="input correlation ρ" value={rho} onChange={setRho} min={0} max={0.95} step={0.05} />
          <ParamSlider
            label="log₁₀ λ"
            value={logLambda}
            onChange={setLogLambda}
            min={-8}
            max={0}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          <ParamSlider label="sweeps" value={sweep} onChange={setSweep} min={1} max={MAX_SWEEPS} step={1} withArrows />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="corr(x₁, x₂)" value={formatNumber(xCorr)} />
          <Readout label="edf f₁, f₂" value={`${formatNumber(S[0].edf)}, ${formatNumber(S[1].edf)}`} />
          <Readout label="change in this sweep" value={formatNumber(state.change)} />
          <Readout label="sweeps to converge" value={converged > 0 ? String(converged) : `> ${MAX_SWEEPS}`} />
          <Readout label="R²" value={formatNumber(r2)} />
        </>
      }
    >
      <div className="grid gap-2 md:grid-cols-2">
        {panel(0)}
        {panel(1)}
      </div>
    </Interactive>
  )
}
